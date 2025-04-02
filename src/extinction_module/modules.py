import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.interpolate import UnivariateSpline
from pathlib import Path
from typing import Tuple

class ExtinctionModule(nn.Module):
    def __init__(
        self,
        wavelength_min: float = 0.3,    # microns
        wavelength_max: float = 5e1,    # microns
        num_wavelength: int = 10000,
        av_min: float = 0.0,
        av_max: float = 10.0,
        num_av: int = 10000,
        transition_width: float = 0.02,
        extinction_ccm89_path: str = None,
        extinction_mcclure_path: str = None,
        device: str = "cuda"
    ):
        super(ExtinctionModule, self).__init__()

        module_dir = Path(__file__).parent
        data_dir = module_dir.parent.parent / "data"
        if extinction_ccm89_path is None:
            extinction_ccm89_path = str(data_dir / "Mathis_CCM89_ext.txt")
        if extinction_mcclure_path is None:
            extinction_mcclure_path = str(data_dir / "extinc3_8.dat")

        self.device = torch.device(device)
        self.wavelength_min = wavelength_min
        self.wavelength_max = wavelength_max
        self.log_wavelength_min = np.log10(wavelength_min)
        self.log_wavelength_max = np.log10(wavelength_max)
        self.num_wavelength = num_wavelength
        self.av_min = av_min
        self.av_max = av_max
        self.num_av = num_av
        self.transition_width = transition_width
        self.extinction_ccm89_path = extinction_ccm89_path
        self.extinction_mcclure_path = extinction_mcclure_path

        self.A_ref_ccm89 = 3.55
        self.A_ref_mcclure = 7.75

        # Precompute wavelength grid (in log10-space)
        wavelength_grid = torch.log10(
            torch.logspace(
                start=self.log_wavelength_min,
                end=self.log_wavelength_max,
                steps=num_wavelength,
                dtype=torch.float32,
                device=self.device
            )
        )
        self.register_buffer('wavelength_grid', wavelength_grid)

        # Precompute Av grid
        av_grid = torch.linspace(
            start=av_min,
            end=av_max,
            steps=num_av,
            dtype=torch.float32,
            device=self.device
        )
        self.av_min = av_grid.min().item()
        self.av_max = av_grid.max().item()
        self.num_av = len(av_grid)
        self.register_buffer('av_grid', av_grid)

        # Precompute extinction-law grid (shape: [num_av, num_wavelength])
        self.extinction_law = self._compute_extinction_law_grid()

    def _compute_extinction_law_grid(self) -> torch.Tensor:
        try:
            ccm89_data = np.loadtxt(Path(self.extinction_ccm89_path), skiprows=8)
            mcclure_data = np.loadtxt(Path(self.extinction_mcclure_path))
        except Exception as e:
            raise FileNotFoundError(f"Error loading extinction data files: {e}")

        # Unpack extinction data
        wave_ccm89, ext_ccm89 = ccm89_data.T  
        wave_mcclure = mcclure_data[:, 0]
        ext_mcclure_low = mcclure_data[:, 1]   # for Av < 8
        ext_mcclure_high = mcclure_data[:, 2]    # for Av > 8

        # Use linear splines (degree=1)
        smooth = 0
        degree = 1
        wavelength_grid_np = 10 ** self.wavelength_grid.cpu().numpy()

        spline_ccm89 = UnivariateSpline(wave_ccm89, ext_ccm89, k=degree, s=smooth, ext='extrapolate')
        ext_ccm89_interp = spline_ccm89(wavelength_grid_np)

        spline_mcclure_low = UnivariateSpline(wave_mcclure, ext_mcclure_low, k=degree, s=smooth, ext='extrapolate')
        ext_mcclure_low_interp = spline_mcclure_low(wavelength_grid_np)

        spline_mcclure_high = UnivariateSpline(wave_mcclure, ext_mcclure_high, k=degree, s=smooth, ext='extrapolate')
        ext_mcclure_high_interp = spline_mcclure_high(wavelength_grid_np)

        # Convert to torch tensors
        ext_ccm89_interp = torch.tensor(ext_ccm89_interp, dtype=torch.float32, device=self.device)
        ext_mcclure_low_interp = torch.tensor(ext_mcclure_low_interp, dtype=torch.float32, device=self.device)
        ext_mcclure_high_interp = torch.tensor(ext_mcclure_high_interp, dtype=torch.float32, device=self.device)

        # Build the extinction-law grid: one row per Av value.
        av = self.av_grid.unsqueeze(1)  # (num_av, 1)
        half_transition = self.transition_width / 2.0

        # Define regions in Av space.
        cond1 = av <= (3.0 - half_transition)
        cond2 = (av > (3.0 - half_transition)) & (av <= (3.0 + half_transition))
        cond3 = (av > (3.0 + half_transition)) & (av <= (8.0 - half_transition))
        cond4 = (av > (8.0 - half_transition)) & (av <= (8.0 + half_transition))
        cond5 = av > (8.0 + half_transition)

        extinction_law = torch.ones(
            (self.av_grid.shape[0], self.wavelength_grid.shape[0]),
            dtype=torch.float32, 
            device=self.device
        )

        if cond1.any():
            extinction_law[cond1.squeeze(), :] = ext_ccm89_interp.unsqueeze(0) / self.A_ref_ccm89
        if cond2.any():
            av_cond2 = av[cond2].view(-1, 1)
            weight_mcclure_low = (av_cond2 - 3.0 + half_transition) / self.transition_width
            weight_ccm89 = 1.0 - weight_mcclure_low

            law_ccm89 = ext_ccm89_interp.unsqueeze(0) * weight_ccm89 / self.A_ref_ccm89
            law_mcclure_low = ext_mcclure_low_interp.unsqueeze(0) * weight_mcclure_low / self.A_ref_mcclure
            extinction_law[cond2.squeeze(), :] = law_ccm89 + law_mcclure_low
        if cond3.any():
            extinction_law[cond3.squeeze(), :] = ext_mcclure_low_interp.unsqueeze(0) / self.A_ref_mcclure
        if cond4.any():
            av_cond4 = av[cond4].view(-1, 1)
            weight_mcclure_high = (av_cond4 - 8.0 + half_transition) / self.transition_width
            weight_mcclure_low = 1.0 - weight_mcclure_high

            law_mcclure_low = ext_mcclure_low_interp.unsqueeze(0) * weight_mcclure_low / self.A_ref_mcclure
            law_mcclure_high = ext_mcclure_high_interp.unsqueeze(0) * weight_mcclure_high / self.A_ref_mcclure
            extinction_law[cond4.squeeze(), :] = law_mcclure_low + law_mcclure_high
        if cond5.any():
            extinction_law[cond5.squeeze(), :] = ext_mcclure_high_interp.unsqueeze(0) / self.A_ref_mcclure

        return extinction_law  # shape: (num_av, num_wavelength)

    def _validate_inputs(self, y: torch.Tensor, x: torch.Tensor, av: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, ...]]:
        # Validate y: must be (batch, n_wavelength) or (batch, n_wavelength, H, W)
        if not isinstance(y, torch.Tensor):
            raise TypeError("y must be a torch.Tensor")
        if y.ndim not in (2, 4):
            raise ValueError("y must have 2 or 4 dimensions: (batch, n_wavelength) or (batch, n_wavelength, H, W)")

        # Validate x: must be (batch, n_wavelength)
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim != 2:
            raise ValueError("x must have 2 dimensions: (batch, n_wavelength)")

        # Validate av: must be (batch, 1)
        if not isinstance(av, torch.Tensor):
            raise TypeError("av must be a torch.Tensor")
        if av.ndim != 2 or av.shape[1] != 1:
            raise ValueError("av must have 2 dimensions: (batch, 1)")

        batch_y = y.shape[0]
        batch_x = x.shape[0]
        batch_av = av.shape[0]

        if batch_x not in (1, batch_y):
            raise ValueError(f"Batch dimension of x ({batch_x}) must be either 1 or match that of y ({batch_y})")
        if batch_av != batch_y:
            raise ValueError(f"Batch dimension of av ({batch_av}) must match that of y ({batch_y})")

        # If x has batch dimension 1 but y has more, expand x.
        if batch_x == 1 and batch_y > 1:
            x = x.expand(batch_y, -1)

        # Ensure the wavelength dimension matches
        n_wave_y = y.shape[1]
        n_wave_x = x.shape[1]
        if n_wave_y != n_wave_x:
            raise ValueError(f"Mismatch in wavelength count: y has {n_wave_y} but x has {n_wave_x}.")

        # Move tensors to the correct device and type
        y = y.to(self.device, dtype=torch.float32)
        x = x.to(self.device, dtype=torch.float32)
        av = av.to(self.device, dtype=torch.float32)

        extra_shape = None
        if y.ndim == 4:
            extra_shape = y.shape[2:]

        return y, x, av, extra_shape

    def _compute_extinction_law(self, x: torch.Tensor, av: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_wave), av: (batch, 1)
        # Normalize wavelengths and Av to [-1, 1] for grid_sample.
        av_norm = (2.0 * (av - self.av_min) / (self.av_max - self.av_min)) - 1.0  # (batch, 1)
        wavelength_norm = (2.0 * (torch.log10(x) - self.log_wavelength_min) /
                           (self.log_wavelength_max - self.log_wavelength_min)) - 1.0  # (batch, n_wave)

        # Build grid for grid_sample: shape (batch, 1, n_wave, 2)
        grid_x = wavelength_norm.unsqueeze(1)  # (batch, 1, n_wave)
        grid_y = av_norm.unsqueeze(1).expand(-1, 1, x.shape[1])  # (batch, 1, n_wave)
        grid = torch.stack((grid_x, grid_y), dim=-1)  # (batch, 1, n_wave, 2)

        # Precomputed extinction law grid is (num_av, n_wave); add batch and channel dims.
        law = self.extinction_law.unsqueeze(0).unsqueeze(0)  # (1, 1, num_av, n_wave)
        batch_size = x.shape[0]
        law = law.expand(batch_size, -1, -1, -1)

        sampled = F.grid_sample(law, grid, mode='bilinear', padding_mode='border', align_corners=False)
        # sampled: (batch, 1, 1, n_wave) -> squeeze to (batch, n_wave)
        sampled = sampled.squeeze(1).squeeze(1)
        return sampled

    def compute_extinction_factors(self, x: torch.Tensor, av: torch.Tensor) -> torch.Tensor:
        sampled = self._compute_extinction_law(x, av)  # (batch, n_wave)
        extinction_factors_exp = 0.4 * sampled * av  # av: (batch, 1) broadcasts along n_wave
        extinction_factors = 10 ** extinction_factors_exp  # (batch, n_wave)
        return extinction_factors

    def forward(self, y: torch.Tensor, x: torch.Tensor, av: torch.Tensor) -> torch.Tensor:
        """
        Apply the extinction correction.

        y: Flux tensor of shape (batch, n_wavelengths) or (batch, n_wavelengths, H, W)
        x: Wavelength tensor of shape (batch, n_wavelengths)
        av: Extinction amount tensor of shape (batch, 1)
        """
        y, x, av, extra_shape = self._validate_inputs(y, x, av)
        extinction_factors = self.compute_extinction_factors(x, av)  # (batch, n_wave)

        # Apply extinction correction only for wavelengths < 40.
        mask = x < 40.0  # (batch, n_wave)
        if extra_shape is not None:
            # Expand mask and extinction_factors so that they broadcast with y.
            mask = mask.unsqueeze(-1).unsqueeze(-1)  # becomes (batch, n_wave, 1, 1)
            extinction_factors = extinction_factors.unsqueeze(-1).unsqueeze(-1)
        y_corr = torch.where(mask, y / extinction_factors, y)
        return y_corr
    

class ScaleFluxModule(nn.Module):
    def __init__(self, reference_distance: float = 1.0):
        super(ScaleFluxModule, self).__init__()
        self.reference_distance = reference_distance

    def forward(self, y: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
        """
        Scale the flux by (reference_distance / distance)².

        y: Flux tensor of shape (batch, n_wavelengths) or (batch, n_wavelengths, H, W)
        distance: Tensor of shape (batch, 1)
        """
        if not isinstance(y, torch.Tensor):
            raise TypeError("y must be a torch.Tensor")
        if not isinstance(distance, torch.Tensor):
            raise TypeError("distance must be a torch.Tensor")
        if distance.ndim != 2 or distance.shape[1] != 1:
            raise ValueError("distance must have shape (batch, 1)")
        if y.ndim not in (2, 4):
            raise ValueError("y must have 2 or 4 dimensions")
        if y.shape[0] != distance.shape[0]:
            raise ValueError("Batch dimension of y and distance must match.")

        # Ensure that distance broadcasts over wavelength and any extra dimensions.
        while distance.ndim < y.ndim:
            distance = distance.unsqueeze(-1)
        return y * (self.reference_distance ** 2) / (distance ** 2)


class ExtinctionScalingModel(nn.Module):
    def __init__(
        self, 
        extinction_ccm89_path: str = None,
        extinction_mcclure_path: str = None, 
        reference_distance: float = 1.0
    ):
        super(ExtinctionScalingModel, self).__init__()
        self.extinction = ExtinctionModule(
            extinction_ccm89_path=extinction_ccm89_path,
            extinction_mcclure_path=extinction_mcclure_path
        )
        self.scaling = ScaleFluxModule(reference_distance)

    def forward(self, y: torch.Tensor, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Compute the final corrected flux.

        y: Flux tensor of shape (batch, n_wavelengths) or (batch, n_wavelengths, H, W)
        x: Wavelength tensor of shape (batch, n_wavelengths)
        theta: Parameter tensor of shape (batch, n_param), where the second-to-last column is distance and the last is Av.
        """
        if not isinstance(theta, torch.Tensor):
            raise TypeError("theta must be a torch.Tensor")
        if theta.ndim != 2:
            raise ValueError("theta must have 2 dimensions: (batch, n_param)")
        if theta.shape[1] < 2:
            raise ValueError("theta must have at least 2 columns (distance and Av).")

        # Extract distance and Av from theta and reshape them to (batch, 1)
        distance = theta[:, -2].unsqueeze(1)
        av = theta[:, -1].unsqueeze(1)

        y_prime = self.extinction(y, x, av)
        y_prime = self.scaling(y_prime, distance)
        return y_prime