# Extinction Module

The **Extinction Module** is a differentiable module designed for modeling and correcting interstellar extinction effects in astrophysical flux measurements in 1D spectra, 2D images or 3D spectral cubes.

---

## Overview

This project provides a PyTorch-based implementation of an extinction correction module. It incorporates two (three) extinction laws from Mathis (1990) and McClure (2009).
The module precomputes extinction-law grids over a range of wavelengths and extinction values (Av) to efficiently apply corrections via differentiable grid sampling.

---

## Features

- **Extinction Correction:**  
  Applies extinction correction to input fluxes based on wavelength and extinction (Av).

- **Distance Scaling:**  
  Scales the corrected flux according to a reference distance using inverse square law scaling.

- **Differentiable:**  
  The module is fully differentiable, allowing seamless integration into deep learning workflows and optimization. GPU (CUDA) acceleration is fully supported.

- **Configurable Parameters:**  
  Easily adjust wavelength ranges, extinction laws, and grid resolutions.

---

## Installation

Follow the installation instructions commented in the `pyproject.toml`.

---

## Usage

Import the module and create an instance of the model:

```python
import torch
from extinction_module.modules import ExtinctionScalingModel

# Instantiate the model
model = ExtinctionScalingModel(
    extinction_ccm89_path="path/to/Mathis_CCM89_ext.txt",
    extinction_mcclure_path="path/to/extinc3_8.dat",
    reference_distance=1.0
)

# Example inputs
image_size = 51
n_wavelength = 1000

# Dummy flux, wavelength, and parameter (theta) tensors
flux = torch.rand(image_size, image_size, n_wavelength)
wavelength = torch.linspace(0.35, 50.0, n_wavelength)
# theta: last two elements are [distance [pc], Av [-]]
theta = torch.tensor([[100.0, 2.0]] * batch_size, dtype=torch.float32)

# Compute the corrected flux
corrected_flux = model(flux, wavelength, theta)
```

### Module Breakdown

- **`ExtinctionModule`**:  
  Computes the extinction law grid and applies the extinction correction based on the input flux, wavelengths, and extinction values.

- **`ScaleFluxModule`**:  
  Scales the flux to a reference distance using the inverse-square law.

- **`ExtinctionScalingModel`**:  
  Combines both modules to provide a complete correction pipeline.

---

## Data Files

The extinction law data files are required for the module to function:

- **Mathis CCM89 Extinction Data** (`Mathis_CCM89_ext.txt`)
- **McClure Extinction Data** (`extinc3_8.dat`)

Place these files in the `data` directory as expected by the module, or provide custom paths during initialization.

---

## License

This project is licensed under the terms specified in the [LICENSE.txt](LICENSE.txt) file.

---

## References

- Mathis, J. S. (1990). *Interstellar dust and extinction*. In **The Evolution of the Interstellar Medium** (Vol. 12, pp. 63–77). [SAO/NASA ADS](https://ui.adsabs.harvard.edu/abs/1990ASPC...12...63M)  
- McClure, M. (2009). *Observational 5-20 μm Interstellar Extinction Curves Toward Star-Forming Regions Derived From Spitzer IRS Spectra*. The Astrophysical Journal Letters, 693(2), L81–L85. [DOI:10.1088/0004-637X/693/2/L81](https://ui.adsabs.harvard.edu/abs/2009ApJ...693L..81M)
