"""Dielectric models used by the clean S-SNOM inversion pipeline."""

from __future__ import annotations

import torch


def _as_complex(value: float | complex, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.complex128, device=like.device)


def constant_epsilon(wavenumber: torch.Tensor, epsilon: float | complex) -> torch.Tensor:
    """Return a frequency-independent isotropic dielectric function."""
    return torch.ones_like(wavenumber, dtype=torch.complex128) * _as_complex(epsilon, wavenumber)


def lorentz_oscillator_epsilon(
    wavenumber: torch.Tensor,
    w_to: torch.Tensor,
    w_lo: torch.Tensor,
    gamma: torch.Tensor,
    eps_inf: torch.Tensor,
) -> torch.Tensor:
    """One-phonon Lorentz dielectric model in wavenumber units."""
    w = wavenumber.to(torch.float64)
    return eps_inf.to(torch.complex128) * (
        (w**2 - w_lo.to(torch.float64) ** 2 + 1j * gamma.to(torch.float64) * w)
        / (w**2 - w_to.to(torch.float64) ** 2 + 1j * gamma.to(torch.float64) * w)
    )


def drude_epsilon(
    wavenumber: torch.Tensor,
    plasma_wavenumber: torch.Tensor,
    gamma: torch.Tensor,
    eps_inf: torch.Tensor,
) -> torch.Tensor:
    """Simple Drude dielectric model in wavenumber units."""
    w = wavenumber.to(torch.float64)
    return eps_inf.to(torch.complex128) * (
        1 - plasma_wavenumber.to(torch.float64) ** 2 / (w**2 + 1j * gamma.to(torch.float64) * w)
    )


def multi_lorentz_epsilon(
    wavenumber: torch.Tensor,
    centers: torch.Tensor,
    strengths: torch.Tensor,
    gammas: torch.Tensor,
    eps_inf: torch.Tensor,
) -> torch.Tensor:
    """Broad multi-Lorentz dielectric model for biological spectra.

    The oscillator strengths are constrained positive in the fitting model. This
    keeps the dielectric response Kramers-Kronig-like while allowing multiple
    broad molecular absorption bands.
    """
    w = wavenumber.to(torch.float64).unsqueeze(0)
    centers = centers.to(torch.float64).unsqueeze(1)
    strengths = strengths.to(torch.float64).unsqueeze(1)
    gammas = gammas.to(torch.float64).unsqueeze(1)
    terms = strengths * centers**2 / (centers**2 - w**2 - 1j * gammas * w)
    return eps_inf.to(torch.complex128) + terms.sum(dim=0)


def reference_epsilon(wavenumber: torch.Tensor, material: str) -> torch.Tensor:
    """Reference dielectric function.

    Si is intentionally the default practical reference for this project because
    the measured sample spectra were exported relative to Si.
    """
    key = material.lower()
    if key in {"si", "silicon"}:
        return constant_epsilon(wavenumber, 13.0 + 0.0j)
    if key in {"au", "gold"}:
        # A coarse metal-like placeholder. Use measured Au optical constants
        # before making final quantitative claims with Au reference.
        return constant_epsilon(wavenumber, -1.0e5 + 1.0e5j)
    if key in {"air", "vac", "vacuum"}:
        return constant_epsilon(wavenumber, 1.0 + 0.0j)
    raise ValueError(f"Unsupported reference material: {material!r}")
