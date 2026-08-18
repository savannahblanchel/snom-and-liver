"""Clean S-SNOM forward model for the project-specific inversion pipeline.

The goal is a compact, inspectable baseline that can be compared with the
legacy FDM/PIML scripts without editing those original files.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .berreman import BerremanTMM
from .dielectric import lorentz_oscillator_epsilon, multi_lorentz_epsilon, reference_epsilon


@dataclass(frozen=True)
class TipParameters:
    length_m: float = 300e-9
    radius_m: float = 33e-9
    tapping_amplitude_m: float = 62.944e-9
    tapping_frequency_hz: float = 256100.066
    incidence_angle_deg: float = 60.0
    sample_thickness_m: float = 1000e-9
    substrate_material: str = "si"
    use_three_layer_reflectivity: bool = True
    reflectivity_backend: str = "fresnel"
    g_factor: float = 0.7
    g_phase: float = 0.06
    phase_offset: float = 0.0


@dataclass(frozen=True)
class SinglePhononParameters:
    w_to: float = 900.0
    w_lo: float = 1000.0
    gamma: float = 10.0
    eps_inf: float = 5.0


class SemiInfiniteSnomModel(nn.Module):
    """Single-phonon S-SNOM model normalized to a reference.

    By default the near-field interaction uses the sample dielectric response
    and the far-field propagation term uses an air / sample / substrate
    p-polarized reflectivity.
    """

    def __init__(
        self,
        initial: SinglePhononParameters | None = None,
        tip: TipParameters | None = None,
        reference_material: str = "si",
        bounds: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        super().__init__()
        initial = initial or SinglePhononParameters()
        self.tip = tip or TipParameters()
        self.reference_material = reference_material
        self.berreman_tmm = BerremanTMM()
        self.bounds = bounds or {
            "w_to": (690.0, 1700.0),
            "w_lo": (700.0, 1800.0),
            "gamma": (1.0, 100.0),
            "eps_inf": (1.0, 30.0),
            "g_factor": (0.1, 1.5),
            "g_phase": (-0.5, 0.5),
        }

        self.w_to = nn.Parameter(torch.tensor(initial.w_to, dtype=torch.float64))
        self.w_lo = nn.Parameter(torch.tensor(initial.w_lo, dtype=torch.float64))
        self.gamma = nn.Parameter(torch.tensor(initial.gamma, dtype=torch.float64))
        self.eps_inf = nn.Parameter(torch.tensor(initial.eps_inf, dtype=torch.float64))
        self.g_factor = nn.Parameter(torch.tensor(self.tip.g_factor, dtype=torch.float64))
        self.g_phase = nn.Parameter(torch.tensor(self.tip.g_phase, dtype=torch.float64))

    def clamp_parameters(self) -> None:
        with torch.no_grad():
            for name, (lower, upper) in self.bounds.items():
                getattr(self, name).clamp_(lower, upper)
            if self.w_lo <= self.w_to:
                self.w_lo.copy_(self.w_to + 1.0)

    def parameters_dict(self) -> dict[str, float]:
        return {
            "w_to": float(self.w_to.detach().cpu()),
            "w_lo": float(self.w_lo.detach().cpu()),
            "gamma": float(self.gamma.detach().cpu()),
            "eps_inf": float(self.eps_inf.detach().cpu()),
            "g_factor": float(self.g_factor.detach().cpu()),
            "g_phase": float(self.g_phase.detach().cpu()),
            "reference_material": self.reference_material,
        }

    def sample_epsilon(self, wavenumber: torch.Tensor) -> torch.Tensor:
        return lorentz_oscillator_epsilon(
            wavenumber=wavenumber,
            w_to=self.w_to,
            w_lo=self.w_lo,
            gamma=self.gamma,
            eps_inf=self.eps_inf,
        )

    @staticmethod
    def beta(epsilon: torch.Tensor) -> torch.Tensor:
        return (epsilon - 1.0) / (epsilon + 1.0)

    @staticmethod
    def p_polarized_reflectivity(epsilon: torch.Tensor, incidence_angle_rad: torch.Tensor) -> torch.Tensor:
        """Semi-infinite p-polarized Fresnel reflection coefficient.

        This keeps the far-field reflection coefficient separate from the
        near-field image-dipole beta, following the structure of the validated
        FDM `main.py` path without importing that long script.
        """
        eps_in = torch.ones_like(epsilon, dtype=torch.complex128)
        sin_theta = torch.sin(incidence_angle_rad).to(torch.complex128)
        kx_over_k0 = torch.sqrt(eps_in) * sin_theta
        kz_in = torch.sqrt(eps_in - kx_over_k0**2)
        kz_sample = torch.sqrt(epsilon - kx_over_k0**2)
        return (epsilon * kz_in - eps_in * kz_sample) / (epsilon * kz_in + eps_in * kz_sample)

    @staticmethod
    def three_layer_p_reflectivity(
        wavenumber: torch.Tensor,
        sample_epsilon: torch.Tensor,
        substrate_epsilon: torch.Tensor,
        sample_thickness_m: torch.Tensor,
        incidence_angle_rad: torch.Tensor,
    ) -> torch.Tensor:
        """Air / sample layer / substrate p-polarized reflection coefficient.

        Wavenumber is in cm^-1 and thickness is in meters. This is a compact
        isotropic 3-layer Fresnel/TMM expression for the far-field reflection
        coefficient used in the FDM propagation factor.
        """
        eps_air = torch.ones_like(sample_epsilon, dtype=torch.complex128)
        eps_layer = sample_epsilon.to(torch.complex128)
        eps_sub = substrate_epsilon.to(torch.complex128)
        sin_theta = torch.sin(incidence_angle_rad).to(torch.complex128)
        kx_over_k0 = torch.sqrt(eps_air) * sin_theta
        kz_air = torch.sqrt(eps_air - kx_over_k0**2)
        kz_layer = torch.sqrt(eps_layer - kx_over_k0**2)
        kz_sub = torch.sqrt(eps_sub - kx_over_k0**2)
        r01 = (eps_layer * kz_air - eps_air * kz_layer) / (eps_layer * kz_air + eps_air * kz_layer)
        r12 = (eps_sub * kz_layer - eps_layer * kz_sub) / (eps_sub * kz_layer + eps_layer * kz_sub)
        k0 = 2 * torch.pi * wavenumber.to(torch.float64) * 100.0
        phase = torch.exp(2j * k0.to(torch.complex128) * kz_layer * sample_thickness_m.to(torch.complex128))
        return (r01 + r12 * phase) / (1.0 + r01 * r12 * phase)

    @staticmethod
    def three_layer_berreman_p_reflectivity(
        wavenumber: torch.Tensor,
        sample_epsilon: torch.Tensor,
        substrate_epsilon: torch.Tensor,
        sample_thickness_m: torch.Tensor,
        incidence_angle_rad: torch.Tensor,
        tmm: BerremanTMM,
    ) -> torch.Tensor:
        eps_air = torch.ones_like(sample_epsilon, dtype=torch.complex128)
        eps_tensors = [
            tmm.isotropic_tensor(eps_air),
            tmm.isotropic_tensor(sample_epsilon),
            tmm.isotropic_tensor(substrate_epsilon),
        ]
        thicknesses = torch.stack(
            [
                torch.zeros_like(sample_thickness_m),
                sample_thickness_m,
                torch.zeros_like(sample_thickness_m),
            ]
        )
        return tmm.p_reflectivity(wavenumber, incidence_angle_rad, thicknesses, eps_tensors)

    def _demodulated_signal(self, wavenumber: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        tip = self.tip
        beta = self.beta(epsilon)

        device = wavenumber.device
        dtype = torch.float64
        length = torch.tensor(tip.length_m, dtype=dtype, device=device)
        radius = torch.tensor(tip.radius_m, dtype=dtype, device=device)
        z_amp = torch.tensor(tip.tapping_amplitude_m, dtype=dtype, device=device)
        freq = torch.tensor(tip.tapping_frequency_hz, dtype=dtype, device=device)
        incidence_angle = torch.tensor(tip.incidence_angle_deg * torch.pi / 180, dtype=dtype, device=device)
        sample_thickness = torch.tensor(tip.sample_thickness_m, dtype=dtype, device=device)
        g = self.g_factor.to(torch.complex128) * torch.exp(1j * self.g_phase.to(torch.complex128))
        if tip.use_three_layer_reflectivity:
            substrate_epsilon = reference_epsilon(wavenumber, tip.substrate_material)
            if tip.reflectivity_backend == "berreman":
                reflectivity = self.three_layer_berreman_p_reflectivity(
                    wavenumber,
                    epsilon,
                    substrate_epsilon,
                    sample_thickness,
                    incidence_angle,
                    self.berreman_tmm,
                )
            elif tip.reflectivity_backend == "fresnel":
                reflectivity = self.three_layer_p_reflectivity(
                    wavenumber,
                    epsilon,
                    substrate_epsilon,
                    sample_thickness,
                    incidence_angle,
                )
            else:
                raise ValueError(f"Unsupported reflectivity backend: {tip.reflectivity_backend!r}")
        else:
            reflectivity = self.p_polarized_reflectivity(epsilon, incidence_angle)

        omega = 2 * torch.pi * freq
        period = 1 / freq
        t = torch.linspace(-period / 2, period / 2, 501, dtype=dtype, device=device)
        height = z_amp * (1 + torch.cos(omega * t))
        e_const = torch.tensor(2.718281828, dtype=dtype, device=device)

        p1 = radius**2 * length * (2 * length / radius + torch.log(radius / (4 * e_const * length)))
        p1 = p1 / torch.log(4 * length / (e_const**2))

        b = beta.unsqueeze(-1)
        h = height.unsqueeze(0)
        numerator = b * (g - (radius + h) / length) * torch.log(4 * length / (4 * h + 3 * radius))
        denominator = torch.log(4 * length / radius) - b * (
            g - (3 * radius + 4 * h) / (4 * length)
        ) * torch.log(2 * length / (2 * h + radius))
        polarizability = p1 * (2 + numerator / denominator)

        harmonic = torch.exp(-1j * 2 * omega * t).unsqueeze(0)
        s2 = omega * torch.trapz(polarizability * harmonic, t, dim=1) / period
        propagation = torch.exp(-1j * 200 * torch.pi * wavenumber * z_amp * 0.5)
        return s2 * (1 + propagation * reflectivity) ** 2

    def forward(self, wavenumber: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.clamp_parameters()
        w = wavenumber.to(torch.float64)
        sample_signal = self._demodulated_signal(w, self.sample_epsilon(w))
        ref_signal = self._demodulated_signal(w, reference_epsilon(w, self.reference_material))
        normalized = sample_signal / ref_signal
        phase = torch.angle(normalized) + self.tip.phase_offset
        phase = torch.atan2(torch.sin(phase), torch.cos(phase))
        amplitude = torch.abs(normalized)
        return phase, amplitude


class MultiLorentzSnomModel(SemiInfiniteSnomModel):
    """S-SNOM model with multiple broad Lorentz oscillators."""

    def __init__(
        self,
        n_oscillators: int = 3,
        centers: list[float] | None = None,
        strengths: list[float] | None = None,
        gammas: list[float] | None = None,
        eps_inf: float = 2.5,
        tip: TipParameters | None = None,
        reference_material: str = "si",
        bounds: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.tip = tip or TipParameters()
        self.reference_material = reference_material
        self.berreman_tmm = BerremanTMM()
        self.n_oscillators = n_oscillators
        centers = centers or [900.0, 1200.0, 1450.0][:n_oscillators]
        strengths = strengths or [1.0] * n_oscillators
        gammas = gammas or [80.0] * n_oscillators
        if not (len(centers) == len(strengths) == len(gammas) == n_oscillators):
            raise ValueError("centers, strengths, and gammas must match n_oscillators")

        self.bounds = bounds or {
            "centers": (700.0, 1800.0),
            "strengths": (0.0, 30.0),
            "gammas": (10.0, 300.0),
            "eps_inf": (1.0, 30.0),
            "g_factor": (0.1, 1.5),
            "g_phase": (-0.5, 0.5),
        }

        self.centers = nn.Parameter(torch.tensor(centers, dtype=torch.float64))
        self.strengths = nn.Parameter(torch.tensor(strengths, dtype=torch.float64))
        self.gammas = nn.Parameter(torch.tensor(gammas, dtype=torch.float64))
        self.eps_inf = nn.Parameter(torch.tensor(eps_inf, dtype=torch.float64))
        self.g_factor = nn.Parameter(torch.tensor(self.tip.g_factor, dtype=torch.float64))
        self.g_phase = nn.Parameter(torch.tensor(self.tip.g_phase, dtype=torch.float64))

    def clamp_parameters(self) -> None:
        with torch.no_grad():
            self.centers.clamp_(*self.bounds["centers"])
            self.strengths.clamp_(*self.bounds["strengths"])
            self.gammas.clamp_(*self.bounds["gammas"])
            self.eps_inf.clamp_(*self.bounds["eps_inf"])
            self.g_factor.clamp_(*self.bounds["g_factor"])
            self.g_phase.clamp_(*self.bounds["g_phase"])

    def parameters_dict(self) -> dict[str, float | list[float]]:
        return {
            "centers": [float(v) for v in self.centers.detach().cpu()],
            "strengths": [float(v) for v in self.strengths.detach().cpu()],
            "gammas": [float(v) for v in self.gammas.detach().cpu()],
            "eps_inf": float(self.eps_inf.detach().cpu()),
            "g_factor": float(self.g_factor.detach().cpu()),
            "g_phase": float(self.g_phase.detach().cpu()),
            "reference_material": self.reference_material,
        }

    def sample_epsilon(self, wavenumber: torch.Tensor) -> torch.Tensor:
        return multi_lorentz_epsilon(
            wavenumber=wavenumber,
            centers=self.centers,
            strengths=self.strengths,
            gammas=self.gammas,
            eps_inf=self.eps_inf,
        )
