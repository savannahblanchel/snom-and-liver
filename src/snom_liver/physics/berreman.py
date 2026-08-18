"""Berreman 4x4 transfer-matrix backend.

This module keeps the useful part of the legacy FDM TMM path: arbitrary
anisotropic 3x3 dielectric tensors, multiple layers, and p/s reflection
coupling. Plotting, progress bars, global device state, and script-level
experiment settings are intentionally left out.
"""

from __future__ import annotations

import torch


class BerremanTMM:
    """Vectorized Berreman 4x4 transfer matrix for layered media.

    Parameters use the same units as the legacy FDM code:
    wavenumber in cm^-1, in-plane momentum `kx0` in cm^-1, and layer
    thicknesses in meters. `eps_tensors` is a list of `[3, 3, Nw]` complex
    dielectric tensors; the first and last layers should have zero thickness.
    """

    def __init__(self, qsd_threshold: float = 1e-10) -> None:
        self.qsd_threshold = qsd_threshold

    def forward(
        self,
        wavenumber: torch.Tensor,
        kx0: torch.Tensor,
        thicknesses_m: torch.Tensor | list[float],
        eps_tensors: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        w, kx0, ds, eps_list = self._prepare_inputs(wavenumber, kx0, thicknesses_m, eps_tensors)
        n_wavenumbers = w.shape[0]
        n_layers = len(ds)

        zeta = (kx0 / w).unsqueeze(0).unsqueeze(0)
        qs, gammas, amplitudes = self._process_layers(eps_list, zeta, n_wavenumbers)
        transfer = self._build_transfer_matrix(qs, amplitudes, w, ds, n_wavenumbers, n_layers)
        return self._reflection_transmission(transfer)

    def p_reflectivity(
        self,
        wavenumber: torch.Tensor,
        incidence_angle_rad: torch.Tensor,
        thicknesses_m: torch.Tensor | list[float],
        eps_tensors: list[torch.Tensor],
    ) -> torch.Tensor:
        eps_in = eps_tensors[0][0, 0, :].to(torch.complex128)
        kx0 = wavenumber.to(torch.complex128) * torch.sqrt(eps_in) * torch.sin(
            incidence_angle_rad.to(torch.complex128)
        )
        _, reflection = self.forward(wavenumber, kx0, thicknesses_m, eps_tensors)
        return reflection[0, :]

    @staticmethod
    def isotropic_tensor(epsilon: torch.Tensor) -> torch.Tensor:
        eps = epsilon.to(torch.complex128)
        tensor = torch.zeros((3, 3, eps.shape[0]), dtype=torch.complex128, device=eps.device)
        tensor[0, 0, :] = eps
        tensor[1, 1, :] = eps
        tensor[2, 2, :] = eps
        return tensor

    @staticmethod
    def _prepare_inputs(
        wavenumber: torch.Tensor,
        kx0: torch.Tensor,
        thicknesses_m: torch.Tensor | list[float],
        eps_tensors: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        w = wavenumber.to(torch.float64).contiguous()
        device = w.device
        if isinstance(kx0, (int, float)):
            kx = torch.full_like(w, float(kx0), dtype=torch.float64)
        else:
            kx = kx0.to(device=device).contiguous()
        if isinstance(thicknesses_m, list):
            ds = torch.tensor(thicknesses_m, dtype=torch.float64, device=device)
        else:
            ds = thicknesses_m.to(device=device, dtype=torch.float64).contiguous()
        eps_list = [eps.to(device=device, dtype=torch.complex128).contiguous() for eps in eps_tensors]
        if len(ds) != len(eps_list):
            raise ValueError("thicknesses_m and eps_tensors must have the same number of layers")
        return w, kx, ds, eps_list

    def _process_layers(
        self,
        eps_tensors: list[torch.Tensor],
        zeta: torch.Tensor,
        n_wavenumbers: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        device = zeta.device
        eye3 = torch.eye(3, dtype=torch.complex128, device=device)
        all_qs: list[torch.Tensor] = []
        all_gammas: list[torch.Tensor] = []
        all_amplitudes: list[torch.Tensor] = []

        for eps_layer in eps_tensors:
            matrix = torch.zeros((6, 6, n_wavenumbers), dtype=torch.complex128, device=device)
            matrix[:3, :3, :] = eps_layer
            matrix[3:6, 3:6, :] = eye3.unsqueeze(-1).expand(-1, -1, n_wavenumbers)

            a_matrix = self._a_matrix(matrix, zeta)
            s_matrix = self._s_matrix(matrix, a_matrix, zeta)
            delta = self._delta_matrix(s_matrix)
            eigvals, eigvecs = torch.linalg.eig(delta.permute(2, 0, 1))

            qs_layer = torch.zeros((4, n_wavenumbers), dtype=torch.complex128, device=device)
            gamma_layer = torch.zeros((4, 3, n_wavenumbers), dtype=torch.complex128, device=device)
            amplitude_layer = torch.zeros((4, 4, n_wavenumbers), dtype=torch.complex128, device=device)

            for index in range(n_wavenumbers):
                qs, gamma, amplitude = self._single_frequency_modes(
                    eigvals[index],
                    eigvecs[index],
                    eps_layer[:, :, index],
                    a_matrix[:, :, index],
                    zeta[0, 0, index],
                )
                qs_layer[:, index] = qs
                gamma_layer[:, :, index] = gamma
                amplitude_layer[:, :, index] = amplitude

            all_qs.append(qs_layer)
            all_gammas.append(gamma_layer)
            all_amplitudes.append(amplitude_layer)

        return all_qs, all_gammas, all_amplitudes

    @staticmethod
    def _a_matrix(matrix: torch.Tensor, zeta: torch.Tensor) -> torch.Tensor:
        device = matrix.device
        d = matrix[2, 2, :] * matrix[5, 5, :] - matrix[2, 5, :] * matrix[5, 2, :]
        zeta_flat = zeta[0, 0, :]
        a = torch.zeros((6, 6, matrix.shape[2]), dtype=torch.complex128, device=device)
        a[2, 0, :] = (matrix[5, 0, :] * matrix[2, 5, :] - matrix[2, 0, :] * matrix[5, 5, :]) / d
        a[2, 1, :] = ((matrix[5, 1, :] - zeta_flat) * matrix[2, 5, :] - matrix[2, 1, :] * matrix[5, 5, :]) / d
        a[2, 3, :] = (matrix[5, 3, :] * matrix[2, 5, :] - matrix[2, 3, :] * matrix[5, 5, :]) / d
        a[2, 4, :] = (matrix[5, 4, :] * matrix[2, 5, :] - (matrix[2, 4, :] + zeta_flat) * matrix[5, 5, :]) / d
        a[5, 0, :] = (matrix[5, 2, :] * matrix[2, 0, :] - matrix[2, 2, :] * matrix[5, 0, :]) / d
        a[5, 1, :] = (matrix[5, 2, :] * matrix[2, 1, :] - matrix[2, 2, :] * (matrix[5, 1, :] - zeta_flat)) / d
        a[5, 3, :] = (matrix[5, 2, :] * matrix[2, 3, :] - matrix[2, 2, :] * matrix[5, 3, :]) / d
        a[5, 4, :] = (matrix[5, 2, :] * (matrix[2, 4, :] + zeta_flat) - matrix[2, 2, :] * matrix[5, 4, :]) / d
        return a

    @staticmethod
    def _s_matrix(matrix: torch.Tensor, a: torch.Tensor, zeta: torch.Tensor) -> torch.Tensor:
        device = matrix.device
        zeta_flat = zeta[0, 0, :]
        s = torch.zeros((4, 4, matrix.shape[2]), dtype=torch.complex128, device=device)
        s[0, 0, :] = matrix[0, 0, :] + matrix[0, 2, :] * a[2, 0, :] + matrix[0, 5, :] * a[5, 0, :]
        s[0, 1, :] = matrix[0, 1, :] + matrix[0, 2, :] * a[2, 1, :] + matrix[0, 5, :] * a[5, 1, :]
        s[0, 2, :] = matrix[0, 3, :] + matrix[0, 2, :] * a[2, 3, :] + matrix[0, 5, :] * a[5, 3, :]
        s[0, 3, :] = matrix[0, 4, :] + matrix[0, 2, :] * a[2, 4, :] + matrix[0, 5, :] * a[5, 4, :]
        s[1, 0, :] = matrix[1, 0, :] + matrix[1, 2, :] * a[2, 0, :] + (matrix[1, 5, :] - zeta_flat) * a[5, 0, :]
        s[1, 1, :] = matrix[1, 1, :] + matrix[1, 2, :] * a[2, 1, :] + (matrix[1, 5, :] - zeta_flat) * a[5, 1, :]
        s[1, 2, :] = matrix[1, 3, :] + matrix[1, 2, :] * a[2, 3, :] + (matrix[1, 5, :] - zeta_flat) * a[5, 3, :]
        s[1, 3, :] = matrix[1, 4, :] + matrix[1, 2, :] * a[2, 4, :] + (matrix[1, 5, :] - zeta_flat) * a[5, 4, :]
        s[2, 0, :] = matrix[3, 0, :] + matrix[3, 2, :] * a[2, 0, :] + matrix[3, 5, :] * a[5, 0, :]
        s[2, 1, :] = matrix[3, 1, :] + matrix[3, 2, :] * a[2, 1, :] + matrix[3, 5, :] * a[5, 1, :]
        s[2, 2, :] = matrix[3, 3, :] + matrix[3, 2, :] * a[2, 3, :] + matrix[3, 5, :] * a[5, 3, :]
        s[2, 3, :] = matrix[3, 4, :] + matrix[3, 2, :] * a[2, 4, :] + matrix[3, 5, :] * a[5, 4, :]
        s[3, 0, :] = matrix[4, 0, :] + (matrix[4, 2, :] + zeta_flat) * a[2, 0, :] + matrix[4, 5, :] * a[5, 0, :]
        s[3, 1, :] = matrix[4, 1, :] + (matrix[4, 2, :] + zeta_flat) * a[2, 1, :] + matrix[4, 5, :] * a[5, 1, :]
        s[3, 2, :] = matrix[4, 3, :] + (matrix[4, 2, :] + zeta_flat) * a[2, 3, :] + matrix[4, 5, :] * a[5, 3, :]
        s[3, 3, :] = matrix[4, 4, :] + (matrix[4, 2, :] + zeta_flat) * a[2, 4, :] + matrix[4, 5, :] * a[5, 4, :]
        return s

    @staticmethod
    def _delta_matrix(s: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros_like(s)
        delta[0, 0, :] = s[3, 0, :]
        delta[0, 1, :] = s[3, 3, :]
        delta[0, 2, :] = s[3, 1, :]
        delta[0, 3, :] = -s[3, 2, :]
        delta[1, 0, :] = s[0, 0, :]
        delta[1, 1, :] = s[0, 3, :]
        delta[1, 2, :] = s[0, 1, :]
        delta[1, 3, :] = -s[0, 2, :]
        delta[2, 0, :] = -s[2, 0, :]
        delta[2, 1, :] = -s[2, 3, :]
        delta[2, 2, :] = -s[2, 1, :]
        delta[2, 3, :] = s[2, 2, :]
        delta[3, 0, :] = s[1, 0, :]
        delta[3, 1, :] = s[1, 3, :]
        delta[3, 2, :] = s[1, 1, :]
        delta[3, 3, :] = -s[1, 2, :]
        return delta

    def _single_frequency_modes(
        self,
        eigvals: torch.Tensor,
        eigvecs: torch.Tensor,
        eps_layer: torch.Tensor,
        a_matrix: torch.Tensor,
        zeta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = eigvals.device
        transmitted, reflected = self._split_modes(eigvals)
        transmitted = self._sort_pair_by_polarization(transmitted, eigvecs)
        reflected = self._sort_pair_by_polarization(reflected, eigvecs)
        order = torch.cat([transmitted, reflected])
        qs = eigvals[order]
        gamma = self._gamma_matrix(eps_layer, zeta, qs)
        amplitude = self._amplitude_matrix(gamma, qs, zeta)
        return qs.to(device), gamma, amplitude

    def _split_modes(self, eigvals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = eigvals.device
        transmitted: list[int] = []
        reflected: list[int] = []
        if torch.any(torch.abs(torch.imag(eigvals)) > self.qsd_threshold):
            for index in range(4):
                if torch.imag(eigvals[index]) >= 0 and len(transmitted) < 2:
                    transmitted.append(index)
                elif len(reflected) < 2:
                    reflected.append(index)
        else:
            for index in range(4):
                if torch.real(eigvals[index]) > 0 and len(transmitted) < 2:
                    transmitted.append(index)
                elif len(reflected) < 2:
                    reflected.append(index)

        if len(transmitted) != 2 or len(reflected) != 2:
            order = torch.argsort(torch.real(eigvals), descending=True).tolist()
            transmitted = order[:2]
            reflected = order[2:]

        return (
            torch.tensor(transmitted, dtype=torch.long, device=device),
            torch.tensor(reflected, dtype=torch.long, device=device),
        )

    @staticmethod
    def _sort_pair_by_polarization(pair: torch.Tensor, eigvecs: torch.Tensor) -> torch.Tensor:
        def p_content(mode_index: torch.Tensor) -> torch.Tensor:
            ex = eigvecs[0, mode_index]
            ey = eigvecs[2, mode_index]
            return torch.abs(ex) ** 2 / (torch.abs(ex) ** 2 + torch.abs(ey) ** 2 + 1e-10)

        first = p_content(pair[0])
        second = p_content(pair[1])
        if second > first:
            return torch.stack([pair[1], pair[0]])
        return pair

    def _gamma_matrix(self, eps_layer: torch.Tensor, zeta: torch.Tensor, qs: torch.Tensor) -> torch.Tensor:
        device = eps_layer.device
        gamma = torch.zeros((4, 3), dtype=torch.complex128, device=device)
        gamma[0, 0] = 1.0
        gamma[1, 1] = 1.0
        gamma[2, 0] = -1.0
        gamma[3, 1] = 1.0
        denom = eps_layer[2, 2] - zeta**2

        if torch.abs(qs[0] - qs[1]) < self.qsd_threshold:
            gamma[0, 2] = -(eps_layer[2, 0] + zeta * qs[0]) / denom
            gamma[1, 2] = -eps_layer[2, 1] / denom
        else:
            gamma12 = (eps_layer[1, 2] * (eps_layer[2, 0] + zeta * qs[0]) - eps_layer[1, 0] * denom) / (
                denom * (eps_layer[1, 1] - zeta**2 - qs[0] ** 2) - eps_layer[1, 2] * eps_layer[2, 1]
            )
            gamma13 = -(eps_layer[2, 0] + zeta * qs[0]) / denom - eps_layer[2, 1] * gamma12 / denom
            gamma21 = (eps_layer[2, 1] * (eps_layer[0, 2] + zeta * qs[1]) - eps_layer[0, 1] * denom) / (
                denom * (eps_layer[0, 0] - qs[1] ** 2)
                - (eps_layer[0, 2] + zeta * qs[1]) * (eps_layer[2, 0] + zeta * qs[1])
            )
            gamma23 = -(eps_layer[2, 0] + zeta * qs[1]) * gamma21 / denom - eps_layer[2, 1] / denom
            gamma[0, 1] = torch.nan_to_num(gamma12)
            gamma[0, 2] = torch.nan_to_num(gamma13)
            gamma[1, 0] = torch.nan_to_num(gamma21)
            gamma[1, 2] = torch.nan_to_num(gamma23)

        if torch.abs(qs[2] - qs[3]) < self.qsd_threshold:
            gamma[2, 2] = (eps_layer[2, 0] + zeta * qs[2]) / denom
            gamma[3, 2] = -eps_layer[2, 1] / denom
        else:
            gamma32 = (eps_layer[1, 0] * denom - eps_layer[1, 2] * (eps_layer[2, 0] + zeta * qs[2])) / (
                denom * (eps_layer[1, 1] - zeta**2 - qs[2] ** 2) - eps_layer[1, 2] * eps_layer[2, 1]
            )
            gamma33 = (eps_layer[2, 0] + zeta * qs[2]) / denom + eps_layer[2, 1] * gamma32 / denom
            gamma41 = (eps_layer[2, 1] * (eps_layer[0, 2] + zeta * qs[3]) - eps_layer[0, 1] * denom) / (
                denom * (eps_layer[0, 0] - qs[3] ** 2)
                - (eps_layer[0, 2] + zeta * qs[3]) * (eps_layer[2, 0] + zeta * qs[3])
            )
            gamma43 = -(eps_layer[2, 0] + zeta * qs[3]) * gamma41 / denom - eps_layer[2, 1] / denom
            gamma[2, 1] = torch.nan_to_num(gamma32)
            gamma[2, 2] = torch.nan_to_num(gamma33)
            gamma[3, 0] = torch.nan_to_num(gamma41)
            gamma[3, 2] = torch.nan_to_num(gamma43)

        norms = torch.clamp(torch.linalg.norm(gamma, dim=1, keepdim=True), min=1e-10)
        return gamma / norms

    @staticmethod
    def _amplitude_matrix(gamma: torch.Tensor, qs: torch.Tensor, zeta: torch.Tensor) -> torch.Tensor:
        amplitude = torch.zeros((4, 4), dtype=torch.complex128, device=gamma.device)
        amplitude[0:2, 0:4] = gamma[0:4, 0:2].T
        for index in range(4):
            amplitude[2, index] = qs[index] * gamma[index, 0] - zeta * gamma[index, 2]
            amplitude[3, index] = qs[index] * gamma[index, 1]
        return amplitude

    @staticmethod
    def _build_transfer_matrix(
        qs: list[torch.Tensor],
        amplitudes: list[torch.Tensor],
        wavenumber: torch.Tensor,
        thicknesses_m: torch.Tensor,
        n_wavenumbers: int,
        n_layers: int,
    ) -> torch.Tensor:
        device = wavenumber.device
        transfer = torch.eye(4, dtype=torch.complex128, device=device).unsqueeze(0).expand(n_wavenumbers, -1, -1)
        for layer in range(n_layers - 1, 1, -1):
            phase = -1j * 2 * torch.pi * wavenumber * 1e2 * qs[layer - 1] * thicknesses_m[layer - 1]
            propagation = torch.zeros((n_wavenumbers, 4, 4), dtype=torch.complex128, device=device)
            for mode in range(4):
                propagation[:, mode, mode] = torch.exp(phase[mode, :])
            amplitude = amplitudes[layer - 1].permute(2, 0, 1)
            layer_transfer = torch.bmm(torch.bmm(amplitude, propagation), torch.linalg.inv(amplitude))
            transfer = torch.bmm(layer_transfer, transfer)

        incident = amplitudes[0].permute(2, 0, 1)
        exit_layer = amplitudes[n_layers - 1].permute(2, 0, 1)
        transfer = torch.bmm(torch.bmm(torch.linalg.inv(incident), transfer), exit_layer)
        swap = torch.tensor(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=torch.complex128,
            device=device,
        )
        transfer = torch.bmm(torch.bmm(swap.unsqueeze(0).expand(n_wavenumbers, -1, -1), transfer), swap.unsqueeze(0).expand(n_wavenumbers, -1, -1))
        return transfer.permute(1, 2, 0)

    @staticmethod
    def _reflection_transmission(transfer: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = transfer.device
        n_wavenumbers = transfer.shape[2]
        denominator = transfer[0, 0, :] * transfer[2, 2, :] - transfer[0, 2, :] * transfer[2, 0, :]
        inv_denominator = 1.0 / (denominator + 1e-12)

        r_matrix = torch.zeros((2, 2, n_wavenumbers), dtype=torch.complex128, device=device)
        r_matrix[0, 0, :] = (transfer[1, 0, :] * transfer[2, 2, :] - transfer[1, 2, :] * transfer[2, 0, :]) * inv_denominator
        r_matrix[0, 1, :] = (transfer[3, 0, :] * transfer[2, 2, :] - transfer[3, 2, :] * transfer[2, 0, :]) * inv_denominator
        r_matrix[1, 0, :] = (transfer[0, 0, :] * transfer[1, 2, :] - transfer[1, 0, :] * transfer[0, 2, :]) * inv_denominator
        r_matrix[1, 1, :] = (transfer[0, 0, :] * transfer[3, 2, :] - transfer[3, 0, :] * transfer[0, 2, :]) * inv_denominator

        reflectance = torch.zeros((4, n_wavenumbers), dtype=torch.float64, device=device)
        reflectance[0, :] = torch.abs(r_matrix[0, 0, :]) ** 2
        reflectance[1, :] = torch.abs(r_matrix[1, 1, :]) ** 2
        reflectance[2, :] = torch.abs(r_matrix[1, 0, :]) ** 2
        reflectance[3, :] = torch.abs(r_matrix[0, 1, :]) ** 2

        reflection = torch.zeros((4, n_wavenumbers), dtype=torch.complex128, device=device)
        reflection[0, :] = r_matrix[0, 0, :]
        reflection[1, :] = r_matrix[0, 1, :]
        reflection[2, :] = r_matrix[1, 1, :]
        reflection[3, :] = r_matrix[1, 0, :]
        return reflectance, reflection
