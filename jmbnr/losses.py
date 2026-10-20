# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Nimrod Glazer and JMBNR contributors
"""Executed baseline and protected-refinement objectives.

These minibatch proxies differ from the full-capture reporting metrics.
"""

import torch
from .data import FS, BW
from .metrics import masks


def original_loss(output_iq, target_iq, baseline_iq, evm_weight=1.0, fs=FS, bw=BW):
    """Notebook equations on PA outputs; no synthetic PA hidden in the loss."""
    target = torch.complex(target_iq[..., 0], target_iq[..., 1])

    def components(values):
        z = torch.complex(values[..., 0], values[..., 1])
        gain = (z.conj() * target).sum(1, keepdim=True) / (abs(z) ** 2).sum(
            1, keepdim=True
        ).clamp_min(1e-12)
        z = z * gain
        f = torch.fft.fftshift(torch.fft.fft(z, dim=1, norm="ortho"), dim=1)
        return (z, f, abs(f) ** 2)

    z, f, power = components(output_iq)
    _, _, baseline_power = components(baseline_iq.detach())
    ref = torch.fft.fftshift(torch.fft.fft(target, dim=1, norm="ortho"), dim=1)
    main, lower, upper, far = [
        torch.as_tensor(m, device=target.device) for m in masks(target.shape[1], fs, bw)
    ]
    energy = (abs(ref[:, main]) ** 2).sum(1).clamp_min(1e-12)
    error = abs(f - ref) ** 2

    def robust(error_power):
        values = torch.sqrt((error_power / energy).clamp_min(1e-24))
        return (
            values.mean()
            + 0.5 * torch.topk(values, max(1, (len(values) + 3) // 4)).values.mean()
        )

    evm = torch.sqrt(
        (abs(z - target) ** 2).mean() / (abs(target) ** 2).mean().clamp_min(1e-12)
    )
    ib = robust(error[:, main].sum(1))
    adj = robust(0.5 * (error[:, lower].sum(1) + error[:, upper].sum(1)))
    oob = robust(error[:, far].sum(1))
    groups = torch.round(
        10 * torch.log10((abs(target) ** 2).mean(1).clamp_min(1e-24))
    ).long()
    violations = []
    for group in torch.unique(groups):
        select = groups == group
        for region in (lower | upper, far):
            ratio = power[select][:, region].sum() / power[select][
                :, main
            ].sum().clamp_min(1e-24)
            base_ratio = baseline_power[select][:, region].sum() / baseline_power[
                select
            ][:, main].sum().clamp_min(1e-24)
            delta = 10 * torch.log10(
                ratio.clamp_min(1e-24) / base_ratio.clamp_min(1e-24)
            )
            violations.append(torch.relu(delta + 0.1).square())
    violation = torch.stack(violations)
    return (
        evm_weight * evm
        + ib
        + 4 * adj
        + 2 * oob
        + 0.01 * (violation.mean() + violation.max())
    )


def fixed_gain_loss(output_iq, target_iq):
    """RMS waveform error at the frozen PA's train-calibrated gain, without alignment."""
    return torch.sqrt(
        (output_iq - target_iq).square().sum(-1).mean().clamp_min(1e-24)
        / target_iq.square().sum(-1).mean().clamp_min(1e-12)
    )


def balanced_region_loss(output_iq, target_iq, baseline_iq, fs=FS, bw=BW):
    """Fixed-gain inband error and separate near-adjacent residual penalties.

    Short training windows truncate OFDM symbols. Subtract the desired waveform
    before spectral analysis so its window-edge leakage is not treated as PA
    distortion. Hann-window the adjacent residual; native validation still uses
    the unchanged output-power legacy ACPR, without subtracting the reference.
    """
    target = torch.complex(target_iq[..., 0], target_iq[..., 1])
    output = torch.complex(output_iq[..., 0], output_iq[..., 1])
    baseline = torch.complex(baseline_iq[..., 0], baseline_iq[..., 1]).detach()
    n = target.shape[1]
    frequency = torch.fft.fftshift(torch.fft.fftfreq(n, 1 / fs, device=target.device))
    main = (frequency >= -bw / 2) & (frequency < bw / 2)
    left = (frequency >= -bw / 2 - bw / 4) & (frequency < -bw / 2)
    right = (frequency >= bw / 2) & (frequency < bw / 2 + bw / 4)
    window = torch.hann_window(
        n, periodic=True, device=target.device, dtype=target.real.dtype
    )
    energy = abs(target).square().sum(1).clamp_min(1e-12)

    def regional_error(z):
        error = z - target
        f = torch.fft.fftshift(torch.fft.fft(error, dim=1, norm="ortho"), dim=1)
        w = torch.fft.fftshift(
            torch.fft.fft(error * window, dim=1, norm="ortho"), dim=1
        )
        return [
            abs(f[:, main]).square().sum(1),
            abs(w[:, left]).square().sum(1) / window.square().mean(),
            abs(w[:, right]).square().sum(1) / window.square().mean(),
        ]

    errors = regional_error(output)
    controls = regional_error(baseline)
    rms = [torch.sqrt((e / energy).clamp_min(1e-24)).mean() for e in errors]
    groups = torch.round(
        10 * torch.log10(abs(target).square().mean(1).clamp_min(1e-24))
    ).long()
    penalties = []
    for group in torch.unique(groups):
        select = groups == group
        for e, base, margin in zip(errors, controls, (0.0, 0.1, 0.1)):
            delta = 10 * torch.log10(
                e[select].sum().clamp_min(1e-24) / base[select].sum().clamp_min(1e-24)
            )
            penalties.append(torch.relu(delta + margin).square())
    penalties = torch.stack(penalties)
    return (
        2 * rms[0]
        + 4 * rms[1]
        + 4 * rms[2]
        + 0.01 * (penalties.mean() + penalties.max())
    )


def components(prediction, target):
    """Fixed-gain errors; amplitude is explicitly separated from phase error."""
    y = torch.complex(prediction[..., 0], prediction[..., 1])
    x = torch.complex(target[..., 0], target[..., 1])
    error = y - x
    n = x.shape[1]
    energy = x.abs().square().mean().clamp_min(1e-12)
    f = torch.fft.fftshift(torch.fft.fftfreq(n, 1 / FS, device=x.device))
    main = (f >= -BW / 2) & (f < BW / 2)
    left = (f >= -BW / 2 - BW / 4) & (f < -BW / 2)
    right = (f >= BW / 2) & (f < BW / 2 + BW / 4)
    broad = (f >= -2 * BW) & (f < -BW) | (f >= BW) & (f < 2 * BW)
    xf = torch.fft.fftshift(torch.fft.fft(x, dim=1, norm="ortho"), dim=1)
    ef = torch.fft.fftshift(torch.fft.fft(error, dim=1, norm="ortho"), dim=1)
    active = main & (
        xf.abs().square().mean(0) > xf.abs().square().mean(0).max() * 1e-08
    )
    main_energy = xf[:, main].abs().square().sum().clamp_min(1e-12)
    filtered_x = torch.fft.ifft(
        torch.fft.ifftshift(xf * main, dim=1), dim=1, norm="ortho"
    )
    filtered_e = torch.fft.ifft(
        torch.fft.ifftshift(ef * main, dim=1), dim=1, norm="ortho"
    )
    window = torch.hann_window(n, device=x.device, dtype=x.real.dtype)
    we = torch.fft.fftshift(torch.fft.fft(error * window, dim=1, norm="ortho"), dim=1)
    wy = torch.fft.fftshift(torch.fft.fft(y * window, dim=1, norm="ortho"), dim=1)
    return torch.stack(
        (
            error.abs().square().mean() / energy,
            (y.abs() - x.abs()).square().mean() / energy,
            ((filtered_x + filtered_e).abs() - filtered_x.abs()).square().sum()
            / main_energy,
            ef[:, active].abs().square().sum()
            / xf[:, active].abs().square().sum().clamp_min(1e-12),
            wy[:, broad].abs().square().sum()
            / wy[:, main].abs().square().sum().clamp_min(1e-12),
            we[:, left].abs().square().sum()
            / (n * len(x) * window.square().mean() * energy),
            we[:, right].abs().square().sum()
            / (n * len(x) * window.square().mean() * energy),
        )
    )


def refinement_loss(prediction, target, donor, strength=25.0, amplitude_weight=0.25):
    values = components(prediction, target)
    controls = components(donor.detach(), target).clamp_min(1e-12)
    ratios = values / controls
    violation = torch.relu(ratios[:5] - 1.0)
    adjacent = ratios[5:].mean() + 0.5 * ratios[5:].max()
    loss = (
        adjacent
        + amplitude_weight * ratios[1:3].sum()
        + 0.1 * (ratios[0] + ratios[3])
        + 2.0 * violation.sum()
        + strength * violation.square().sum()
    )
    return (loss, ratios.detach())


def far_components(prediction):
    """Raw output power proxies for the two previously unguarded metrics."""
    y = torch.complex(prediction[..., 0], prediction[..., 1])
    n = y.shape[1]
    f = torch.fft.fftshift(torch.fft.fftfreq(n, 1 / FS, device=y.device))
    window = torch.hann_window(n, device=y.device, dtype=y.real.dtype)
    spectrum = torch.fft.fftshift(torch.fft.fft(y * window, dim=1, norm="ortho"), dim=1)
    power = spectrum.abs().square().sum(0)
    main = (f >= -BW / 2) & (f < BW / 2)
    legacy_far = abs(f) > BW / 2 + BW / 4
    notebook_gap = (abs(f) >= BW / 2) & (abs(f) < BW)
    energy = power[main].sum().clamp_min(1e-12)
    return torch.stack(
        (power[legacy_far].sum() / energy, power[notebook_gap].sum() / energy)
    )


def far_loss(prediction, donor, strength, train_margin_db):
    ratios = far_components(prediction) / far_components(donor.detach()).clamp_min(
        1e-12
    )
    ceiling = 10.0 ** (-train_margin_db / 10.0)
    violations = torch.relu(ratios / ceiling - 1.0)
    return (
        2.0 * violations.sum() + strength * violations.square().sum(),
        ratios.detach(),
    )
