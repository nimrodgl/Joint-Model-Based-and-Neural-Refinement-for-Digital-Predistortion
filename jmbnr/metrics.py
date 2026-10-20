"""Exact archived reporting metrics and validation guards, separate from losses."""

import numpy as np
from .data import FS, BW, NFFT
from . import opendpd_metrics as official_metrics


def masks(nfft=NFFT, fs=FS, bw=BW):
    frequency = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / fs))
    main = (frequency >= -bw / 2) & (frequency < bw / 2)
    lower = (frequency >= -2 * bw) & (frequency < -bw)
    upper = (frequency >= bw) & (frequency < 2 * bw)
    return (main, lower, upper, ~(main | lower | upper))


def aligned(reference, output):
    gain = (output.conj() * reference).sum(-1, keepdims=True) / (abs(output) ** 2).sum(
        -1, keepdims=True
    ).clip(1e-24)
    return output * gain


def spectral_metrics(reference, output):
    reference = reference.reshape(-1, NFFT)
    output = aligned(reference, output.reshape(-1, NFFT))
    a = np.fft.fftshift(np.fft.fft(reference, axis=1), axes=1)
    b = np.fft.fftshift(np.fft.fft(output, axis=1), axes=1)
    main, lower, upper, far = masks()
    power = abs(b) ** 2
    main_power = power[:, main].sum()
    return {
        "evm_db": float(
            10
            * np.log10(
                (abs(b[:, main] - a[:, main]) ** 2).sum() / (abs(a[:, main]) ** 2).sum()
            )
        ),
        "acpr_db": float(
            10
            * np.log10(
                0.5 * (power[:, lower].sum() + power[:, upper].sum()) / main_power
            )
        ),
        "oob_db": float(10 * np.log10(power[:, far].sum() / main_power)),
    }


def gain_metrics(reference, output):
    x = reference.reshape(-1, NFFT)
    y = output.reshape(-1, NFFT)
    energy = (abs(x) ** 2).sum(1).clip(1e-24)
    gain = (x.conj() * y).sum(1) / energy
    return {
        "output_gain_db": float(20 * np.log10(abs(gain).clip(1e-24)).mean()),
        "worst_frame_gain_error_db": float(
            abs(20 * np.log10(abs(gain).clip(1e-24))).max()
        ),
        "fixed_gain_nmse_db": float(
            10 * np.log10(((abs(y - x) ** 2).sum() / energy.sum()).clip(1e-24))
        ),
    }


def validation_result(
    reference,
    output,
    baseline,
    gain_tolerance_db=None,
    fixed_gain_degradation_db=0.5,
    balanced_selection=False,
):
    x = reference.reshape(-1, NFFT)
    y = output.reshape(-1, NFFT)
    b = baseline.reshape(-1, NFFT)
    levels = np.round(10 * np.log10(np.mean(abs(x) ** 2, axis=1))).astype(int)
    overall = spectral_metrics(x, y)
    base = spectral_metrics(x, b)
    deltas = []
    for level in np.unique(levels):
        selected = levels == level
        row, control = (
            spectral_metrics(x[selected], y[selected]),
            spectral_metrics(x[selected], b[selected]),
        )
        deltas.extend((row[key] - control[key] for key in ("acpr_db", "oob_db")))
    overall["worst_spectral_delta_db"] = float(max(deltas))
    overall["spectral_score"] = (
        overall["acpr_db"] - base["acpr_db"] + overall["oob_db"] - base["oob_db"]
    )
    overall["eligible"] = (
        max(deltas) <= -0.1
        and overall["acpr_db"] <= base["acpr_db"] - 0.1
        and (overall["oob_db"] <= base["oob_db"] - 0.1)
    )
    overall.update(gain_metrics(x, y))
    fixed_deltas = []
    for level in np.unique(levels):
        selected = levels == level
        fixed_deltas.append(
            gain_metrics(x[selected], y[selected])["fixed_gain_nmse_db"]
            - gain_metrics(x[selected], b[selected])["fixed_gain_nmse_db"]
        )
    overall["worst_fixed_gain_delta_db"] = float(max(fixed_deltas))
    if gain_tolerance_db is not None:
        overall["gain_eligible"] = (
            overall["worst_frame_gain_error_db"] <= gain_tolerance_db
            and max(fixed_deltas) <= fixed_gain_degradation_db
        )
        overall["eligible"] = overall["eligible"] and overall["gain_eligible"]
    if balanced_selection:
        fixed, control = (score(x, y), score(x, b))
        overall["fixed_gain_inband_rms_db"] = fixed["rms_active_bin_evm_db"]
        overall["fixed_gain_inband_delta_db"] = (
            fixed["rms_active_bin_evm_db"] - control["rms_active_bin_evm_db"]
        )
        for side in ("left", "right"):
            key = f"acpr_{side}_db"
            overall[key] = fixed[key]
            overall[f"acpr_{side}_delta_db"] = fixed[key] - control[key]
        overall["balanced_eligible"] = (
            overall["fixed_gain_inband_delta_db"] <= 0
            and overall["acpr_left_delta_db"] <= -0.1
            and (overall["acpr_right_delta_db"] <= -0.1)
        )
        overall["eligible"] = overall["eligible"] and overall["balanced_eligible"]
    return overall


def score(reference, output):
    """Official legacy metrics plus a separately named conventional RMS EVM."""
    a = np.stack((output.real, output.imag), -1).reshape(-1, NFFT, 2).astype(np.float32)
    b = (
        np.stack((reference.real, reference.imag), -1)
        .reshape(-1, NFFT, 2)
        .astype(np.float32)
    )
    left, right = official_metrics.ACLR(
        a, fs=FS, nperseg=NFFT, bw_main_ch=BW, n_sub_ch=4
    )
    fa = np.fft.fftshift(np.fft.fft(output.reshape(-1, NFFT), axis=1), axes=1)
    fb = np.fft.fftshift(np.fft.fft(reference.reshape(-1, NFFT), axis=1), axes=1)
    frequency = np.fft.fftshift(np.fft.fftfreq(NFFT, 1 / FS))
    main = abs(frequency) <= BW / 2
    active = main & (
        np.mean(abs(fb) ** 2, axis=0) > np.max(np.mean(abs(fb) ** 2, axis=0)) * 1e-08
    )
    evm = 10 * np.log10(
        np.sum(abs(fa[:, active] - fb[:, active]) ** 2)
        / np.sum(abs(fb[:, active]) ** 2)
    )
    power = abs(fa) ** 2
    far = abs(frequency) > BW / 2 + BW / 4
    return {
        "nmse_db": float(official_metrics.NMSE(a, b)),
        "legacy_spectral_evm_db": float(
            official_metrics.EVM(
                a, b, sample_rate=int(FS), bw_main_ch=BW, n_sub_ch=4, nperseg=NFFT
            )
        ),
        "rms_active_bin_evm_db": float(evm),
        "acpr_left_db": float(left),
        "acpr_right_db": float(right),
        "far_oob_db": float(10 * np.log10(power[:, far].sum() / power[:, main].sum())),
    }


def original_metrics(reference, output):
    return {
        "fixed": gain_metrics(reference, output),
        "legacy": score(reference, output),
        "notebook": spectral_metrics(reference, output),
    }


def metrics(reference, output):
    measured = original_metrics(reference, output)
    n = len(reference)
    f = np.fft.fftfreq(n, 1 / FS)
    main = (f >= -BW / 2) & (f < BW / 2)
    x = np.fft.ifft(np.fft.fft(reference) * main)
    y = np.fft.ifft(np.fft.fft(output) * main)
    measured["amplitude"] = {
        "envelope_nmse_db": float(
            10
            * np.log10(
                max(
                    np.mean((abs(output) - abs(reference)) ** 2)
                    / np.mean(abs(reference) ** 2),
                    1e-24,
                )
            )
        ),
        "main_envelope_nmse_db": float(
            10
            * np.log10(
                max(np.mean((abs(y) - abs(x)) ** 2) / np.mean(abs(x) ** 2), 1e-24)
            )
        ),
    }
    return measured


def selection_score(candidate, reference):
    """Minimize the worse relative left/right change; no test-based selection."""
    return max(
        (
            candidate["legacy"][k] - reference["legacy"][k]
            for k in ("acpr_left_db", "acpr_right_db")
        )
    )


def admissible(candidate, reference):
    return (
        candidate["fixed"]["worst_frame_gain_error_db"] <= 0.1
        and all(
            (
                candidate["amplitude"][key] <= reference["amplitude"][key] + 0.1
                for key in reference["amplitude"]
            )
        )
        and (
            candidate["fixed"]["fixed_gain_nmse_db"]
            <= reference["fixed"]["fixed_gain_nmse_db"] + 0.1
        )
        and (
            candidate["legacy"]["rms_active_bin_evm_db"]
            <= reference["legacy"]["rms_active_bin_evm_db"] + 0.1
        )
        and all(
            (
                candidate["legacy"][key] <= reference["legacy"][key] + 0.1
                for key in ("acpr_left_db", "acpr_right_db")
            )
        )
        and (candidate["notebook"]["acpr_db"] <= reference["notebook"]["acpr_db"] + 0.1)
    )


def far_admissible(values, reference, margin):
    return (
        values["legacy"]["far_oob_db"] <= reference["legacy"]["far_oob_db"] + margin
        and values["notebook"]["oob_db"] <= reference["notebook"]["oob_db"] + margin
    )


def amplitude_admissible(values, reference, margin=0.1):
    return all(
        (
            values["amplitude"][key] <= reference["amplitude"][key] + margin
            for key in reference["amplitude"]
        )
    )
