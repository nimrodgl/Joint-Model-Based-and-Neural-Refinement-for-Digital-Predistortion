# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Nimrod Glazer and JMBNR contributors
"""The 132-term GMP, offline ridge identification, and selected joint-update basis."""

from dataclasses import dataclass
import numpy as np
from .data import dump


@dataclass
class GmpConfig:
    Ka: int = 5
    La: int = 12
    Kb: int = 3
    Lb: int = 12
    Mb: int = 1
    Kc: int = 3
    Lc: int = 12
    Mc: int = 1

    @property
    def Ncoeff(self):
        return (
            self.Ka * self.La
            + self.Kb * self.Lb * self.Mb
            + self.Kc * self.Lc * self.Mc
        )


class GMP:
    def __init__(self, cfg=None, coeffs=None):
        self.cfg = cfg or GmpConfig()
        self.coeffs = np.asarray(
            coeffs if coeffs is not None else np.zeros(132), dtype=complex
        )

    def predistort(self, values):
        return self.basis(values) @ self.coeffs

    def basis(self, x):
        x = np.asarray(x, dtype=complex).reshape(-1)

        def shifted(offset):
            indices = np.arange(len(x)) + offset
            valid = (indices >= 0) & (indices < len(x))
            values = np.zeros_like(x)
            values[valid] = x[indices[valid]]
            return values

        cfg = self.cfg
        columns = [
            shifted(-l) * np.abs(shifted(-l)) ** k
            for k in range(cfg.Ka)
            for l in range(cfg.La)
        ]
        columns.extend(
            (
                shifted(-l) * np.abs(shifted(-l - m)) ** (k + 1)
                for k in range(cfg.Kb)
                for l in range(cfg.Lb)
                for m in range(1, cfg.Mb + 1)
            )
        )
        columns.extend(
            (
                shifted(-l) * np.abs(shifted(-l + m)) ** (k + 1)
                for k in range(cfg.Kc)
                for l in range(cfg.Lc)
                for m in range(1, cfg.Mc + 1)
            )
        )
        return np.column_stack(columns)


LEARNED_LAGS = (0, 5, 11)


def descriptors(cfg):
    columns = []
    index = 0
    for k in range(cfg.Ka):
        for lag in range(cfg.La):
            if k > 0 and lag in LEARNED_LAGS:
                columns.append(
                    dict(
                        index=index,
                        kind="aligned",
                        lag=lag,
                        power=k,
                        envelope_offset=-lag,
                    )
                )
            index += 1
    for kind, count, memory, orders in [
        ("lag", cfg.Kb, cfg.Lb, cfg.Mb),
        ("lead", cfg.Kc, cfg.Lc, cfg.Mc),
    ]:
        for k in range(count):
            for lag in range(memory):
                for offset in range(1, orders + 1):
                    if k < 2 and lag in LEARNED_LAGS and (offset == 1):
                        columns.append(
                            dict(
                                index=index,
                                kind=kind,
                                lag=lag,
                                power=k + 1,
                                envelope_offset=-lag
                                + (-offset if kind == "lag" else offset),
                            )
                        )
                    index += 1
    assert index == 132 and len(columns) == 24
    return columns


def selected_basis(signal, columns):
    signal = np.asarray(signal, dtype=np.complex128).reshape(-1)

    def shifted(offset):
        values = np.zeros_like(signal)
        index = np.arange(len(signal)) + offset
        valid = (index >= 0) & (index < len(signal))
        values[valid] = signal[index[valid]]
        return values

    return np.column_stack(
        [
            shifted(-c["lag"]) * abs(shifted(c["envelope_offset"])) ** c["power"]
            for c in columns
        ]
    )


def fit_rls(data, gain, out, cfg=None):
    """Full-training-capture inverse fitting; no validation/test adaptation."""
    cfg = cfg or GmpConfig()
    base = GMP(cfg, np.zeros(cfg.Ncoeff, complex))
    y = data["train"][1] / gain
    x = data["train"][0]
    phi = base.basis(y)
    warm = max(cfg.La, cfg.Lb + cfg.Mb, cfg.Lc + cfg.Mc)
    phi = phi[warm:-1]
    target = x[warm:-1]
    scale = np.sqrt(np.mean(abs(phi) ** 2, axis=0)).clip(1e-12)
    normalized = phi / scale
    ridge = 1e-06
    R = np.eye(cfg.Ncoeff, dtype=complex) * ridge
    b = np.zeros(cfg.Ncoeff, complex)
    for a, t in zip(np.array_split(normalized, 18), np.array_split(target, 18)):
        R += a.conj().T @ a
        b += a.conj().T @ t
    base.coeffs = np.linalg.solve(R, b) / scale
    report = {
        "coefficients_complex": cfg.Ncoeff,
        "gmp_config": vars(cfg),
        "forgetting_factor": 1.0,
        "solver": "Information-form block RLS; train-only normalized basis; ridge 1e-6",
        "basis_rank": int(np.linalg.matrix_rank(normalized)),
        "inverse_training_nmse_db": float(
            10
            * np.log10(
                np.sum(abs(phi @ base.coeffs - target) ** 2) / np.sum(abs(target) ** 2)
            )
        ),
        "lookahead_samples": 1,
        "calibration_samples": len(target),
        "covariance_condition": float(np.linalg.cond(R)),
    }
    dump(out / "rls_calibration.json", report)
    np.save(out / "rls_coefficients.npy", base.coeffs)
    return (base, report)
