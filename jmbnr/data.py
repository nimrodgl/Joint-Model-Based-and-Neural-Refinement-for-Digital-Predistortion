# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Nimrod Glazer and JMBNR contributors
"""Original-capture IO and CPU inference helpers. Input values are unchanged."""

from pathlib import Path
import hashlib, json
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DEVICE = torch.device("cpu")
FS, BW, NFFT = 640e6, 160e6, 16384
CONTEXT = 32
torch.set_num_threads(2)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data():
    spec = read(DATA / "spec.json")
    assert (spec["input_signal_fs"], spec["bw_main_ch"], spec["nperseg"]) == (
        FS,
        BW,
        NFFT,
    )
    for name, expected in read(DATA / "checksums.json").items():
        assert sha(DATA / name) == expected, name
    result = {}
    for split in ("train", "val", "test"):
        result[split] = []
        for kind in ("input", "output"):
            a = np.loadtxt(DATA / f"{split}_{kind}.csv", delimiter=",", skiprows=1)
            assert a.shape[1] == 2 and np.isfinite(a).all() and len(a) % NFFT == 0
            result[split].append(a[:, 0] + 1j * a[:, 1])
    return result


def iq(z):
    return torch.as_tensor(
        np.stack((z.real, z.imag), -1), dtype=torch.float32, device=DEVICE
    )


def complex_np(x):
    a = x.detach().cpu().numpy()
    return a[..., 0] + 1j * a[..., 1]


def limit(x, maximum):
    r = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return x * torch.clamp(maximum / r.clamp_min(1e-12), max=1)


def evaluate_dpd(model, x, base, maximum):
    with torch.no_grad():
        return complex_np(
            limit(model(iq(x.reshape(1, -1)), iq(base.reshape(1, -1))), maximum)
        ).reshape(-1)


def complex_frames(signal, block):
    signal = np.asarray(signal)
    padding = [(CONTEXT, 1)] + [(0, 0)] * (signal.ndim - 1)
    padded = np.pad(signal, padding)
    starts = np.arange(0, len(signal) - block + 1, block)
    return torch.as_tensor(
        padded[starts[:, None] + np.arange(CONTEXT + block + 1)],
        dtype=torch.complex128,
        device=DEVICE,
    )


def frames(values, block):
    padded = np.pad(values, (CONTEXT, 1))
    starts = np.arange(0, len(values) - block + 1, block)
    return iq(padded[starts[:, None] + np.arange(CONTEXT + block + 1)])


def load_pa(path):
    from .models import PaSurrogate

    saved = torch.load(path, weights_only=True, map_location="cpu")
    model = PaSurrogate(saved["hidden_size"]).eval()
    model.load_state_dict(saved["state_dict"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def evaluate_pa(model, values):
    # One continuous capture; state is not restarted at scoring frame boundaries.
    with torch.no_grad():
        return complex_np(model(iq(np.asarray(values).reshape(1, -1))))
