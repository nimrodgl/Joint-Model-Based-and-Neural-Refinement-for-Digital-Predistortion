# SPDX-License-Identifier: MIT AND Apache-2.0
# OpenDPD DGRU portions: Yizhuo Wu, Chang Gao
# Upstream contact: yizhuo.wu@tudelft.nl, chang.gao@tudelft.nl
# Modified: standalone eager PA wrapper; other JMBNR models are original.
# Apache-2.0 covers adapted portions; MIT covers original contributions.
"""Readable network definitions; .pt files contain only saved weights/metadata."""

import torch
from torch import nn
from torch.nn import functional as F
from .data import limit

OPTIONS = ("continuation", "centered_residual")


def amplitude(squared_magnitude):
    zero = squared_magnitude == 0
    safe_squared = torch.where(
        zero, torch.ones_like(squared_magnitude), squared_magnitude
    )
    return torch.where(
        zero, torch.zeros_like(squared_magnitude), torch.sqrt(safe_squared)
    )


def phase_denominator(magnitude):
    return torch.where(magnitude == 0, torch.ones_like(magnitude), magnitude)


class PaSurrogate(nn.Module):
    """OpenDPD DGRU equations: six features, one GRU layer, hidden/readout layers.

    Adapted from OpenDPD (Apache-2.0); see THIRD_PARTY_NOTICES.md.
    The frozen checkpoint predicts training-gain-normalized PA output.
    """

    def __init__(self, hidden_size=24):
        super().__init__()
        self.rnn = nn.GRU(6, hidden_size, 1, batch_first=True)
        self.fc_out = nn.Linear(hidden_size + 6, 2)
        self.fc_hid = nn.Linear(hidden_size, hidden_size)

    def forward(self, values):
        i = values[..., 0:1]
        q = values[..., 1:2]
        radius = amplitude(i.pow(2) + q.pow(2))
        denominator = phase_denominator(radius)
        features = torch.cat(
            (i, q, radius, radius.pow(3), q / denominator, i / denominator), -1
        )
        hidden, _ = self.rnn(features, None)
        hidden = torch.relu(self.fc_hid(hidden))
        return self.fc_out(torch.cat((hidden, features), -1))


class ResidualCNN(nn.Module):
    """Seven-sample causal CNN: 4 -> 24 -> 12 -> 2, additive output / 16.

    Centered features are used only during refinement and folded for export.
    """

    def __init__(self, variant, scale):
        super().__init__()
        if variant not in OPTIONS or scale <= 0:
            raise ValueError("Invalid residual variant/scale.")
        self.variant, self.scale = (variant, float(scale))
        self.backbone = nn.Sequential(
            nn.Conv1d(4, 24, 5),
            nn.ReLU(),
            nn.ConstantPad1d((2, 0), 0.0),
            nn.Conv1d(24, 12, 3),
            nn.ReLU(),
        )
        self.head = nn.Conv1d(12, 2, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def initialize(self, state):
        self.load_state_dict(state)
        if self.variant != "continuation":
            with torch.no_grad():
                old = self.backbone[0].weight.clone()
                self.backbone[0].weight[:, :2] = old[:, :2] + old[:, 2:]
                self.backbone[0].weight[:, 2:] = old[:, 2:] * self.scale
        return self

    def forward(self, x, base):
        other = base if self.variant == "continuation" else (base - x) / self.scale
        features = F.pad(
            torch.cat((x, other), -1).transpose(1, 2), (4, 0)
        )
        return base + self.head(self.backbone(features)).transpose(1, 2) / 16.0


class RVTDCNN(nn.Module):

    def __init__(self, memory_depth=3):
        super().__init__()
        if memory_depth < 2:
            raise ValueError("The 3x3 kernel requires at least three time columns.")
        self.memory_depth = memory_depth
        self.conv = nn.Conv2d(1, 3, kernel_size=3, padding=0)
        self.hidden = nn.Linear(3 * 3 * (memory_depth - 1), 6)
        self.output = nn.Linear(6, 2)

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != 2:
            raise ValueError("Expected batch x samples x I/Q.")
        r = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        features = torch.cat((x, r, r.square(), r.pow(3)), dim=-1)
        padded = torch.cat(
            (features.new_zeros(x.shape[0], self.memory_depth, 5), features), dim=1
        )
        windows = padded.unfold(1, self.memory_depth + 1, 1).flip(-1)
        windows = windows.reshape(-1, 1, 5, self.memory_depth + 1)
        encoded = torch.tanh(self.conv(windows)).flatten(1)
        output = self.output(torch.tanh(self.hidden(encoded)))
        return output.reshape(x.shape[0], x.shape[1], 2)


def matched_training_generator(seed, device):
    """Replay the original seven-sample model's training permutations for either size."""
    device = torch.device(device)
    generator = torch.Generator(device=device)
    if device.type == "cpu":
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
            ResidualCNN("continuation", 1.0)
            state = torch.get_rng_state()
        generator.set_state(state)
    else:
        generator.manual_seed(seed)
    return generator


class JointSubset(nn.Module):

    def __init__(self, state, scale, maximum):
        super().__init__()
        self.cnn = ResidualCNN("centered_residual", scale).initialize(state)
        self.offsets = nn.Parameter(torch.zeros(24, 2, dtype=torch.float64))
        self.scale, self.maximum = (float(scale), float(maximum))

    def forward(self, x, raw, features):
        delta = features @ torch.complex(self.offsets[:, 0], self.offsets[:, 1])
        z = raw + delta
        base = limit(torch.stack((z.real, z.imag), -1).to(x.dtype), self.maximum)
        return (limit(self.cnn(x, base), self.maximum), delta)


def fold(state, scale):
    result = {key: value.detach().clone() for key, value in state.items()}
    w = result["backbone.0.weight"].clone()
    result["backbone.0.weight"][:, :2] = w[:, :2] - w[:, 2:] / scale
    result["backbone.0.weight"][:, 2:] = w[:, 2:] / scale
    return result
