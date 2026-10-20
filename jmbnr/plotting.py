"""Result plots use a common desired-signal FFT scale, not an ACPR estimator."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {
    "no_dpd": "No DPD",
    "gmp_rls": "GMP--RLS",
    "donor_cnn": "GMP + residual CNN",
    "long_joint_cnn": "JMBNR",
    "rvtdcnn_158": "Reconstructed RVTDCNN",
}


def plot_spectra(reference, outputs, directory):
    def power(values):
        frames = np.asarray(values).reshape(-1, 16384)
        return np.mean(
            abs(np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)) ** 2, axis=0
        )

    scale = power(reference).max()
    frequency = np.fft.fftshift(np.fft.fftfreq(16384, 1 / 640e6)) / 1e6
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.plot(
        frequency,
        10 * np.log10(np.maximum(power(reference) / scale, 1e-16)),
        label="Desired signal",
        lw=0.8,
    )
    for key, values in outputs.items():
        axis.plot(
            frequency,
            10 * np.log10(np.maximum(power(values) / scale, 1e-16)),
            label=LABELS[key],
            lw=0.8,
        )
    axis.set(
        xlabel="Frequency (MHz)",
        ylabel="Mean FFT power (dB, common reference)",
        xlim=(-320, 320),
        ylim=(-65, 5),
    )
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(directory / "spectrum.png", dpi=200)
    fig.savefig(directory / "spectrum.pdf")
    plt.close(fig)


def write_table(results, directory):
    lines = [
        "| Method | Fixed-gain NMSE (dB) | Left ratio (dBc) | Right ratio (dBc) |",
        "|---|---:|---:|---:|",
    ]
    for name, m in results.items():
        lines.append(
            f"| {LABELS[name]} | {m['fixed']['fixed_gain_nmse_db']:.3f} | {m['legacy']['acpr_left_db']:.3f} | {m['legacy']['acpr_right_db']:.3f} |"
        )
    (directory / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
