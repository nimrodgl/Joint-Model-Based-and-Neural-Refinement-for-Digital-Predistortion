"""Recompute V05 table aggregates and plot the exact archived spectrum curves."""

import csv
import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import numpy as np

from .data import ROOT, dump, read


def metric(record, group, name):
    return record["metrics"][group][name]


def statistics(records, group, name):
    values = [metric(r, group, name) for r in records]
    return {
        "mean": float(np.mean(values)),
        "sample_sd": float(np.std(values, ddof=1)),
        "n": len(values),
    }


def table_data():
    """Aggregate the same cohorts and dB-domain means used by V05."""
    evidence = read(ROOT / "results/study_summaries.json")
    tables = {}
    tables["I"] = {
        "title": "Executed Adam settings",
        "columns": ["Setting", "Predistorted baseline", "JMBNR (proposed)"],
        "rows": [
            ["Updates; batch", "7,800; 16", "600; 4"],
            ["Scored/context samples", "512/32", "2,048/32"],
            ["CNN learning rate", "1e-3 -> 1e-4", "1e-4"],
            ["GMP learning rate", "frozen", "1e-5"],
            ["Rate change", "after 6,300", "x0.3 after 450"],
            ["Weight decay", "1e-6", "zero"],
            ["Gradient-norm cap", "1", "1, separately CNN/GMP"],
            ["Validation interval", "50 updates", "50 updates"],
        ],
        "scope": "Documented executed settings; context precedes the scored samples.",
    }
    names = {
        "conditioned_small": "Conditioned CNN (small)",
        "parallel_small": "Parallel CNN (small)",
        "identity_small": "Identity CNN (small)",
        "conditioned_large": "Conditioned CNN (large)",
        "parallel_large": "Parallel CNN (large)",
        "identity_large": "Identity CNN (large)",
        "r2tdnn_direct": "R2TDNN structure (adapt.)",
        "par_srtdnn_joint_direct": "Joint PAR structure (adapt.)",
        "fixed_only": "Large CNN: fixed-only loss",
        "original_fixed": "Large CNN: original+fixed",
        "full_gated": "Large CNN: full-gated selection",
        "random_head": "Large CNN: random head",
    }
    rows, details = [], []
    fields = [
        ("legacy", "rms_active_bin_evm_db"),
        ("legacy", "acpr_left_db"),
        ("legacy", "acpr_right_db"),
        ("notebook", "acpr_db"),
        ("notebook", "oob_db"),
    ]
    for method, label in names.items():
        group = [r for r in evidence["construction_jobs"] if r["method"] == method]
        assert len(group) == 5 and len({r["seed"] for r in group}) == 5
        capacity = group[0]["capacity"]
        nmse = statistics(group, "fixed", "fixed_gain_nmse_db")
        other = [statistics(group, *field) for field in fields]
        rows.append(
            [
                label,
                f"{capacity['trainable_real_parameters']:,}/{capacity['total_real_coefficients']:,}",
                f"{nmse['mean']:.2f} ± {nmse['sample_sd']:.2f}",
                *[f"{s['mean']:.2f}" for s in other],
            ]
        )
        details.append(
            {
                "method": method,
                "nmse": nmse,
                "other_metrics": dict(zip([k for _, k in fields], other)),
            }
        )
    tables["II"] = {
        "title": "Construction controls before JMBNR",
        "columns": [
            "Method",
            "Trainable/total",
            "NMSE (dB)",
            "Active (dB)",
            "Left (dBc)",
            "Right (dBc)",
            "Broad (dB)",
            "Gap (dB)",
        ],
        "rows": rows,
        "unrounded": details,
        "scope": "Five full-training seeds, primary PA fit; NMSE mean ± sample SD, other entries means.",
    }
    rows, details = [], []
    arms = [
        ("frozen_gmp", "CNN-only refinement", 1406),
        ("short_gmp", "JMBNR, short", 1454),
        ("guarded_long", "JMBNR (proposed)", 1454),
        ("no_explicit_amplitude", "JMBNR, no amp. weight", 1454),
    ]
    for arm, label, trainable in arms:
        if arm == "guarded_long":
            group = [
                r
                for r in evidence["refinement_results"]["results"]
                if r["far_constrained"]
            ]
        else:
            group = [r for r in evidence["ablation_results"] if r["ablation"] == arm]
        assert len(group) == 3
        left = [r["adjacent_gains_db"]["acpr_left_db"] for r in group]
        right = [r["adjacent_gains_db"]["acpr_right_db"] for r in group]
        passes = sum(
            r["test_base_safeguards"] and r["test_far_safeguards"] for r in group
        )
        rows.append(
            [
                label,
                f"{trainable:,}",
                f"{np.mean(left):.3f}",
                f"{np.mean(right):.3f}",
                f"{passes}/3",
            ]
        )
        details.append(
            {
                "arm": arm,
                "left_mean": float(np.mean(left)),
                "right_mean": float(np.mean(right)),
                "left_sample_sd": float(np.std(left, ddof=1)),
                "right_sample_sd": float(np.std(right, ddof=1)),
                "passes": passes,
            }
        )
    tables["III"] = {
        "title": "JMBNR refinement controls",
        "columns": ["Arm", "Trainable", "Left gain (dB)", "Right gain (dB)", "Passes"],
        "rows": rows,
        "unrounded": details,
        "scope": "Three adaptation orders sharing one pretrained baseline; failed arms retained.",
    }
    rows, details = [], []
    records = evidence["reference_comparison"]["records"]
    methods = [
        ("no_dpd", "No DPD"),
        ("gmp_rls", "GMP--RLS"),
        ("rvtdcnn_158", "RVTDCNN (158)"),
        ("donor_cnn", "Predistorted hybrid"),
        ("short_joint_cnn", "JMBNR, short"),
        ("long_joint_cnn", "JMBNR (proposed)"),
    ]
    fields = [
        ("fixed", "fixed_gain_nmse_db"),
        ("legacy", "acpr_left_db"),
        ("legacy", "acpr_right_db"),
    ]
    for method, label in methods:
        primary = next(
            r
            for r in records
            if r["method"] == method and r["evaluator"] == "training_pa"
        )
        alternate = [
            r
            for r in records
            if r["method"] == method and r["evaluator"] != "training_pa"
        ]
        assert len(alternate) == 3
        pairs = [
            (
                metric(primary, *field),
                float(np.mean([metric(r, *field) for r in alternate])),
            )
            for field in fields
        ]
        rows.append([label, *[f"{a:.2f}/{b:.2f}" for a, b in pairs]])
        details.append(
            {
                "method": method,
                "primary_and_alternate_mean": dict(zip([k for _, k in fields], pairs)),
            }
        )
    tables["IV"] = {
        "title": "Complete-system comparison",
        "columns": ["Method", "NMSE (dB)", "Left (dBc)", "Right (dBc)"],
        "rows": rows,
        "unrounded": details,
        "scope": "Fixed DPD seed 1234; primary / mean of three same-capture alternate fits; unequal budgets/capacities.",
    }
    rows, details = [], []
    for method, label in [
        ("conditioned_large", "I/Q CNN"),
        ("rvtdcnn_residual_matched", "RVT residual"),
    ]:
        group = [
            r for r in evidence["capacity_training_results"] if r["method"] == method
        ]
        assert len(group) == 5
        values = [statistics(group, *field) for field in fields]
        rows.append(
            [
                label,
                f"{group[0]['parameters']:,}",
                *[f"{s['mean']:.2f} ± {s['sample_sd']:.2f}" for s in values],
            ]
        )
        details.append(
            {"method": method, "statistics": dict(zip([k for _, k in fields], values))}
        )
    tables["V"] = {
        "title": "Similar-capacity residual comparison before JMBNR",
        "columns": ["Residual", "Weights", "NMSE (dB)", "Left (dBc)", "Right (dBc)"],
        "rows": rows,
        "unrounded": details,
        "scope": "Five paired construction seeds, primary PA fit; mean ± sample SD; 264 additional frozen GMP real coefficients each.",
    }
    return tables


def export_table(number, table, output):
    stem = "table_" + number
    with (output / (stem + ".csv")).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(table["columns"])
        writer.writerows(table["rows"])
    text = [
        "# Table " + number + ": " + table["title"],
        "",
        table["scope"],
        "",
        "| " + " | ".join(table["columns"]) + " |",
        "|" + "---|" * len(table["columns"]),
    ]
    text += ["| " + " | ".join(row) + " |" for row in table["rows"]]
    (output / (stem + ".md")).write_text("\n".join(text) + "\n", encoding="utf-8")
    lengths = np.array(
        [
            max(len(str(row[i])) for row in [table["columns"], *table["rows"]]) + 3
            for i in range(len(table["columns"]))
        ],
        dtype=float,
    )
    fig, axis = plt.subplots(
        figsize=(max(9, len(lengths) * 1.8), 1.5 + 0.34 * len(table["rows"]))
    )
    axis.axis("off")
    rendered = axis.table(
        cellText=table["rows"],
        colLabels=table["columns"],
        colWidths=lengths / lengths.sum(),
        cellLoc="left",
        loc="center",
    )
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(9)
    rendered.scale(1, 1.6)
    for (row, _), cell in rendered.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e9eef7")
            cell.set_text_props(weight="bold")
    axis.set_title("Table " + number + ": " + table["title"], fontsize=12, pad=12)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / (stem + "." + suffix), dpi=200, bbox_inches="tight")
    plt.close(fig)


def spectrum(output):
    """Use the exact curves, labels, scale and layout of V05 Figure 2."""
    with np.load(ROOT / "results/v05_spectrum_curves.npz", allow_pickle=False) as data:
        fig, axis = plt.subplots(figsize=(6.4, 3.4))
        for label, color in [
            ("Desired signal", "tab:blue"),
            ("Predistorted GMP + I/Q CNN", "tab:orange"),
            ("Matched RVT residual", "tab:red"),
            ("Proposed joint refinement", "tab:green"),
        ]:
            axis.plot(
                data["frequency_mhz"], data[label], label=label, color=color, lw=1
            )
        axis.set(
            xlim=(-320, 320),
            ylim=(-60, 5),
            xlabel="Frequency (MHz)",
            ylabel="Mean FFT power (dB, common reference)",
        )
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7.5, loc="upper right", ncol=2)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(output / ("figure_2_spectrum." + suffix), dpi=200)
        plt.close(fig)


def architecture(output):
    """Redraw Figure 1's topology as a readable schematic without LaTeX."""
    fig, ax = plt.subplots(figsize=(12, 6.3))
    ax.set(xlim=(0, 14), ylim=(0, 7.6))
    ax.axis("off")

    def box(x, y, width, text, height=0.8):
        ax.add_patch(
            FancyBboxPatch(
                (x - width / 2, y - height / 2),
                width,
                height,
                boxstyle="round,pad=0.03",
                fc="white",
                ec="black",
                lw=1.2,
            )
        )
        ax.text(x, y, text, ha="center", va="center", fontsize=10)

    def arrow(points, dashed=False):
        for first, second in zip(points[:-2], points[1:-1]):
            ax.plot(
                [first[0], second[0]],
                [first[1], second[1]],
                color="black",
                lw=1.1,
                ls="--" if dashed else "-",
            )
        ax.annotate(
            "",
            xy=points[-1],
            xytext=points[-2],
            arrowprops=dict(arrowstyle="->", lw=1.1, linestyle="--" if dashed else "-"),
        )

    ax.text(
        0.3,
        7.1,
        "(a) Deployment: fixed GMP and CNN weights",
        weight="bold",
        fontsize=12,
    )
    ax.text(0.35, 5.1, r"$x[n]$", va="center", fontsize=12)
    box(2.4, 5.1, 2.0, "GMP + limiter\n" + r"$b_c[x]$")
    box(6.1, 6.3, 2.6, "I/Q features\n" + r"$x,\ b_c[x]$")
    box(9.0, 6.3, 2.3, "Residual CNN\n4 → 24 → 12 → 2")
    ax.add_patch(Circle((9.0, 5.1), 0.21, fc="white", ec="black"))
    ax.text(9.0, 5.1, "+", ha="center", va="center", fontsize=15)
    box(10.25, 5.1, 1.1, "Limiter")
    box(12.1, 5.1, 1.6, "Physical PA")
    arrow([(0.8, 5.1), (1.35, 5.1)])
    arrow([(0.9, 5.1), (0.9, 6.3), (4.75, 6.3)])
    arrow([(2.4, 5.53), (2.4, 5.75), (6.1, 5.75), (6.1, 5.87)])
    arrow([(3.43, 5.1), (8.76, 5.1)])
    ax.text(6.0, 5.2, r"$b_c[x]$", ha="center", fontsize=11)
    arrow([(7.43, 6.3), (7.82, 6.3)])
    arrow([(9.0, 5.87), (9.0, 5.34)])
    ax.text(9.15, 5.6, r"$d_{\omega,c}$", fontsize=10)
    arrow([(9.23, 5.1), (9.67, 5.1)])
    arrow([(10.83, 5.1), (11.27, 5.1)])
    ax.text(11.05, 5.22, r"$z$", fontsize=11)
    arrow([(12.93, 5.1), (13.6, 5.1)])
    ax.text(13.65, 5.1, r"$y$", va="center", fontsize=12)
    ax.plot([0.2, 13.8], [4.45, 4.45], color="gray", lw=0.7)
    ax.text(
        0.3,
        4.15,
        "(b) Offline training: first fit the GMP and PA surrogate to measured pairs",
        weight="bold",
        fontsize=11,
    )
    ax.text(0.35, 2.9, r"$x$", fontsize=12)
    box(3.0, 2.9, 3.0, "Trainable DPD\nGMP + CNN + limiters", 1.0)
    box(7.0, 2.9, 3.1, "Frozen differentiable\nDGRU PA surrogate", 1.0)
    box(11.2, 2.9, 3.0, "Output/reference loss\nvalidation safeguards", 1.0)
    box(
        7.0,
        1.35,
        4.6,
        "Stage 2: CNN only\nStage 3: CNN + 24 existing GMP coefficients",
        1.0,
    )
    arrow([(0.8, 2.9), (1.47, 2.9)])
    arrow([(4.53, 2.9), (5.42, 2.9)])
    arrow([(8.58, 2.9), (9.67, 2.9)])
    ax.text(9.05, 3.03, r"$\widehat y$", fontsize=11)
    arrow([(0.9, 2.9), (0.9, 3.65), (11.2, 3.65), (11.2, 3.43)])
    arrow([(11.2, 2.37), (11.2, 1.35), (9.33, 1.35)], True)
    arrow([(4.67, 1.35), (3.0, 1.35), (3.0, 2.37)], True)
    ax.text(
        7.0,
        0.3,
        "Schematic topology; reported PA outputs are surrogate predictions. No online weight updates.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            output / ("figure_1_architecture." + suffix), dpi=200, bbox_inches="tight"
        )
    plt.close(fig)


def generate(output):
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    tables = table_data()
    for number, table in tables.items():
        export_table(number, table, output)
    dump(output / "tables_unrounded.json", tables)
    spectrum(output)
    architecture(output)
    (output / "README.md").write_text(
        "# V05 result exports\n\nTables I–V are exported as CSV, Markdown, PNG and PDF. "
        "Numerical tables are recomputed from the retained archived records, not newly trained models. "
        "Table I documents the executed training settings. Figure 2 uses the exact archived V05 "
        "FFT curves, including the matched 1,424-weight RVT residual control. Figure 1 is a "
        "redrawing of the same architecture topology, not a pixel-identical copy of the LaTeX diagram. "
        "All reported PA output spectra are surrogate predictions.\n",
        encoding="utf-8",
    )
    manifest = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.iterdir()
        if p.is_file()
    }
    dump(output / "generated_files_sha256.json", manifest)
