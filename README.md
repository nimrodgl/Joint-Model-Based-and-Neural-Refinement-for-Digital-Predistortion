# Joint Model-Based and Neural Refinement for Digital Predistortion

Compact, readable JMBNR code, original OpenDPD data, saved results, and V05 table/figure reproduction.
No manuscripts, Overleaf files, publishing tools, or historical run directories are included.

```powershell
git clone https://github.com/nimrodgl/Joint-Model-Based-and-Neural-Refinement-for-Digital-Predistortion.git
cd Joint-Model-Based-and-Neural-Refinement-for-Digital-Predistortion
```

## What to open

- `jmbnr/models.py`: CNN, frozen PA surrogate, and reconstructed RVTDCNN layers.
- `jmbnr/gmp.py`: 132-term basis, offline ridge identification, selected 24 coefficients.
- `jmbnr/losses.py`: baseline, amplitude, adjacent, and far-band objectives.
- `jmbnr/training.py`: one baseline training loop and one protected refinement loop.
- `jmbnr/metrics.py`: reporting metrics and validation guards.
- `jmbnr/data.py`: unchanged CSV loading, limiting, and inference helpers.
- `jmbnr/opendpd_metrics.py`: attributed OpenDPD reporting formulas.
- `jmbnr/plotting.py`: spectrum plotting and comparison table.
- `jmbnr/reporting.py`: V05 table aggregation, exact spectrum plot, and architecture schematic.

The `.py` files are readable source code. The **four `.pt` files** are saved
weights: one PA evaluator, one unrefined CNN, one final JMBNR CNN, and one
reconstructed RVTDCNN. The two small `.npy` files are the original and refined
GMP coefficients. They are the only model artifacts retained.

## Install

Use Python 3.13, as used for validation. Windows PowerShell:

```powershell
python verify.py
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux/macOS use `source .venv/bin/activate` instead.
CPU execution is used to match the archived reference implementation.

## Generate V05 tables and figures

```powershell
python generate_results.py
```

This prints an output directory containing Tables I–V as CSV, Markdown,
PNG and PDF, plus `tables_unrounded.json`. Tables II–V recompute the paper's
aggregates from the retained study records; Table I reports the documented
executed training settings. No benchmark retraining is performed.

Figure 2 uses the exact saved V05 curves, including the matched-capacity
RVT residual control. Its raw curves are in `results/v05_spectrum_curves.npz`.
Figure 1 is redrawn from the same deployment/training topology as a schematic;
it is not a pixel-identical rendering of the paper's LaTeX diagram.
Both figures are generated as PNG and PDF, without any Overleaf dependency.

## Reproduce the saved run

```powershell
# Fast: generate a table and spectrum from the saved raw PA outputs.
python run.py --plots-only

# Load the five methods' saved coefficients/weights and run the PA surrogate.
# Each input/output and every reported metric is checked against the archive.
python run.py
```

Every command prints its output location and writes to a new `outputs/` folder.
The methods are no DPD, GMP--RLS, unrefined GMP + CNN, JMBNR, and the native
reconstructed RVTDCNN. These comparisons use unequal capacities/budgets.

## Retrain the proposed method

```powershell
# Execution check only: one update per stage.
python train.py --smoke

# Fresh 7,800-update baseline and 600-update protected joint refinement.
python train.py

# Refine the bundled, already-trained baseline instead.
python train.py --stage refinement

# Also re-identify the GMP from unchanged measured training pairs.
python train.py --refit-gmp
```

Training uses the bundled frozen PA; it does not retrain the PA evaluator or
the native RVTDCNN. Fresh training may differ numerically across environments.
To replay newly trained models, use the `--models-dir` command printed at the
end of training. Model updates never overwrite the bundled saved run.

## What saved results remain

`results/v05_spectrum_curves.npz` preserves the exact V05 Figure 2 curves.
`results/expected_outputs.npz` contains one test capture's raw predistorted
inputs and predicted PA outputs for the five methods. `expected_metrics.json`
contains their full unrounded scores. Only the baseline and final-refinement
histories for this run are retained. `study_summaries.json` keeps numerical
records for the broader seed, ablation, and capacity studies without their
hundreds of checkpoints or repeated waveforms.

**Scope:** the commands reproduce the representative saved run and retrain the
proposed method. They do not rerun every model/seed in the broader studies;
those original records are provided for inspection, not as newly recomputed
results. The previous complete archive remains local, outside this candidate.

The six original OpenDPD train/validation/test CSVs and metadata are included
unchanged. Results are surrogate predictions from one physical PA capture,
not newly measured predistorted outputs. Test data were inspected during
development. Across three adaptation orders, archived refinement means are
0.671/0.694 dB; those means are not the gain of this single representative run.
The earlier extra-1-dB target was not achieved. The reconstructed RVTDCNN is
not a reproduction of the original study's hardware measurements.

See `VALIDATION.json`, `SOURCE_PROVENANCE.json`, and `THIRD_PARTY_NOTICES.md`
for extraction checks and attribution. No new blanket project license is assigned.
