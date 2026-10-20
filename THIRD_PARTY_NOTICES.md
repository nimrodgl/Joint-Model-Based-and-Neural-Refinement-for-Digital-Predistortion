# Provenance and license scope

Original JMBNR code, documentation, locally trained artifacts, and original
results are offered under the root MIT `LICENSE`, to the extent of the
authors' rights. This does not relicense incorporated third-party material.
`FILE_LICENSES.json` identifies every bundled file, including binary artifacts.

## Downloaded material: Apache-2.0 retained

OpenDPD source snapshot: https://github.com/lab-emi/OpenDPD/tree/1207a6f06349d25756a792a76241150bdf518af4

The six `data/*_input.csv` / `data/*_output.csv` captures and `data/spec.json`
are unchanged files from `datasets/DPA_160MHz`. Their Git blob hashes were
checked against that exact upstream tree. The repository-wide upstream
Apache-2.0 license is retained verbatim in `LICENSE-OpenDPD`; no separate
dataset license is present in this pinned dataset subtree. `data/checksums.json`
is project-generated metadata, rather than an upstream download.

## Adapted source: upstream rights retained

Upstream authors: Yizhuo Wu, Chang Gao.
Contact: yizhuo.wu@tudelft.nl, chang.gao@tudelft.nl.

- `jmbnr/opendpd_metrics.py`: extracted from OpenDPD `utils/metrics.py` and
  reformatted for standalone use. Apache-2.0 applies to the upstream code;
  MIT covers only original JMBNR modifications.
- `jmbnr/models.py`: mixed-origin file. `PaSurrogate` adapts OpenDPD
  `backbones/dgru.py`; upstream portions remain Apache-2.0. Changes implement
  the eager wrapper without unused CUDA graph machinery. Original JMBNR
  contributions, including its other model implementations, are MIT.

No root OpenDPD NOTICE file was found in the pinned tree. Notices for its
frontend fonts, flags, and Plotly assets do not apply: none of those assets
is bundled here. The original source author/contact notices are retained
in the adapted Python headers above.

## Locally generated artifacts and results

`models/pa.pt` is a locally trained DGRU checkpoint, not a downloaded pretrained
checkpoint. Its provenance traces to `fit_pa` in the project's
`experiments/run_opendpd_comparison.py`. The three CNN checkpoints and two
GMP coefficient files are also project-generated. MIT covers the authors'
rights in these artifacts; the upstream code and training captures retain
their own terms. Training on a dataset is not described here as copying
ownership of that dataset into the resulting model weights.

`results/expected_outputs.npz` contains original test reference samples along
with project-generated predictions. It is marked mixed MIT / Apache-2.0,
so the MIT grant does not purport to relicense the embedded original samples.
`results/v05_spectrum_curves.npz` contains computed curves from that capture;
its map conservatively retains upstream terms for source-derived content.
Other project-generated numerical summaries/histories are MIT to the extent
of the authors' rights.

The RVTDCNN is the project's reconstruction of Hu et al.,
DOI 10.1109/TNNLS.2021.3054867, trained here on OpenDPD captures. It is not
represented as downloaded author code or the original hardware experiment.

Installed dependencies retain their own licenses and are not bundled or
relicensed by this repository. The MIT grant makes no claim over third-party
copyrights, patents, trademarks, or rights the JMBNR authors do not hold.
No manuscript, Overleaf project, or publication PDF is included.
