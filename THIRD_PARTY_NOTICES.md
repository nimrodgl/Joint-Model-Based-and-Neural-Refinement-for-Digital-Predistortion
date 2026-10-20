# Provenance and attribution

OpenDPD: https://github.com/lab-emi/OpenDPD, snapshot
1207a6f06349d25756a792a76241150bdf518af4. The original DPA_160MHz CSVs,
PA checkpoint, DGRU equations and reporting metrics are retained. Extracted
OpenDPD code is covered by the bundled LICENSE-OpenDPD (Apache-2.0).
The readable PA wrapper bypasses the unused CUDA graph helper and implements
the archived eager equations. Finite zero-I/Q features remain unchanged.

The RVTDCNN is the project's reconstruction of Hu et al.,
DOI 10.1109/TNNLS.2021.3054867. It is trained on OpenDPD data;
it is not a reproduction of the original authors' PA/hardware results.

Original JMBNR contributions are licensed under the root MIT `LICENSE`.
That grant covers only rights held by the JMBNR authors; it does not replace
upstream licenses or grant rights to third-party material.

OpenDPD material retaining upstream terms includes:

- The adapted PA/DGRU implementation in `jmbnr/models.py` (mixed original and
  adapted code; OpenDPD portions remain Apache-2.0).
- `jmbnr/opendpd_metrics.py` (extracted reporting code).
- The original captures and metadata in `data/` and the upstream PA weights
  in `models/pa.pt`, distributed with the retained upstream license and provenance.

Modifications to OpenDPD code: the PA wrapper implements the archived eager
equations without the unused CUDA graph helper; metric functions were extracted
and reformatted for standalone use. Project modifications are MIT-licensed
without removing the upstream Apache-2.0 obligations.

The upstream Apache-2.0 text is preserved verbatim in `LICENSE-OpenDPD`.
Installed dependencies are separately licensed by their respective owners.
No manuscript, Overleaf project, or upstream publication PDF is included.
