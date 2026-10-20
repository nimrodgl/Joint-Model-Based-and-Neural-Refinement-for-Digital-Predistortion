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

Research code and manuscript rights are not assigned a new blanket license.
No manuscript, Overleaf project, or upstream publication PDF is included.
