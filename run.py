# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Nimrod Glazer and JMBNR contributors
"""Replay one saved five-method comparison, or plot its archived raw outputs."""

import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from jmbnr.data import (
    ROOT,
    load_data,
    load_pa,
    evaluate_pa,
    evaluate_dpd,
    read,
    dump,
    iq,
    complex_np,
    limit,
)
from jmbnr.gmp import GMP, GmpConfig
from jmbnr.models import ResidualCNN, RVTDCNN
from jmbnr.metrics import metrics
from jmbnr.plotting import plot_spectra, write_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Use saved raw outputs; no model loading or PA inference.",
    )
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or ROOT / "outputs" / datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    output.mkdir(parents=True, exist_ok=False)
    expected = np.load(ROOT / "results/expected_outputs.npz", allow_pickle=False)
    reference = expected["reference"]
    if args.plots_only:
        outputs = {
            key[7:]: expected[key]
            for key in expected.files
            if key.startswith("output_")
        }
        results = read(ROOT / "results/expected_metrics.json")
    else:
        data = load_data()
        np.testing.assert_array_equal(reference, data["test"][0])
        config = read(args.models_dir / "config.json")
        maximum = config["maximum_pa_input"]
        cfg = GmpConfig(**config["gmp_config"])
        pa = load_pa(args.models_dir / "pa.pt")
        initial = GMP(cfg, np.load(args.models_dir / "gmp_initial.npy"))
        baseline = complex_np(
            limit(iq(initial.predistort(reference)), maximum)
        ).reshape(-1)
        inputs = {"no_dpd": reference, "gmp_rls": baseline}
        for method, weight, coefficient in [
            ("donor_cnn", "baseline_cnn.pt", "gmp_initial.npy"),
            ("long_joint_cnn", "jmbnr_cnn.pt", "gmp_refined.npy"),
        ]:
            if not (args.models_dir / weight).exists():
                continue
            cnn = ResidualCNN("continuation", 1.0).eval()
            cnn.load_state_dict(
                torch.load(
                    args.models_dir / weight, weights_only=True, map_location="cpu"
                )["state_dict"]
            )
            gmp = GMP(cfg, np.load(args.models_dir / coefficient))
            base = complex_np(limit(iq(gmp.predistort(reference)), maximum)).reshape(-1)
            inputs[method] = evaluate_dpd(cnn, reference, base, maximum)
        rvt = RVTDCNN(3).double().eval()
        rvt.load_state_dict(
            torch.load(
                args.models_dir / "rvtdcnn.pt", weights_only=True, map_location="cpu"
            )["state_dict"]
        )
        # Native reconstructed benchmark uses float64 internally, then the common float32 limiter.
        samples = torch.as_tensor(
            np.stack((reference.real, reference.imag), -1), dtype=torch.float64
        )
        with torch.no_grad():
            prediction = rvt(samples.unsqueeze(0))[0].numpy()
        inputs["rvtdcnn_158"] = complex_np(
            limit(iq(prediction[:, 0] + 1j * prediction[:, 1]), maximum)
        ).reshape(-1)
        outputs = {}
        results = {}
        compare = args.models_dir.resolve() == (ROOT / "models").resolve()
        expected_metrics = read(ROOT / "results/expected_metrics.json")
        for method, values in inputs.items():
            outputs[method] = evaluate_pa(pa, values).reshape(-1)
            results[method] = metrics(reference, outputs[method])
            if compare:
                if method != "no_dpd":
                    np.testing.assert_allclose(
                        values, expected["input_" + method], atol=1e-6, rtol=1e-6
                    )
                np.testing.assert_allclose(
                    outputs[method], expected["output_" + method], atol=2e-6, rtol=1e-6
                )
                for group, scores in results[method].items():
                    for key, value in scores.items():
                        np.testing.assert_allclose(
                            value,
                            expected_metrics[method][group][key],
                            atol=2e-5,
                            rtol=0,
                        )
            print(
                method
                + ": replay complete"
                + ("; matches archived output and metrics" if compare else ""),
                flush=True,
            )
        np.savez_compressed(output / "outputs.npz", reference=reference, **outputs)
    dump(output / "metrics.json", results)
    write_table(results, output)
    plot_spectra(reference, outputs, output)
    print("Outputs location:", output.resolve(), flush=True)


if __name__ == "__main__":
    main()
