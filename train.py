# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Nimrod Glazer and JMBNR contributors
"""Retrain the proposed baseline/refinement through the bundled frozen PA."""

import argparse
from datetime import datetime
from pathlib import Path
import shutil
import numpy as np
import torch
from jmbnr.data import ROOT, load_data, read, dump
from jmbnr.gmp import fit_rls, GmpConfig
from jmbnr.training import train_baseline, train_refinement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["both", "baseline", "refinement"], default="both"
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--baseline-updates", type=int, default=7800)
    parser.add_argument("--refinement-updates", type=int, default=600)
    parser.add_argument(
        "--refit-gmp",
        action="store_true",
        help="Refit the offline postinverse from unchanged measured training pairs.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One optimizer update per requested stage; execution check, not published performance.",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if min(args.baseline_updates, args.refinement_updates) <= 0:
        parser.error("Update counts must be positive.")
    output = args.output_dir or ROOT / "outputs" / (
        "training-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    output.mkdir(parents=True, exist_ok=False)
    models = output / "models"
    models.mkdir()
    for name in ["pa.pt", "rvtdcnn.pt", "gmp_initial.npy", "config.json"]:
        shutil.copy2(ROOT / "models" / name, models / name)
    if args.refit_gmp:
        checkpoint = torch.load(models / "pa.pt", weights_only=True, map_location="cpu")
        gain = complex(checkpoint["gain_real"], checkpoint["gain_imag"])
        model, report = fit_rls(
            load_data(),
            gain,
            output,
            GmpConfig(**read(models / "config.json")["gmp_config"]),
        )
        np.save(models / "gmp_initial.npy", model.coeffs)
    if args.stage in ("both", "baseline"):
        train_baseline(
            models,
            output / "baseline",
            1 if args.smoke else args.baseline_updates,
            args.seed,
        )
    else:
        shutil.copy2(ROOT / "models/baseline_cnn.pt", models / "baseline_cnn.pt")
    if args.stage in ("both", "refinement"):
        job = dict(
            folder=str(output / "refinement"),
            seed=args.seed,
            far_constrained=True,
            updates=1 if args.smoke else args.refinement_updates,
            block=2048,
            batch=4,
            validation_every=1 if args.smoke else 50,
            lr=1e-4,
            gmp_lr=1e-5,
            gmp_anchor=1.0,
            constraint_strength=25.0,
            amplitude_weight=0.25,
            far_strength=25.0,
            far_train_margin_db=0.1,
            far_tolerance_db=0.05,
        )
        train_refinement(job, models)
    dump(
        output / "training_scope.json",
        dict(
            stage=args.stage,
            seed=args.seed,
            smoke=args.smoke,
            pa_retrained=False,
            measured_inputs_changed=False,
            published_performance_claim=False,
        ),
    )
    print("Outputs location:", output.resolve(), flush=True)
    print(
        'Replay these models: python run.py --models-dir "'
        + str(models.resolve())
        + '"',
        flush=True,
    )


if __name__ == "__main__":
    main()
