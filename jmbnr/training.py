"""Train the conditioned baseline and protected long-memory refinement.

Only this single proposed-method path is supported here. Archived benchmark
summaries are provided as data; their many other models are not bundled.
"""

import copy, time, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .data import (
    ROOT,
    DATA,
    DEVICE,
    CONTEXT,
    load_data,
    load_pa,
    evaluate_pa,
    evaluate_dpd,
    frames,
    complex_frames,
    iq,
    complex_np,
    limit,
    dump,
    read,
    sha,
)
from .gmp import GMP, GmpConfig, descriptors, selected_basis
from .models import ResidualCNN, JointSubset, matched_training_generator, fold
from .metrics import (
    metrics,
    validation_result,
    selection_score,
    admissible,
    far_admissible,
    amplitude_admissible,
)
from .losses import (
    fixed_gain_loss,
    original_loss,
    balanced_region_loss,
    refinement_loss,
    far_loss,
)


def train_baseline(model_dir, output, updates=7800, seed=1234):
    """Zero-head CNN training with the frozen PA and GMP; validation selects NMSE."""
    output.mkdir(parents=True, exist_ok=False)
    data = load_data()
    config = read(model_dir / "config.json")
    maximum = config["maximum_pa_input"]
    pa = load_pa(model_dir / "pa.pt")
    gmp = GMP(GmpConfig(**config["gmp_config"]), np.load(model_dir / "gmp_initial.npy"))
    baseline = {
        s: complex_np(limit(iq(gmp.predistort(pair[0])), maximum)).reshape(-1)
        for s, pair in data.items()
    }
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ResidualCNN("continuation", 1.0)
    generator = matched_training_generator(seed, DEVICE)
    x = frames(data["train"][0], 512)
    b = frames(baseline["train"], 512)
    with torch.no_grad():
        controls = torch.cat([pa(a)[:, CONTEXT : CONTEXT + 512] for a in b.split(16)])
    val = data["val"][0]
    val_base = evaluate_pa(pa, baseline["val"]).reshape(-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    def evaluate(update, loss):
        values = evaluate_pa(
            pa, evaluate_dpd(model, val, baseline["val"], maximum)
        ).reshape(-1)
        row = validation_result(
            val, values, val_base, 0.1, 0.5, balanced_selection=True
        )
        row.update(
            optimizer_updates=update,
            learning_rate=optimizer.param_groups[0]["lr"],
            training_loss=loss,
        )
        return row

    row = evaluate(0, None)
    history = [row]
    best = copy.deepcopy(model.state_dict())
    best_row = row
    permutation = torch.empty(0, dtype=torch.long)
    position = 0
    for update in range(1, updates + 1):
        if position >= len(permutation):
            permutation = torch.randperm(len(x), generator=generator)
            position = 0
        indices = permutation[position : position + 16]
        position += 16
        model.train()
        optimizer.zero_grad()
        prediction = pa(limit(model(x[indices], b[indices]), maximum))[
            :, CONTEXT : CONTEXT + 512
        ]
        target = x[indices, CONTEXT : CONTEXT + 512]
        loss = (
            fixed_gain_loss(prediction, target)
            + original_loss(prediction, target, controls[indices])
            + balanced_region_loss(prediction, target, controls[indices])
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite baseline loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update % 50 == 0 or update == updates:
            row = evaluate(update, float(loss.detach()))
            history.append(row)
            if row["fixed_gain_nmse_db"] < best_row["fixed_gain_nmse_db"]:
                best = copy.deepcopy(model.state_dict())
                best_row = row
            dump(output / "history.json", history)
            print(
                f"Baseline {update}/{updates}: validation NMSE {row['fixed_gain_nmse_db']:.3f} dB",
                flush=True,
            )
        if update == 6300:
            for group in optimizer.param_groups:
                group["lr"] = 1e-4
    model.load_state_dict(best)
    model.eval()
    torch.save(
        dict(
            state_dict=model.state_dict(), selected_update=best_row["optimizer_updates"]
        ),
        model_dir / "baseline_cnn.pt",
    )
    reference = data["test"][0]
    u = evaluate_dpd(model, reference, baseline["test"], maximum)
    y = evaluate_pa(pa, u).reshape(-1)
    np.savez_compressed(
        output / "evaluation.npz", reference=reference, pa_input=u, pa_output=y
    )
    dump(
        output / "result.json",
        dict(
            selected_validation=best_row,
            test=metrics(reference, y),
            updates=updates,
            seed=seed,
        ),
    )
    return model


def infer(model, reference, raw, features):
    model.eval()
    with torch.no_grad():
        u, _ = model(
            iq(reference).unsqueeze(0),
            torch.as_tensor(raw, dtype=torch.complex128, device=DEVICE).unsqueeze(0),
            torch.as_tensor(features, dtype=torch.complex128, device=DEVICE).unsqueeze(
                0
            ),
        )
    return complex_np(u).reshape(-1)


def train_refinement(job, model_dir):
    folder = Path(job["folder"])
    folder.mkdir(parents=True, exist_ok=False)
    dump(folder / "configuration.json", job)
    try:
        cfg = read(model_dir / "config.json")
        data = load_data()
        gmp_cfg = GmpConfig(**cfg["gmp_config"])
        coefficients = np.load(model_dir / "gmp_initial.npy")
        gmp = GMP(gmp_cfg, coefficients)
        raw = {s: gmp.predistort(v[0]) for s, v in data.items()}
        columns = descriptors(gmp_cfg)
        indices = [c["index"] for c in columns]
        phi = {s: selected_basis(v[0], columns) for s, v in data.items()}
        scales = np.sqrt(np.mean(abs(phi["train"]) ** 2, axis=0)).clip(1e-12)
        phi = {s: v / scales for s, v in phi.items()}
        maximum = float(abs(data["train"][0]).max())
        baseline = complex_np(limit(iq(raw["train"]), maximum)).reshape(-1)
        scale = max(
            float(np.sqrt(np.mean(abs(baseline - data["train"][0]) ** 2))), 1e-06
        )
        state = torch.load(
            model_dir / "baseline_cnn.pt", weights_only=True, map_location="cpu"
        )["state_dict"]
        torch.manual_seed(job["seed"])
        np.random.seed(job["seed"])
        model = JointSubset(state, scale, maximum).to(DEVICE)
        assert sum((p.numel() for p in model.parameters())) == 1454
        if job.get("freeze_gmp", False):
            model.offsets.requires_grad_(False)
        pa = load_pa(model_dir / "pa.pt")
        for p in pa.parameters():
            p.requires_grad_(False)
        val = data["val"][0]
        donor_val = metrics(
            val, evaluate_pa(pa, infer(model, val, raw["val"], phi["val"])).reshape(-1)
        )
        dump(folder / "donor_validation.json", donor_val)
        x = frames(data["train"][0], job["block"])
        r = complex_frames(raw["train"], job["block"])
        p = complex_frames(phi["train"], job["block"])
        with torch.no_grad():
            controls = torch.cat(
                [
                    pa(
                        model(
                            x[i : i + job["batch"]],
                            r[i : i + job["batch"]],
                            p[i : i + job["batch"]],
                        )[0]
                    )[:, CONTEXT : CONTEXT + job["block"]].detach()
                    for i in range(0, len(x), job["batch"])
                ]
            )
        groups = [{"params": model.cnn.parameters(), "lr": job["lr"]}]
        if model.offsets.requires_grad:
            groups.append({"params": [model.offsets], "lr": job["gmp_lr"]})
        optimizer = torch.optim.Adam(groups, weight_decay=0.0)
        generator = matched_training_generator(job["seed"], DEVICE)
        permutation = torch.empty(0, dtype=torch.long, device=DEVICE)
        position = 0
        history = []
        best = None
        start = time.perf_counter()
        for update in range(job["updates"] + 1):
            loss_value = None
            far_ratios = None
            if update:
                if position >= len(permutation):
                    permutation = torch.randperm(
                        len(x), device=DEVICE, generator=generator
                    )
                    position = 0
                batch = permutation[position : position + job["batch"]]
                position += job["batch"]
                model.train()
                optimizer.zero_grad()
                u, delta = model(x[batch], r[batch], p[batch])
                prediction = pa(u)[:, CONTEXT : CONTEXT + job["block"]]
                target = x[batch, CONTEXT : CONTEXT + job["block"]]
                loss, _ = refinement_loss(
                    prediction,
                    target,
                    controls[batch],
                    job["constraint_strength"],
                    job["amplitude_weight"],
                )
                anchor = (
                    delta[:, CONTEXT : CONTEXT + job["block"]].abs().square().mean()
                    + 0.1 * model.offsets.square().sum()
                ) / scale**2
                loss = loss + job["gmp_anchor"] * anchor.to(loss.dtype)
                if job["far_constrained"]:
                    penalty, far_ratios = far_loss(
                        prediction,
                        controls[batch],
                        job["far_strength"],
                        job["far_train_margin_db"],
                    )
                    loss = loss + penalty
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.cnn.parameters(), 1.0)
                nn.utils.clip_grad_norm_([model.offsets], 1.0)
                optimizer.step()
                loss_value = float(loss.detach())
            if update % job["validation_every"] == 0 or update == job["updates"]:
                measured = metrics(
                    val,
                    evaluate_pa(pa, infer(model, val, raw["val"], phi["val"])).reshape(
                        -1
                    ),
                )
                base_ok = admissible(measured, donor_val)
                far_ok = far_admissible(measured, donor_val, job["far_tolerance_db"])
                amplitude_ok = amplitude_admissible(
                    measured, donor_val, job.get("amplitude_validation_margin_db", 0.1)
                )
                feasible = (
                    base_ok and amplitude_ok and (far_ok or not job["far_constrained"])
                )
                row = dict(
                    update=update,
                    metrics=measured,
                    admissible=feasible,
                    base_safeguards=base_ok,
                    far_safeguards=far_ok,
                    amplitude_selection_guard=amplitude_ok,
                    loss=loss_value,
                    last_batch_far_ratios=(
                        None if far_ratios is None else far_ratios.cpu().tolist()
                    ),
                    elapsed_seconds=time.perf_counter() - start,
                )
                history.append(row)
                dump(folder / "history.json", history)
                if feasible and (
                    best is None
                    or selection_score(measured, donor_val)
                    < selection_score(best["row"]["metrics"], donor_val)
                ):
                    best = dict(
                        row=copy.deepcopy(row),
                        state={
                            k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()
                        },
                    )
                print(
                    f"seed {job['seed']} {('far' if job['far_constrained'] else 'free')}: {update}/{job['updates']}; L {measured['legacy']['acpr_left_db']:.3f}; R {measured['legacy']['acpr_right_db']:.3f}; eligible {feasible}",
                    flush=True,
                )
            if update == int(job["updates"] * 0.75):
                for group in optimizer.param_groups:
                    group["lr"] *= 0.3
        assert best is not None
        model.load_state_dict(best["state"])
        model.eval()
        dump(folder / "selection.json", best["row"])
        replay = metrics(
            val, evaluate_pa(pa, infer(model, val, raw["val"], phi["val"])).reshape(-1)
        )
        for category, values in replay.items():
            for key, value in values.items():
                np.testing.assert_allclose(
                    value, best["row"]["metrics"][category][key], atol=1e-06
                )
        reference = data["test"][0]
        u = infer(model, reference, raw["test"], phi["test"])
        y = evaluate_pa(pa, u).reshape(-1)
        test = metrics(reference, y)
        np.savez_compressed(
            folder / "evaluation.npz", reference=reference, pa_input=u, pa_output=y
        )
        baseline_model = ResidualCNN("continuation", 1.0).eval()
        baseline_model.load_state_dict(state)
        baseline_input = evaluate_dpd(
            baseline_model,
            reference,
            complex_np(limit(iq(raw["test"]), maximum)).reshape(-1),
            maximum,
        )
        donor_test = metrics(reference, evaluate_pa(pa, baseline_input).reshape(-1))
        offsets = model.offsets.detach().cpu().numpy()
        learned = coefficients.copy()
        learned[indices] += (offsets[:, 0] + 1j * offsets[:, 1]) / scales
        frozen = np.setdiff1d(np.arange(132), indices)
        assert np.array_equal(learned[frozen], coefficients[frozen])
        cnn = ResidualCNN("continuation", 1.0).to(DEVICE).eval()
        cnn.load_state_dict(fold(model.cnn.state_dict(), scale))
        deployed = GMP(gmp_cfg, learned)
        base = complex_np(limit(iq(deployed.predistort(reference)), maximum)).reshape(
            -1
        )
        replay_input = evaluate_dpd(cnn, reference, base, maximum)
        replay_output = evaluate_pa(pa, replay_input).reshape(-1)
        np.testing.assert_allclose(replay_input, u, atol=3e-07, rtol=1e-06)
        np.testing.assert_allclose(replay_output, y, atol=1e-06, rtol=1e-05)
        np.save(model_dir / "gmp_refined.npy", learned)
        torch.save(
            dict(
                state_dict={k: v.detach().cpu() for k, v in cnn.state_dict().items()},
                variant="continuation",
                scale=1.0,
                selected_update=best["row"]["update"],
            ),
            model_dir / "jmbnr_cnn.pt",
        )
        result = dict(
            seed=job["seed"],
            far_constrained=job["far_constrained"],
            freeze_gmp=job.get("freeze_gmp", False),
            trainable_real_parameters=sum(
                (p.numel() for p in model.parameters() if p.requires_grad)
            ),
            selected_update=best["row"]["update"],
            test=test,
            donor_test=donor_test,
            adjacent_gains_db={
                k: donor_test["legacy"][k] - test["legacy"][k]
                for k in ("acpr_left_db", "acpr_right_db")
            },
            far_changes_db=dict(
                legacy=test["legacy"]["far_oob_db"]
                - donor_test["legacy"]["far_oob_db"],
                notebook=test["notebook"]["oob_db"] - donor_test["notebook"]["oob_db"],
            ),
            test_base_safeguards=admissible(test, donor_test),
            test_far_safeguards=far_admissible(
                test, donor_test, job["far_tolerance_db"]
            ),
            validation_replay=True,
            deployment_replay=True,
            max_deployed_input_error=float(abs(replay_input - u).max()),
            max_deployed_output_error=float(abs(replay_output - y).max()),
            runtime_seconds=time.perf_counter() - start,
        )
        dump(folder / "result.json", result)
        dump(
            folder / "artifact_sha256.json",
            {
                p.name: sha(p)
                for p in folder.iterdir()
                if p.is_file() and p.name != "artifact_sha256.json"
            },
        )
        return result
    except Exception as error:
        dump(folder / "error.json", dict(error=str(error)))
        raise
