"""
train_real_repeatability.py
Conv-TasNet — 3-run repeatability training on real drone scenes.
Trains with seeds 42, 123, 456 and reports mean ± std for all metrics.
Addresses supervisor Gap 1: no confidence intervals in single-run results.

Usage (from src/ folder):
    python train_real_repeatability.py

Outputs saved to ../results/:
    repeatability_results.json      — all per-run and summary metrics
    repeatability_curves.png        — training loss curves for all 3 runs
    repeatability_metrics.png       — bar chart with error bars
"""

import os
import sys
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── import from your existing files ──────────────────────────────────────────
from train_real import RealScenesDataset, si_sdr_loss, pit_loss
from models import ConvTasNet

# ── config (matches train_real.py) ───────────────────────────────────────────
SEEDS         = [42, 123, 456]
EPOCHS        = 30
BATCH_SIZE    = 4
LEARNING_RATE = 1e-3
SCENES_DIR    = "../data/scenes"
MANIFEST_PATH = os.path.join(SCENES_DIR, "scenes_manifest.json")
RESULTS_DIR   = "../results"
DEVICE        = torch.device("cpu")


# ── reproducibility helper ────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── evaluation metrics ────────────────────────────────────────────────────────
def compute_metrics(estimate: np.ndarray, target: np.ndarray) -> dict:
    try:
        import mir_eval
        sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(
            target[np.newaxis, :], estimate[np.newaxis, :]
        )
        eps      = 1e-8
        t        = target   - target.mean()
        e        = estimate - estimate.mean()
        dot      = np.dot(e, t)
        s_t      = dot * t / (np.dot(t, t) + eps)
        e_n      = e - s_t
        si_sdr_v = 10 * np.log10(np.dot(s_t, s_t) / (np.dot(e_n, e_n) + eps) + eps)
        return {"sdr": float(sdr[0]), "si_sdr": float(si_sdr_v),
                "sir": float(sir[0]), "sar":    float(sar[0])}
    except Exception:
        eps  = 1e-8
        t    = target   - target.mean()
        e    = estimate - estimate.mean()
        dot  = np.dot(e, t)
        s_t  = dot * t / (np.dot(t, t) + eps)
        e_n  = e - s_t
        val  = 10 * np.log10(np.dot(s_t, s_t) / (np.dot(e_n, e_n) + eps) + eps)
        return {"sdr": float(val), "si_sdr": float(val),
                "sir": float(val), "sar":    float(val)}


# ── single training run ───────────────────────────────────────────────────────
def run_once(seed: int, dataset) -> dict:
    print(f"\n{'='*56}")
    print(f"  Run with seed {seed}")
    print(f"{'='*56}")

    set_seed(seed)

    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val   = n_total - n_train
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,          shuffle=False)

    model     = ConvTasNet().to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loss_curve = []
    best_loss  = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses = []

        # Dataset returns tuple: (mixture, drone, noise, noise_type)
        for mixture, drone, noise, _ in train_loader:
            mixture = mixture.to(DEVICE)
            drone   = drone.to(DEVICE)
            noise   = noise.to(DEVICE)

            optimiser.zero_grad()

            # model returns (B, 2, T) — channel 0 = drone, channel 1 = noise
            estimated = model(mixture)
            loss = pit_loss(estimated, drone, noise)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimiser.step()
            epoch_losses.append(loss.item())

        epoch_loss = np.mean(epoch_losses)
        loss_curve.append(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(),
                       os.path.join(RESULTS_DIR, f"conv_tasnet_seed{seed}.pt"))

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}/{EPOCHS}  Loss: {epoch_loss:+.4f} dB  "
                  f"Best: {best_loss:+.4f} dB")

    # ── evaluation ────────────────────────────────────────────────────────
    print(f"\n  Evaluating seed {seed} on validation set ...")
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for mixture, drone, noise, _ in val_loader:
            mixture = mixture.to(DEVICE)
            drone_np = drone.numpy().squeeze()
            noise_np = noise.numpy().squeeze()

            # estimated shape: (1, 2, T)
            estimated    = model(mixture)
            est_drone_np = estimated[:, 0, :].cpu().numpy().squeeze()
            est_noise_np = estimated[:, 1, :].cpu().numpy().squeeze()

            m_drone = compute_metrics(est_drone_np, drone_np)
            m_noise = compute_metrics(est_noise_np, noise_np)

            all_metrics.append({
                "sdr":    (m_drone["sdr"]    + m_noise["sdr"])    / 2,
                "si_sdr": (m_drone["si_sdr"] + m_noise["si_sdr"]) / 2,
                "sir":    (m_drone["sir"]    + m_noise["sir"])    / 2,
                "sar":    (m_drone["sar"]    + m_noise["sar"])    / 2,
            })

    avg = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}

    print(f"\n  Seed {seed} results:")
    print(f"    SDR    : {avg['sdr']:+.2f} dB")
    print(f"    SI-SDR : {avg['si_sdr']:+.2f} dB")
    print(f"    SIR    : {avg['sir']:+.2f} dB")
    print(f"    SAR    : {avg['sar']:+.2f} dB")

    return {"seed": seed, "loss_curve": loss_curve, "metrics": avg}


# ── summary stats ─────────────────────────────────────────────────────────────
def summarise(runs: list) -> dict:
    keys = ["sdr", "si_sdr", "sir", "sar"]
    summary = {}
    for k in keys:
        vals = [r["metrics"][k] for r in runs]
        summary[k] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals, ddof=1)),
            "runs": vals,
        }
    return summary


# ── plotting ──────────────────────────────────────────────────────────────────
def plot_loss_curves(runs: list):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors  = ["#E24B4A", "#1D9E75", "#7F77DD"]
    for run, color in zip(runs, colors):
        ax.plot(range(1, EPOCHS + 1), run["loss_curve"],
                label=f"Seed {run['seed']}", color=color, linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SI-SDR Loss (dB)")
    ax.set_title("Conv-TasNet training loss — 3 runs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "repeatability_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  Training curves -> {path}")


def plot_metrics_bar(summary: dict):
    metrics = ["SDR", "SI-SDR", "SIR", "SAR"]
    keys    = ["sdr", "si_sdr", "sir", "sar"]
    means   = [summary[k]["mean"] for k in keys]
    stds    = [summary[k]["std"]  for k in keys]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(metrics, means, yerr=stds, capsize=6,
                  color=["#E24B4A", "#1D9E75", "#7F77DD", "#EF9F27"],
                  error_kw={"elinewidth": 1.5, "ecolor": "#444441"})

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                mean + (std if mean >= 0 else -std) + 0.5,
                f"{mean:.2f}±{std:.2f}", ha="center", va="bottom", fontsize=9)

    ax.axhline(0, color="#444441", linewidth=0.8, linestyle="--")
    ax.set_ylabel("dB")
    ax.set_title("Conv-TasNet on real data — mean ± std (3 runs)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "repeatability_metrics.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Metrics chart  -> {path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(MANIFEST_PATH):
        print(f"ERROR: scenes_manifest.json not found at {MANIFEST_PATH}")
        sys.exit(1)

    print("\nLoading dataset ...")
    dataset = RealScenesDataset(MANIFEST_PATH)
    print(f"RealScenesDataset: {len(dataset)} scenes loaded")
    print(f"Conv-TasNet — 3-run repeatability  |  seeds: {SEEDS}")
    print(f"Device: {DEVICE}  |  Epochs: {EPOCHS}  |  Batch size: {BATCH_SIZE}")

    runs = []
    for seed in SEEDS:
        result = run_once(seed, dataset)
        runs.append(result)

    summary = summarise(runs)

    print(f"\n{'='*56}")
    print("  Final summary — mean ± std across 3 runs")
    print(f"{'='*56}")
    for k, label in [("sdr","SDR"), ("si_sdr","SI-SDR"), ("sir","SIR"), ("sar","SAR")]:
        m = summary[k]["mean"]
        s = summary[k]["std"]
        print(f"  {label:<8}: {m:+.2f} ± {s:.2f} dB")
    print(f"{'='*56}\n")

    output = {
        "config": {
            "seeds": SEEDS, "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        },
        "per_run": [{"seed": r["seed"], "metrics": r["metrics"]} for r in runs],
        "summary": summary,
    }
    json_path = os.path.join(RESULTS_DIR, "repeatability_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results JSON   -> {json_path}")

    plot_loss_curves(runs)
    plot_metrics_bar(summary)

    print("\nDone! All 3-run outputs saved to results/")
    print("="*56)


if __name__ == "__main__":
    main()