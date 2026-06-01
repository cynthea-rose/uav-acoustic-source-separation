"""
snr_vs_sdr.py
Generates the SNR vs SDR performance graph (Figure 4 for thesis).
Groups the 100 real scenes by SNR band and plots average SDR per band.

Usage (from src/ folder):
    python snr_vs_sdr.py

Outputs saved to ../results/:
    snr_vs_sdr.png         — Figure 4 for thesis
    snr_vs_sdr.json        — underlying numbers for thesis table
"""

import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_real import RealScenesDataset
from models import ConvTasNet

# ── config ────────────────────────────────────────────────────────────────────
SCENES_DIR    = "../data/scenes"
MANIFEST_PATH = os.path.join(SCENES_DIR, "scenes_manifest.json")
RESULTS_DIR   = "../results"
MODEL_PATH    = os.path.join(RESULTS_DIR, "conv_tasnet_best.pt")
DEVICE        = torch.device("cpu")

# SNR bands requested by supervisor
SNR_BANDS = [
    (0,  3,  "0–3 dB"),
    (3,  6,  "3–6 dB"),
    (6,  9,  "6–9 dB"),
    (9,  12, "9–12 dB"),
    (12, 15, "12–15 dB"),
]


# ── SI-SDR metric (numpy) ─────────────────────────────────────────────────────
def si_sdr_np(estimate: np.ndarray, target: np.ndarray) -> float:
    eps  = 1e-8
    t    = target   - target.mean()
    e    = estimate - estimate.mean()
    dot  = np.dot(e, t)
    s_t  = dot * t / (np.dot(t, t) + eps)
    e_n  = e - s_t
    return float(10 * np.log10(np.dot(s_t, s_t) / (np.dot(e_n, e_n) + eps) + eps))


def compute_sdr(estimate: np.ndarray, target: np.ndarray) -> float:
    """Use mir_eval SDR if available, otherwise fall back to SI-SDR."""
    try:
        import mir_eval
        sdr, _, _, _ = mir_eval.separation.bss_eval_sources(
            target[np.newaxis, :], estimate[np.newaxis, :]
        )
        return float(sdr[0])
    except Exception:
        return si_sdr_np(estimate, target)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── load manifest ─────────────────────────────────────────────────────
    print("Loading manifest ...")
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    print(f"  {len(manifest)} scenes found")

    # ── load best model ───────────────────────────────────────────────────
    print(f"Loading model from {MODEL_PATH} ...")
    model = ConvTasNet().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    print("  Model loaded")

    # ── load dataset ──────────────────────────────────────────────────────
    dataset = RealScenesDataset(MANIFEST_PATH)

    # ── evaluate each scene and record SDR + SNR ──────────────────────────
    print("\nEvaluating all scenes ...")
    scene_results = []   # list of {"snr_db": float, "sdr": float}

    with torch.no_grad():
        for i, entry in enumerate(manifest):
            snr_db = float(entry["snr_db"])

            # get audio tensors from dataset
            mixture, drone, noise, _ = dataset[i]
            mixture_t = torch.tensor(mixture, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            drone_np  = np.array(drone).squeeze()
            noise_np  = np.array(noise).squeeze()

            # run model — output shape (1, 2, T)
            estimated    = model(mixture_t)
            est_drone_np = estimated[:, 0, :].cpu().numpy().squeeze()
            est_noise_np = estimated[:, 1, :].cpu().numpy().squeeze()

            sdr_drone = compute_sdr(est_drone_np, drone_np)
            sdr_noise = compute_sdr(est_noise_np, noise_np)
            sdr_avg   = (sdr_drone + sdr_noise) / 2

            scene_results.append({"snr_db": snr_db, "sdr": sdr_avg})

            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(manifest)} scenes")

    # ── group by SNR band ─────────────────────────────────────────────────
    print("\nGrouping by SNR band ...")
    band_labels = []
    band_means  = []
    band_stds   = []
    band_counts = []
    band_data   = {}   # for JSON output

    for lo, hi, label in SNR_BANDS:
        sdrs = [r["sdr"] for r in scene_results
                if lo <= r["snr_db"] < hi]
        if len(sdrs) == 0:
            sdrs = [0.0]   # avoid empty-band crash
        mean = float(np.mean(sdrs))
        std  = float(np.std(sdrs, ddof=1)) if len(sdrs) > 1 else 0.0
        band_labels.append(label)
        band_means.append(mean)
        band_stds.append(std)
        band_counts.append(len(sdrs))
        band_data[label] = {"mean_sdr": mean, "std_sdr": std, "n_scenes": len(sdrs)}
        print(f"  {label:9s}  n={len(sdrs):3d}  SDR = {mean:+.2f} ± {std:.2f} dB")

    # ── plot ──────────────────────────────────────────────────────────────
    print("\nGenerating Figure 4 ...")
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(band_labels))

    # Bar chart with error bars
    bars = ax.bar(x, band_means, yerr=band_stds, capsize=6,
                  color="#1D9E75", alpha=0.85,
                  error_kw={"elinewidth": 1.5, "ecolor": "#333330"})

    # Line connecting bar tops to show trend
    ax.plot(x, band_means, "o--", color="#E24B4A",
            linewidth=1.8, markersize=6, label="Mean SDR trend")

    # Annotate each bar with value and scene count
    for i, (mean, std, n) in enumerate(zip(band_means, band_stds, band_counts)):
        ax.text(i, mean + (std if mean >= 0 else -std) + 0.8,
                f"{mean:.1f} dB\n(n={n})",
                ha="center", va="bottom", fontsize=8.5)

    ax.axhline(0, color="#444441", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(band_labels)
    ax.set_xlabel("Input SNR Band", fontsize=11)
    ax.set_ylabel("Average SDR (dB)", fontsize=11)
    ax.set_title("Conv-TasNet — SDR performance vs Input SNR\n(Real drone + background noise, 100 scenes)",
                 fontsize=11)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    png_path = os.path.join(RESULTS_DIR, "snr_vs_sdr.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  Figure 4 saved -> {png_path}")

    # ── save JSON ─────────────────────────────────────────────────────────
    output = {
        "description": "SDR grouped by input SNR band — Conv-TasNet on 100 real scenes",
        "model": MODEL_PATH,
        "bands": band_data,
        "all_scenes": scene_results,
    }
    json_path = os.path.join(RESULTS_DIR, "snr_vs_sdr.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Data JSON saved -> {json_path}")

    print("\nDone! Figure 4 is ready for your thesis.")
    print("="*56)


if __name__ == "__main__":
    main()