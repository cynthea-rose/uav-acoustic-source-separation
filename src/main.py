"""
main.py — UAV Acoustic Source Separation Pipeline
===================================================
Single entry point for the complete UAV acoustic source separation system.

This script runs the full pipeline in order:
  Phase 1 — Build synthetic scenes and train baseline models
  Phase 2 — Build real mixed scenes from AuDroKSoundData
  Phase 3 — Train Conv-TasNet on real data
  Phase 4 — 3-run repeatability study (seeds 42, 123, 456)
  Phase 5 — SNR vs SDR analysis
  Phase 6 — F1 score evaluation (binary detection + drone count)
  Phase 7 — Generate all thesis figures

Usage:
    cd src
    python main.py                  # run full pipeline
    python main.py --phase 3        # run a single phase only
    python main.py --skip-phase 1   # skip Phase 1 (e.g. if synthetic data exists)

Author:      Cynthea Rose Antony
Supervisor:  Dr. Saeed Ur Rehman
Course:      ENGR7002A Honours Thesis — Flinders University
"""

import os
import sys
import time
import argparse
import random
import numpy as np
import torch

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── paths ─────────────────────────────────────────────────────────────────────
SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(SRC_DIR)
DATA_DIR     = os.path.join(ROOT_DIR, "data")
RESULTS_DIR  = os.path.join(ROOT_DIR, "results")
SCENES_DIR   = os.path.join(DATA_DIR, "scenes")
MANIFEST     = os.path.join(SCENES_DIR, "scenes_manifest.json")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SCENES_DIR,  exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────
def banner(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def success(msg):
    print(f"  [OK] {msg}")

def skip(msg):
    print(f"  [SKIP] {msg}")

def check_data():
    """Verify that real audio data is present before running real-data phases."""
    real_dir = os.path.join(DATA_DIR, "real")
    if not os.path.exists(real_dir):
        print("\n  [ERROR] data/real/ directory not found.")
        print("  Please download AuDroKSoundData and place it in data/real/")
        print("  Required folders:")
        print("    data/real/21 MA2_fast/")
        print("    data/real/Babble_Al-Emadi/")
        print("    data/real/23-02-22 Background/32 Hoover/")
        print("    data/real/23-02-22 Background/80 Conversations/")
        return False
    return True

def check_manifest():
    """Check if scenes_manifest.json exists."""
    return os.path.exists(MANIFEST)

def check_model():
    """Check if trained model checkpoint exists."""
    return os.path.exists(os.path.join(RESULTS_DIR, "conv_tasnet_best.pt"))


# ── phase functions ───────────────────────────────────────────────────────────

def phase1_synthetic():
    """Phase 1 — Build synthetic scenes and train baseline models."""
    banner("Phase 1 — Synthetic Scene Builder + Baseline Training")
    try:
        from synthetic_scene_builder import build_dataset
        print("  Building 40 synthetic scenes (1-4 drones + pink noise, SNR 0-15 dB)...")
        build_dataset(n_scenes=40)
        success("Synthetic scenes built")

        from feature_extraction import run_feature_extraction
        synth_manifest = os.path.join(DATA_DIR, "synthetic", "dataset_manifest.json")
        if os.path.exists(synth_manifest):
            print("  Extracting mel-spectrograms for 5 sample scenes...")
            run_feature_extraction(synth_manifest, max_scenes=5)
            success("Feature extraction complete")

        from train_evaluate import run_pipeline
        print("  Training Conv-TasNet and Wave-U-Net on synthetic data (10 epochs)...")
        run_pipeline(synth_manifest)
        success("Synthetic baseline training complete")

    except Exception as e:
        print(f"  [WARNING] Phase 1 error: {e}")
        print("  Continuing to next phase...")


def phase2_real_scenes():
    """Phase 2 — Build 100 real mixed scenes from AuDroKSoundData."""
    banner("Phase 2 — Real Scene Builder")

    if not check_data():
        print("  Skipping Phase 2 — real data not found.")
        return

    if check_manifest():
        skip("scenes_manifest.json already exists — skipping scene generation")
        print("  Delete data/scenes/ and re-run to regenerate scenes.")
        return

    try:
        from real_scene_builder import build_real_scenes
        print("  Mixing real drone clips with background noise...")
        print("  Generating 100 scenes (64 hoover + 36 conversation, SNR 0-15 dB)...")
        build_real_scenes(n_scenes=100)
        success("100 real mixed scenes generated")
        success(f"Manifest saved to {MANIFEST}")
    except Exception as e:
        print(f"  [WARNING] Phase 2 error: {e}")
        print("  Continuing to next phase...")


def phase3_train_real():
    """Phase 3 — Train Conv-TasNet on real data for 30 epochs."""
    banner("Phase 3 — Real Data Training (Conv-TasNet, 30 epochs)")

    if not check_manifest():
        print("  [SKIP] scenes_manifest.json not found — run Phase 2 first.")
        return

    try:
        from train_real import main as train_real_main
        print("  Training Conv-TasNet on 100 real scenes...")
        print("  Device: CPU | Epochs: 30 | Batch size: 4 | LR: 1e-3")
        print("  This may take 30–60 minutes on CPU...")
        train_real_main()
        success("Real data training complete")
        success(f"Best model saved to results/conv_tasnet_best.pt")
    except Exception as e:
        print(f"  [WARNING] Phase 3 error: {e}")
        print("  Continuing to next phase...")


def phase4_repeatability():
    """Phase 4 — 3-run repeatability study with seeds 42, 123, 456."""
    banner("Phase 4 — Repeatability Study (3 seeds)")

    if not check_manifest():
        print("  [SKIP] scenes_manifest.json not found — run Phase 2 first.")
        return

    try:
        from train_real_repeatability import main as rep_main
        print("  Training Conv-TasNet 3 times with seeds 42, 123, 456...")
        print("  This may take 1.5–2 hours on CPU...")
        rep_main()
        success("Repeatability study complete")
        success("Mean ± std saved to results/repeatability_results.json")
        success("Figures saved to results/repeatability_curves.png and repeatability_metrics.png")
    except Exception as e:
        print(f"  [WARNING] Phase 4 error: {e}")
        print("  Continuing to next phase...")


def phase5_snr_analysis():
    """Phase 5 — SNR vs SDR analysis across 5 input SNR bands."""
    banner("Phase 5 — SNR vs SDR Analysis")

    if not check_model():
        print("  [SKIP] conv_tasnet_best.pt not found — run Phase 3 first.")
        return

    try:
        from snr_vs_sdr import main as snr_main
        print("  Evaluating model on all 100 scenes grouped by SNR band...")
        print("  Bands: 0-3, 3-6, 6-9, 9-12, 12-15 dB")
        snr_main()
        success("SNR analysis complete")
        success("Figure 4 saved to results/snr_vs_sdr.png")
        success("Data saved to results/snr_vs_sdr.json")
    except Exception as e:
        print(f"  [WARNING] Phase 5 error: {e}")
        print("  Continuing to next phase...")


def phase6_f1():
    """Phase 6 — F1 score evaluation (binary detection + drone count)."""
    banner("Phase 6 — F1 Score Evaluation")

    if not check_model():
        print("  [SKIP] conv_tasnet_best.pt not found — run Phase 3 first.")
        return

    try:
        from compute_f1 import main as f1_main
        print("  Computing binary drone detection F1 (Conv-TasNet energy threshold)...")
        print("  Computing drone count estimation F1 (DroneCountCNN)...")
        f1_main()
        success("F1 evaluation complete")
        success("Results saved to results/f1_results.json")
        success("Confusion matrix saved to results/f1_confusion.png")
    except Exception as e:
        print(f"  [WARNING] Phase 6 error: {e}")
        print("  Continuing to next phase...")


def phase7_figures():
    """Phase 7 — Generate all thesis figures."""
    banner("Phase 7 — Thesis Figure Generation")

    try:
        from generate_thesis_figures import main as fig_main
        print("  Generating Figure 2 (training loss curve)...")
        print("  Generating Figure 3 (metrics comparison bar chart)...")
        fig_main()
        success("Figure 2 saved to results/figure2_training_loss.png")
        success("Figure 3 saved to results/figure3_metrics_bar.png")
    except Exception as e:
        print(f"  [WARNING] Phase 7 error: {e}")


# ── summary ───────────────────────────────────────────────────────────────────
def print_summary():
    banner("Pipeline Complete — Output Summary")

    files = {
        "results/conv_tasnet_best.pt":          "Trained Conv-TasNet model checkpoint",
        "results/repeatability_results.json":   "Mean ± std metrics across 3 seeds",
        "results/f1_results.json":              "Binary detection and count F1 scores",
        "results/snr_vs_sdr.json":             "SDR per SNR band (5 bands)",
        "results/figure2_training_loss.png":    "Figure 2 — Training loss curve",
        "results/figure3_metrics_bar.png":      "Figure 3 — Metrics comparison chart",
        "results/snr_vs_sdr.png":              "Figure 4 — SNR vs SDR graph",
        "results/f1_confusion.png":            "DroneCountCNN confusion matrix",
        "results/spectrograms/scene_0000.png":  "Figure 1 — Mel-spectrogram comparison",
    }

    print(f"\n  {'File':<45} {'Status':<10} Description")
    print(f"  {'-'*45} {'-'*10} {'-'*35}")
    for rel_path, desc in files.items():
        full_path = os.path.join(ROOT_DIR, rel_path)
        status = "EXISTS" if os.path.exists(full_path) else "MISSING"
        marker = "✓" if status == "EXISTS" else "✗"
        print(f"  {rel_path:<45} {marker} {status:<8} {desc}")

    print("\n" + "=" * 60)
    print("  To reproduce individual steps:")
    print("    python train_real.py                 # Phase 3")
    print("    python train_real_repeatability.py   # Phase 4")
    print("    python snr_vs_sdr.py                 # Phase 5")
    print("    python compute_f1.py                 # Phase 6")
    print("    python generate_thesis_figures.py    # Phase 7")
    print("=" * 60 + "\n")


# ── argument parser ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="UAV Acoustic Source Separation — Full Pipeline"
    )
    parser.add_argument(
        "--phase", type=int, choices=[1, 2, 3, 4, 5, 6, 7],
        help="Run a single phase only (1-7)"
    )
    parser.add_argument(
        "--skip-phase", type=int, choices=[1, 2, 3, 4, 5, 6, 7],
        help="Skip a specific phase"
    )
    return parser.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  UAV Acoustic Source Separation for Drone Detection")
    print("  Cynthea Rose Antony | ENGR7002A | Flinders University")
    print("=" * 60)
    print(f"\n  Root directory : {ROOT_DIR}")
    print(f"  Results        : {RESULTS_DIR}")
    print(f"  Device         : CPU")

    phases = {
        1: phase1_synthetic,
        2: phase2_real_scenes,
        3: phase3_train_real,
        4: phase4_repeatability,
        5: phase5_snr_analysis,
        6: phase6_f1,
        7: phase7_figures,
    }

    start = time.time()

    if args.phase:
        # Run single phase
        phases[args.phase]()
    else:
        # Run all phases in order
        for num, func in phases.items():
            if args.skip_phase and num == args.skip_phase:
                banner(f"Phase {num} — SKIPPED (--skip-phase {num})")
                continue
            func()

    print_summary()
    elapsed = time.time() - start
    print(f"  Total time: {elapsed/60:.1f} minutes\n")


if __name__ == "__main__":
    main()