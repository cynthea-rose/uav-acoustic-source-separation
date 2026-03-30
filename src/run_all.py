"""
run_all.py — Master Script
Runs the complete UAV Acoustic Source Separation pipeline end-to-end:
  1. Synthetic Scene Builder
  2. Feature Extraction + Spectrogram Plots
  3. Model Training (Conv-TasNet + Wave-U-Net)
  4. Evaluation + Metric Charts

Run with:  python run_all.py
"""

import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MANIFEST = "data/synthetic/dataset_manifest.json"

# ── Step 1: Build synthetic dataset ──────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1 — Synthetic Scene Builder")
print("="*60)
from synthetic_scene_builder import build_dataset
build_dataset(n_scenes=40)

# ── Step 2: Feature extraction & spectrogram visualisation ────────────────────
print("\n" + "="*60)
print("STEP 2 — Feature Extraction & Spectrograms")
print("="*60)
from feature_extraction import run_feature_extraction
run_feature_extraction(MANIFEST, max_scenes=5)

# ── Step 3 & 4: Training + evaluation ─────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3 & 4 — Model Training & Evaluation")
print("="*60)
from train_evaluate import run_pipeline
run_pipeline(MANIFEST)

print("\n" + "="*60)
print("ALL STEPS COMPLETE")
print("Outputs:")
print("  data/synthetic/    — audio scenes + metadata")
print("  data/stems/        — ground-truth drone stems")
print("  outputs/spectrograms/ — mel-spectrogram plots")
print("  outputs/training_curves.png")
print("  outputs/validation_metrics.png")
print("  outputs/results.json")
print("="*60)
