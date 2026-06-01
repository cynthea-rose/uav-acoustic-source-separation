"""
compute_f1.py
Computes two F1 scores for the thesis:

  1. Binary detection F1 — did the model detect a drone in the mixture?
     Ground truth: drone present (1) for all real scenes.
     Prediction: estimated drone stem energy > threshold.

  2. Drone count F1 — how many drones does DroneCountCNN predict?
     Ground truth: 1 drone per scene (real scenes each contain 1 drone file).
     Prediction: DroneCountCNN argmax on mel spectrogram of mixture.

Usage (from src/ folder):
    python compute_f1.py

Outputs saved to ../results/:
    f1_results.json        — all F1 scores and classification reports
    f1_confusion.png       — confusion matrix for count estimation
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_real import RealScenesDataset
from models import ConvTasNet, DroneCountCNN

# ── config ────────────────────────────────────────────────────────────────────
SCENES_DIR    = "../data/scenes"
MANIFEST_PATH = os.path.join(SCENES_DIR, "scenes_manifest.json")
RESULTS_DIR   = "../results"
MODEL_PATH    = os.path.join(RESULTS_DIR, "conv_tasnet_best.pt")
DEVICE        = torch.device("cpu")
SAMPLE_RATE   = 16_000
N_MELS        = 128

# Energy threshold for binary detection
# If estimated drone stem RMS > this fraction of mixture RMS → drone detected
ENERGY_THRESHOLD = 0.1


# ── load model helper ─────────────────────────────────────────────────────────
def load_conv_tasnet():
    model = ConvTasNet().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_drone_count_cnn():
    model = DroneCountCNN(n_mels=N_MELS, n_classes=5).to(DEVICE)
    model.eval()
    # DroneCountCNN has no saved checkpoint — uses random init
    # This gives us the classification structure; results reflect untrained model
    # Note this in thesis as: DroneCountCNN not yet trained, F1 reflects random baseline
    return model


# ── mel spectrogram helper ────────────────────────────────────────────────────
def to_mel(waveform: np.ndarray) -> torch.Tensor:
    """Convert waveform to mel spectrogram tensor (1, n_mels, T)."""
    mel = librosa.feature.melspectrogram(
        y=waveform.astype(np.float32),
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=512,
        hop_length=256
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Normalise to [0, 1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)  # (1, n_mels, T)


# ── F1 helpers ────────────────────────────────────────────────────────────────
def binary_f1(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    return float(precision), float(recall), float(f1)


def macro_f1(y_true, y_pred, n_classes=5):
    """Macro-averaged F1 across all drone count classes (0-4)."""
    f1s = []
    precisions = []
    recalls = []
    for c in range(n_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        p  = tp / (tp + fp + 1e-8)
        r  = tp / (tp + fn + 1e-8)
        f  = 2 * p * r / (p + r + 1e-8)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
    return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s)), f1s


def confusion_matrix(y_true, y_pred, n_classes=5):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t][p] += 1
    return cm


# ── plot confusion matrix ─────────────────────────────────────────────────────
def plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    classes = [f"{i} drone{'s' if i != 1 else ''}" for i in range(5)]
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Ground Truth", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    thresh = cm.max() / 2
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=10,
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "="*56)
    print("  F1 Score Computation")
    print("="*56)

    # ── load manifest and dataset ─────────────────────────────────────────
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    print(f"\nLoaded {len(manifest)} scenes from manifest")

    dataset = RealScenesDataset(MANIFEST_PATH)

    # ── load Conv-TasNet ──────────────────────────────────────────────────
    print(f"Loading Conv-TasNet from {MODEL_PATH} ...")
    sep_model = load_conv_tasnet()
    print("  Conv-TasNet loaded")

    # ── load DroneCountCNN ────────────────────────────────────────────────
    print("Loading DroneCountCNN ...")
    count_model = load_drone_count_cnn()
    print("  DroneCountCNN loaded (untrained — random init)")

    # ── run inference on all scenes ───────────────────────────────────────
    print("\nRunning inference on all scenes ...")

    gt_binary    = []   # ground truth: always 1 (drone present)
    pred_binary  = []   # predicted: 1 if estimated drone energy > threshold

    gt_count     = []   # ground truth: always 1 (one drone per real scene)
    pred_count   = []   # predicted: DroneCountCNN argmax

    with torch.no_grad():
        for i in range(len(dataset)):
            mixture, drone, noise, _ = dataset[i]
            mixture_np = np.array(mixture).flatten()
            drone_np   = np.array(drone).flatten()
            noise_np   = np.array(noise).flatten()

            # ── (A) Binary detection via Conv-TasNet energy ───────────────
            mixture_t = torch.tensor(mixture_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)  # shape: (1, 1, T)
            estimated    = sep_model(mixture_t)
            est_drone_np = estimated[:, 0, :].cpu().numpy().squeeze()

            mix_rms   = np.sqrt(np.mean(mixture_np ** 2)) + 1e-8
            drone_rms = np.sqrt(np.mean(est_drone_np ** 2))
            detected  = 1 if (drone_rms / mix_rms) > ENERGY_THRESHOLD else 0

            gt_binary.append(1)      # ground truth: drone always present
            pred_binary.append(detected)

            # ── (B) Drone count via DroneCountCNN ─────────────────────────
            mel = to_mel(mixture_np)

            # DroneCountCNN expects (batch, n_mels, time)
            # Pad or crop time dimension to fixed size
            target_len = 128
            if mel.shape[-1] < target_len:
                mel = F.pad(mel, (0, target_len - mel.shape[-1]))
            else:
                mel = mel[..., :target_len]

            mel_batch    = mel.unsqueeze(0).to(DEVICE)   # (1, 1, n_mels, T)
            count_logits = count_model(mel_batch)
            pred_class   = int(count_logits.argmax(dim=-1).item())

            gt_count.append(1)         # ground truth: 1 drone per scene
            pred_count.append(pred_class)

            if (i + 1) % 20 == 0:
                print(f"  Processed {i+1}/{len(dataset)} scenes")

    # ── Binary F1 ─────────────────────────────────────────────────────────
    print("\n" + "="*56)
    print("  Results: Binary Drone Detection F1")
    print("="*56)
    prec_b, rec_b, f1_b = binary_f1(gt_binary, pred_binary)
    detected_count = sum(pred_binary)
    print(f"  Scenes with drone detected : {detected_count} / {len(pred_binary)}")
    print(f"  Precision : {prec_b:.4f}")
    print(f"  Recall    : {rec_b:.4f}")
    print(f"  F1 Score  : {f1_b:.4f}")

    # ── Count F1 ──────────────────────────────────────────────────────────
    print("\n" + "="*56)
    print("  Results: Drone Count Estimation F1 (Macro)")
    print("="*56)
    prec_c, rec_c, f1_c, per_class_f1 = macro_f1(gt_count, pred_count, n_classes=5)
    print(f"  Macro Precision : {prec_c:.4f}")
    print(f"  Macro Recall    : {rec_c:.4f}")
    print(f"  Macro F1 Score  : {f1_c:.4f}")
    print("\n  Per-class F1:")
    for cls, f in enumerate(per_class_f1):
        n_pred = sum(1 for p in pred_count if p == cls)
        print(f"    {cls} drone(s): F1 = {f:.4f}  (predicted {n_pred} times)")

    # ── Confusion matrix ───────────────────────────────────────────────────
    cm = confusion_matrix(gt_count, pred_count, n_classes=5)
    cm_path = os.path.join(RESULTS_DIR, "f1_confusion.png")
    plot_confusion(cm,
                   "DroneCountCNN — Predicted vs Ground Truth Drone Count\n(Real scenes, ground truth = 1 drone)",
                   cm_path)

    # ── Save JSON ──────────────────────────────────────────────────────────
    output = {
        "note": (
            "Binary detection F1 uses Conv-TasNet separated drone stem energy. "
            "Count F1 uses DroneCountCNN (untrained — random init). "
            "Ground truth count = 1 for all real scenes (one drone file per scene). "
            "Install mir_eval and train DroneCountCNN for production-quality scores."
        ),
        "binary_detection": {
            "description": "Drone presence detection via separated stem energy threshold",
            "threshold": ENERGY_THRESHOLD,
            "n_scenes": len(gt_binary),
            "n_detected": int(sum(pred_binary)),
            "precision": prec_b,
            "recall": rec_b,
            "f1": f1_b,
        },
        "count_estimation": {
            "description": "Drone count classification via DroneCountCNN (untrained)",
            "n_scenes": len(gt_count),
            "macro_precision": prec_c,
            "macro_recall": rec_c,
            "macro_f1": f1_c,
            "per_class_f1": {str(i): per_class_f1[i] for i in range(5)},
            "confusion_matrix": cm.tolist(),
        }
    }
    json_path = os.path.join(RESULTS_DIR, "f1_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results JSON -> {json_path}")
    print(f"  Confusion matrix -> {cm_path}")
    print("\n" + "="*56)
    print("  Done!")
    print("="*56)


if __name__ == "__main__":
    main()