"""
Real Scene Builder
Mixes real drone sounds (Babble_Al-Emadi) with real background 
noise (23-02-22 Background) to create training scenes for Conv-TasNet.

Pipeline:
    Drone clip (Babble_Al-Emadi) + Background noise → Mixed scene
    Conv-TasNet learns to separate: drone | background
"""

import os
import random
import json
import numpy as np
import librosa
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE    = 16000
CLIP_DURATION  = 5       # seconds
CLIP_SAMPLES   = SAMPLE_RATE * CLIP_DURATION
SNR_RANGE      = (0, 15) # dB
N_SCENES       = 100     # number of mixed scenes to generate
MICROPHONE     = 1       # which microphone to use from background

DRONE_DIR      = "../data/real/Babble_Al-Emadi"
BACKGROUND_DIR = "../data/real/23-02-22 Background"
OUTPUT_DIR     = "../data/scenes"
PLOTS_DIR      = "../results/spectrograms"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)


# ── Audio Helpers ─────────────────────────────────────────────────────────────

def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load and resample audio to target sample rate."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


def pad_or_crop(y: np.ndarray, length: int = CLIP_SAMPLES) -> np.ndarray:
    """Pad or crop audio to fixed length."""
    if len(y) >= length:
        # Random crop for variety
        start = random.randint(0, len(y) - length)
        return y[start:start + length]
    # Repeat to fill
    repeats = (length // len(y)) + 1
    y = np.tile(y, repeats)
    return y[:length]


def normalise(y: np.ndarray) -> np.ndarray:
    """Normalise to [-1, 1]."""
    peak = np.max(np.abs(y)) + 1e-8
    return y / peak


def mix_at_snr(
    drone: np.ndarray,
    noise: np.ndarray,
    snr_db: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scale noise to achieve target SNR then mix.
    Returns: mixture, drone_stem, noise_stem
    """
    drone_power = np.mean(drone ** 2) + 1e-8
    noise_power = np.mean(noise ** 2) + 1e-8
    target_noise_power = drone_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)
    noise_scaled = noise * scale

    mixture = drone + noise_scaled
    # Normalise mixture
    peak    = np.max(np.abs(mixture)) + 1e-8
    mixture = mixture / peak
    drone   = drone   / peak
    noise_scaled = noise_scaled / peak

    return (
        mixture.astype(np.float32),
        drone.astype(np.float32),
        noise_scaled.astype(np.float32)
    )


# ── Dataset Collectors ────────────────────────────────────────────────────────

def collect_drone_files(drone_dir: str) -> list[str]:
    """Collect all wav files from Babble_Al-Emadi."""
    files = []
    for f in os.listdir(drone_dir):
        if f.endswith(".wav"):
            files.append(os.path.join(drone_dir, f))
    print(f"Found {len(files)} drone clips in {drone_dir}")
    return files


def collect_background_files(
    background_dir: str,
    mic: int = MICROPHONE
) -> list[str]:
    """Collect background noise files from all subfolders."""
    files = []
    mic_suffix = f"Mi{mic}.wav"
    for subfolder in os.listdir(background_dir):
        subfolder_path = os.path.join(background_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        for f in os.listdir(subfolder_path):
            if f.endswith(mic_suffix) or f.endswith(".wav"):
                files.append(os.path.join(subfolder_path, f))
    print(f"Found {len(files)} background files in {background_dir}")
    return files


# ── Mel Spectrogram Plot ──────────────────────────────────────────────────────

def plot_scene(
    mixture:    np.ndarray,
    drone:      np.ndarray,
    noise:      np.ndarray,
    scene_id:   int,
    snr_db:     float,
    noise_type: str,
) -> str:
    """Plot mixture, drone stem and noise stem spectrograms."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    def draw(ax, y, title):
        S  = librosa.feature.melspectrogram(
            y=y, sr=SAMPLE_RATE, n_mels=128,
            fmin=20, fmax=8000
        )
        Sdb = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(
            Sdb, sr=SAMPLE_RATE, hop_length=256,
            x_axis="time", y_axis="mel",
            fmin=20, fmax=8000,
            cmap="magma", ax=ax
        )
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set_title(title, fontsize=11, fontweight="bold")

    draw(axes[0], mixture,
         f"Scene {scene_id} — Mixture "
         f"(Drone + {noise_type}, SNR={snr_db:.1f} dB)")
    draw(axes[1], drone,  "Drone source (ground truth)")
    draw(axes[2], noise,  f"Background noise — {noise_type}")

    plt.suptitle(
        f"Acoustic Source Separation — Scene {scene_id}",
        fontsize=13, y=1.01
    )
    plt.tight_layout()

    out = os.path.join(PLOTS_DIR, f"scene_{scene_id:04d}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Scene Builder ─────────────────────────────────────────────────────────────

def build_scenes(
    n_scenes:   int = N_SCENES,
    plot_first: int = 5,
) -> list[dict]:
    """
    Build n_scenes mixed scenes from real drone and background audio.
    Returns list of metadata dicts.
    """
    drone_files      = collect_drone_files(DRONE_DIR)
    background_files = collect_background_files(BACKGROUND_DIR)

    if not drone_files:
        raise FileNotFoundError(f"No drone files found in {DRONE_DIR}")
    if not background_files:
        raise FileNotFoundError(
            f"No background files found in {BACKGROUND_DIR}"
        )

    print(f"\nBuilding {n_scenes} real mixed scenes...")
    all_meta = []

    for i in range(n_scenes):
        # Randomly pick drone and background files
        drone_path = random.choice(drone_files)
        bg_path    = random.choice(background_files)
        snr_db     = random.uniform(*SNR_RANGE)

        # Determine noise type from path
        noise_type = "Hoover" if "Hoover" in bg_path else "Conversation"

        # Load and prepare
        drone = load_audio(drone_path)
        noise = load_audio(bg_path)
        drone = pad_or_crop(drone)
        noise = pad_or_crop(noise)
        drone = normalise(drone)
        noise = normalise(noise)

        # Mix at target SNR
        mixture, drone_stem, noise_stem = mix_at_snr(drone, noise, snr_db)

        # Save stems and mixture
        mix_path   = os.path.join(OUTPUT_DIR, f"scene_{i:04d}_mix.wav")
        drone_path_out = os.path.join(
            OUTPUT_DIR, f"scene_{i:04d}_drone.wav"
        )
        noise_path_out = os.path.join(
            OUTPUT_DIR, f"scene_{i:04d}_noise.wav"
        )

        sf.write(mix_path,         mixture,    SAMPLE_RATE)
        sf.write(drone_path_out,   drone_stem, SAMPLE_RATE)
        sf.write(noise_path_out,   noise_stem, SAMPLE_RATE)

        # Save metadata
        meta = {
            "scene_id":    i,
            "drone_file":  os.path.basename(drone_path),
            "bg_file":     os.path.basename(bg_path),
            "noise_type":  noise_type,
            "snr_db":      round(snr_db, 2),
            "mix_path":    mix_path,
            "drone_path":  drone_path_out,
            "noise_path":  noise_path_out,
        }
        all_meta.append(meta)

        # Plot first few scenes
        if i < plot_first:
            plot_scene(
                mixture, drone_stem, noise_stem,
                i, snr_db, noise_type
            )
            print(f"  Scene {i:03d} — SNR={snr_db:.1f} dB "
                  f"— {noise_type} — plot saved")
        elif (i + 1) % 20 == 0:
            print(f"  Scene {i+1}/{n_scenes} done")

    # Save manifest
    manifest = os.path.join(OUTPUT_DIR, "scenes_manifest.json")
    with open(manifest, "w") as f:
        json.dump(all_meta, f, indent=2)

    print(f"\nDataset summary:")
    print(f"  Total scenes : {len(all_meta)}")
    print(f"  Hoover       : "
          f"{sum(1 for m in all_meta if m['noise_type']=='Hoover')}")
    print(f"  Conversation : "
          f"{sum(1 for m in all_meta if m['noise_type']=='Conversation')}")
    print(f"  Manifest     : {manifest}")
    return all_meta


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    build_scenes(n_scenes=100, plot_first=5)