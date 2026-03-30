"""
Synthetic Multi-Drone Scene Builder
Generates mixed audio scenes for UAV acoustic source separation training.
Produces metadata files alongside each mixture.
"""

import os
import json
import random
import numpy as np
import soundfile as sf
import librosa

# ─── Config ───────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16000   # Hz
CLIP_DURATION = 5       # seconds
SNR_RANGE     = (0, 15) # dB
MAX_DRONES    = 4
OUTPUT_DIR    = "data/synthetic"
STEMS_DIR     = "data/stems"


def generate_drone_tone(duration: float, base_freq: float, sr: int) -> np.ndarray:
    """
    Synthesise a realistic drone motor sound:
    fundamental + harmonics + amplitude modulation + slight pitch wobble.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Fundamental + 3 harmonics
    signal = (
        0.6  * np.sin(2 * np.pi * base_freq * t) +
        0.25 * np.sin(2 * np.pi * base_freq * 2 * t) +
        0.10 * np.sin(2 * np.pi * base_freq * 3 * t) +
        0.05 * np.sin(2 * np.pi * base_freq * 4 * t)
    )

    # Amplitude modulation (motor vibration ~8 Hz)
    mod_freq = 8.0
    signal *= (0.85 + 0.15 * np.sin(2 * np.pi * mod_freq * t))

    # Slight pitch wobble
    wobble = 1 + 0.005 * np.sin(2 * np.pi * 0.5 * t)
    signal *= wobble

    # Normalise
    signal = signal / (np.max(np.abs(signal)) + 1e-8)
    return signal.astype(np.float32)


def generate_background_noise(duration: float, sr: int, noise_type: str = "pink") -> np.ndarray:
    """Generate pink or white noise as background."""
    n_samples = int(sr * duration)
    white = np.random.randn(n_samples).astype(np.float32)

    if noise_type == "pink":
        # Approximate pink noise via 1/f shaping
        f = np.fft.rfftfreq(n_samples)
        f[0] = 1e-6  # avoid division by zero
        pink_filter = 1.0 / np.sqrt(f)
        spectrum = np.fft.rfft(white) * pink_filter
        noise = np.fft.irfft(spectrum, n=n_samples).astype(np.float32)
    else:
        noise = white

    noise = noise / (np.max(np.abs(noise)) + 1e-8)
    return noise


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale noise so the mixture is at the target SNR (dB)."""
    signal_power = np.mean(signal ** 2) + 1e-8
    noise_power  = np.mean(noise  ** 2) + 1e-8
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)
    return (signal + scale * noise).astype(np.float32)


def build_scene(scene_id: int, n_drones: int | None = None) -> dict:
    """
    Build one synthetic multi-drone scene.

    Returns a metadata dict describing the scene.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(STEMS_DIR,  exist_ok=True)

    if n_drones is None:
        n_drones = random.randint(1, MAX_DRONES)

    snr_db = random.uniform(*SNR_RANGE)

    # Drone fundamental frequencies (Hz) — typical quadrotor range
    base_freqs = [random.uniform(80, 500) for _ in range(n_drones)]
    drone_types = [f"drone_{i+1}" for i in range(n_drones)]

    # Generate individual drone stems
    stems = []
    for i, freq in enumerate(base_freqs):
        stem = generate_drone_tone(CLIP_DURATION, freq, SAMPLE_RATE)
        # Random amplitude per drone
        amplitude = random.uniform(0.4, 1.0)
        stem = stem * amplitude
        stems.append(stem)

        # Save stem
        stem_path = os.path.join(STEMS_DIR, f"scene_{scene_id:04d}_drone_{i+1}.wav")
        sf.write(stem_path, stem, SAMPLE_RATE)

    # Mix all drone stems
    mixture = np.sum(stems, axis=0) if stems else np.zeros(int(SAMPLE_RATE * CLIP_DURATION))
    mixture = mixture / (np.max(np.abs(mixture)) + 1e-8)

    # Add background noise
    noise = generate_background_noise(CLIP_DURATION, SAMPLE_RATE)
    mixture_noisy = mix_at_snr(mixture, noise, snr_db)

    # Clip to [-1, 1] and save
    mixture_noisy = np.clip(mixture_noisy, -1.0, 1.0)
    mix_path = os.path.join(OUTPUT_DIR, f"scene_{scene_id:04d}_mix.wav")
    sf.write(mix_path, mixture_noisy, SAMPLE_RATE)

    metadata = {
        "scene_id":    scene_id,
        "n_drones":    n_drones,
        "drone_types": drone_types,
        "base_freqs":  [round(f, 2) for f in base_freqs],
        "snr_db":      round(snr_db, 2),
        "mix_path":    mix_path,
    }

    meta_path = os.path.join(OUTPUT_DIR, f"scene_{scene_id:04d}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def build_dataset(n_scenes: int = 50) -> list[dict]:
    """Build a full synthetic dataset of n_scenes scenes."""
    print(f"Building {n_scenes} synthetic scenes …")
    all_meta = []
    for i in range(n_scenes):
        meta = build_scene(scene_id=i)
        all_meta.append(meta)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_scenes} scenes done")

    # Save full dataset manifest
    manifest_path = os.path.join(OUTPUT_DIR, "dataset_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(all_meta, f, indent=2)

    counts = {}
    for m in all_meta:
        n = m["n_drones"]
        counts[n] = counts.get(n, 0) + 1

    print("\nDataset summary:")
    print(f"  Total scenes : {len(all_meta)}")
    for k in sorted(counts):
        print(f"  {k} drone(s)   : {counts[k]} scenes")
    print(f"  Manifest     : {manifest_path}")
    return all_meta


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    build_dataset(n_scenes=40)
