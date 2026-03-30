"""
Feature Extraction Module
Produces STFT, mel-spectrograms, and summary plots for each audio file.
"""

import os
import json
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf


SAMPLE_RATE  = 16000
N_FFT        = 1024
HOP_LENGTH   = 256
N_MELS       = 128
FMIN         = 20
FMAX         = 8000
PLOTS_DIR    = "outputs/spectrograms"


# ─── Core feature extraction ──────────────────────────────────────────────────

def load_audio(path: str, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load and resample audio to a fixed sample rate."""
    y, sr_orig = librosa.load(path, sr=sr, mono=True)
    return y, sr


def compute_stft(y: np.ndarray, n_fft: int = N_FFT, hop: int = HOP_LENGTH) -> np.ndarray:
    """Short-Time Fourier Transform → magnitude spectrogram."""
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    return np.abs(D)


def compute_mel_spectrogram(
    y: np.ndarray,
    sr: int     = SAMPLE_RATE,
    n_fft: int  = N_FFT,
    hop: int    = HOP_LENGTH,
    n_mels: int = N_MELS,
) -> np.ndarray:
    """Mel-frequency spectrogram (in dB)."""
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop,
        n_mels=n_mels, fmin=FMIN, fmax=FMAX,
    )
    return librosa.power_to_db(S, ref=np.max)


def extract_features(audio_path: str) -> dict:
    """
    Extract all features from an audio file.

    Returns a dict with keys: y, sr, stft_mag, mel_db
    """
    y, sr = load_audio(audio_path)
    stft_mag = compute_stft(y)
    mel_db   = compute_mel_spectrogram(y, sr)

    return {
        "y":        y,
        "sr":       sr,
        "stft_mag": stft_mag,
        "mel_db":   mel_db,
        "path":     audio_path,
    }


# ─── Visualisation ────────────────────────────────────────────────────────────

def plot_mel_spectrogram(
    mel_db: np.ndarray,
    sr: int,
    hop: int,
    title: str = "Mel Spectrogram",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot a single mel-spectrogram."""
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        mel_db, sr=sr, hop_length=hop,
        x_axis="time", y_axis="mel",
        fmin=FMIN, fmax=FMAX, cmap="magma", ax=ax,
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_comparison(
    mix_path: str,
    stem_paths: list[str],
    scene_id: int,
    n_drones: int,
    snr_db: float,
) -> str:
    """
    Side-by-side mel-spectrogram comparison:
    mixture (top) + individual drone stems (below).
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    n_panels = 1 + len(stem_paths)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3 * n_panels))
    if n_panels == 1:
        axes = [axes]

    def _plot(ax, path, label):
        y, sr = load_audio(path)
        mel = compute_mel_spectrogram(y, sr)
        img = librosa.display.specshow(
            mel, sr=sr, hop_length=HOP_LENGTH,
            x_axis="time", y_axis="mel",
            fmin=FMIN, fmax=FMAX, cmap="magma", ax=ax,
        )
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Hz")

    _plot(axes[0], mix_path, f"Scene {scene_id} — Mixture ({n_drones} drones, SNR={snr_db:.1f} dB)")
    for i, sp in enumerate(stem_paths):
        _plot(axes[i + 1], sp, f"Drone {i+1} (ground-truth stem)")

    plt.suptitle(f"Acoustic Source Separation — Scene {scene_id}", fontsize=13, y=1.01)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"scene_{scene_id:04d}_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved spectrogram comparison → {out}")
    return out


def run_feature_extraction(manifest_path: str, max_scenes: int = 5) -> None:
    """
    Load dataset manifest, extract features, and save comparison plots
    for the first `max_scenes` scenes.
    """
    with open(manifest_path) as f:
        all_meta = json.load(f)

    print(f"\nExtracting features for {min(max_scenes, len(all_meta))} scenes …")
    for meta in all_meta[:max_scenes]:
        sid = meta["scene_id"]
        mix_path   = meta["mix_path"]
        stem_paths = [
            f"data/stems/scene_{sid:04d}_drone_{i+1}.wav"
            for i in range(meta["n_drones"])
        ]
        # Filter to stems that actually exist
        stem_paths = [p for p in stem_paths if os.path.exists(p)]

        plot_comparison(
            mix_path   = mix_path,
            stem_paths = stem_paths,
            scene_id   = sid,
            n_drones   = meta["n_drones"],
            snr_db     = meta["snr_db"],
        )

    print(f"\nAll spectrogram plots saved to: {PLOTS_DIR}/")


if __name__ == "__main__":
    run_feature_extraction("data/synthetic/dataset_manifest.json", max_scenes=5)
