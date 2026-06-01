"""
AuDroKSoundData — Real Dataset Loader
Loads real UAV recordings from the 23-02-22 Measurements dataset.

Dataset structure:
    data/real/
    └── [session] [drone_model]_[speed]/   e.g. 21 MA2_fast
        └── [manoeuvre]/                    e.g. 1 Takeoff
            └── [session]-[code]-Mi[n].wav  e.g. 21-TO-Mi1.wav
"""

import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE  = 16000
CLIP_SAMPLES = SAMPLE_RATE * 5   # 5 seconds
REAL_DATA_DIR = "../data/real"

# Manoeuvre code mapping
MANOEUVRE_CODES = {
    "1 Takeoff":             "TO",
    "2 HoverAfterTakeoff":   "HA",
    "3 FlyoverForward":      "FF",
    "4 HoverFarSide":        "HF",
    "5 FlyoverBack":         "FB",
    "6 HoverBeforeLanding":  "HB",
    "7 Landing":             "LA",
}


# ── Helper Functions ──────────────────────────────────────────────────────────

def parse_folder_name(folder_name: str) -> dict:
    """
    Parse drone session folder name into components.
    Example: '21 MA2_fast' -> {session: 21, model: MA2, speed: fast}
    """
    parts = folder_name.split(" ", 1)
    session = int(parts[0])
    model_speed = parts[1].rsplit("_", 1)
    model = model_speed[0]
    speed = model_speed[1] if len(model_speed) > 1 else "unknown"
    return {"session": session, "model": model, "speed": speed}


def load_wav(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load a wav file and resample to target sample rate."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


def pad_or_crop(y: np.ndarray, length: int = CLIP_SAMPLES) -> np.ndarray:
    """Pad or crop audio to a fixed length."""
    if len(y) >= length:
        return y[:length]
    return np.pad(y, (0, length - len(y)))


def normalise(y: np.ndarray) -> np.ndarray:
    """Normalise audio to [-1, 1]."""
    peak = np.max(np.abs(y)) + 1e-8
    return y / peak


# ── Dataset Class ─────────────────────────────────────────────────────────────

class AuDroKDataset(Dataset):
    """
    PyTorch Dataset for the AuDroKSoundData real UAV recordings.

    Each sample returns:
        - mixture:  (1, T) tensor  — average of all microphones
        - label:    int            — drone model index
        - metadata: dict           — session, model, speed, manoeuvre
    """

    def __init__(
        self,
        data_dir:   str  = REAL_DATA_DIR,
        sr:         int  = SAMPLE_RATE,
        microphone: int  = 1,
        use_all_mics: bool = False,
    ):
        self.data_dir     = data_dir
        self.sr           = sr
        self.microphone   = microphone
        self.use_all_mics = use_all_mics
        self.samples      = []
        self.label_map    = {}

        self._scan_dataset()

    def _scan_dataset(self):
        """Walk the dataset folder and collect all valid wav file paths."""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}\n"
                f"Please download the dataset and place it in {self.data_dir}"
            )

        label_idx = 0
        for drone_folder in sorted(os.listdir(self.data_dir)):
            drone_path = os.path.join(self.data_dir, drone_folder)
            if not os.path.isdir(drone_path):
                continue

            # Parse folder name
            try:
                info = parse_folder_name(drone_folder)
            except Exception:
                continue

            # Create label for this drone model if not seen before
            model_key = f"{info['model']}_{info['speed']}"
            if model_key not in self.label_map:
                self.label_map[model_key] = label_idx
                label_idx += 1

            label = self.label_map[model_key]

            # Walk manoeuvre subfolders
            for manoeuvre in sorted(os.listdir(drone_path)):
                manoeuvre_path = os.path.join(drone_path, manoeuvre)
                if not os.path.isdir(manoeuvre_path):
                    continue

                # Collect wav files
                wav_files = sorted([
                    f for f in os.listdir(manoeuvre_path)
                    if f.endswith(".wav")
                ])

                if not wav_files:
                    continue

                self.samples.append({
                    "drone_folder":    drone_folder,
                    "manoeuvre":       manoeuvre,
                    "manoeuvre_path":  manoeuvre_path,
                    "wav_files":       wav_files,
                    "label":           label,
                    "model":           info["model"],
                    "speed":           info["speed"],
                    "session":         info["session"],
                })

        print(f"AuDroKDataset loaded:")
        print(f"  Total samples : {len(self.samples)}")
        print(f"  Drone classes : {len(self.label_map)}")
        for k, v in self.label_map.items():
            print(f"    [{v}] {k}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        manoeuvre_path = sample["manoeuvre_path"]
        wav_files      = sample["wav_files"]

        if self.use_all_mics:
            # Load all microphones and average them
            signals = []
            for wav_file in wav_files:
                path = os.path.join(manoeuvre_path, wav_file)
                y = load_wav(path, self.sr)
                y = pad_or_crop(y)
                y = normalise(y)
                signals.append(y)
            mixture = np.mean(signals, axis=0)
        else:
            # Load a single microphone
            mic_file = wav_files[min(self.microphone - 1, len(wav_files) - 1)]
            path     = os.path.join(manoeuvre_path, mic_file)
            mixture  = load_wav(path, self.sr)
            mixture  = pad_or_crop(mixture)
            mixture  = normalise(mixture)

        mixture_tensor = torch.tensor(mixture, dtype=torch.float32).unsqueeze(0)

        metadata = {
            "model":      sample["model"],
            "speed":      sample["speed"],
            "session":    sample["session"],
            "manoeuvre":  sample["manoeuvre"],
        }

        return mixture_tensor, sample["label"], metadata


# ── Quick Test ────────────────────────────────────────────────────────────────

def explore_dataset(data_dir: str = REAL_DATA_DIR) -> None:
    """
    Print a summary of the dataset structure and
    load one sample to verify everything works.
    """
    print("=" * 55)
    print("AuDroKSoundData — Dataset Explorer")
    print("=" * 55)

    if not os.path.exists(data_dir):
        print(f"ERROR: {data_dir} not found.")
        return

    total_files = 0
    for drone_folder in sorted(os.listdir(data_dir)):
        drone_path = os.path.join(data_dir, drone_folder)
        if not os.path.isdir(drone_path):
            continue

        manoeuvres = [
            m for m in os.listdir(drone_path)
            if os.path.isdir(os.path.join(drone_path, m))
        ]
        wav_count = sum(
            len([f for f in os.listdir(os.path.join(drone_path, m))
                 if f.endswith(".wav")])
            for m in manoeuvres
        )
        total_files += wav_count
        print(f"\n  Drone: {drone_folder}")
        print(f"    Manoeuvres : {len(manoeuvres)}")
        print(f"    WAV files  : {wav_count}")
        for m in sorted(manoeuvres):
            files = os.listdir(os.path.join(drone_path, m))
            wavs  = [f for f in files if f.endswith(".wav")]
            print(f"      {m}: {len(wavs)} files")

    print(f"\n  Total WAV files: {total_files}")
    print("=" * 55)

    # Load one sample and print its properties
    print("\nLoading one sample to verify...")
    try:
        dataset = AuDroKDataset(data_dir=data_dir, use_all_mics=True)
        if len(dataset) > 0:
            mixture, label, meta = dataset[0]
            print(f"\nSample loaded successfully:")
            print(f"  Shape    : {tuple(mixture.shape)}")
            print(f"  Label    : {label}")
            print(f"  Model    : {meta['model']}")
            print(f"  Speed    : {meta['speed']}")
            print(f"  Manoeuvre: {meta['manoeuvre']}")
            print(f"  Min/Max  : {mixture.min():.3f} / {mixture.max():.3f}")
            print("\nData loader is working correctly!")
    except Exception as e:
        print(f"ERROR loading sample: {e}")


if __name__ == "__main__":
    explore_dataset()

RealScenesDataset = AuDroKDataset