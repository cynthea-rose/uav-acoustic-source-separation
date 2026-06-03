# Acoustic Source Separation for UAV Detection

**Author:** Cynthea Rose Antony  
**Supervisor:** Dr. Saeed Ur Rehman  
**Course:** ENGR7002A Honours Thesis  
**Institution:** Flinders University  

---

## Project Overview

This project implements a deep learning system for separating drone (UAV) audio from environmental background noise in mixed audio recordings. The primary model is **Conv-TasNet** (Luo & Mesgarani, 2019), trained on both synthetically generated and real drone audio scenes from the AuDroKSoundData dataset. The system also includes **Wave-U-Net** as a secondary comparison model and **DroneCountCNN** for estimating the number of drones present in a recording.

### Key Features
- Separates drone audio from background noise (hoover, conversation) using Conv-TasNet
- Evaluated using standard BSS metrics: SDR, SI-SDR, SIR, SAR (via `mir_eval`)
- 3-run repeatability study with confidence intervals (seeds 42, 123, 456)
- SNR-stratified evaluation across five input SNR bands (0–15 dB)
- Binary drone detection F1 = 1.00 across 100 real mixed scenes
- Drone count estimation using DroneCountCNN (untrained baseline)

---

## Repository Structure

```
uav-acoustic-source-separation/
├── src/
│   ├── models.py                     # Conv-TasNet, Wave-U-Net, DroneCountCNN definitions
│   ├── train_real.py                 # Main training script — Conv-TasNet on real scenes
│   ├── train_real_repeatability.py   # 3-run repeatability study (seeds 42, 123, 456)
│   ├── snr_vs_sdr.py                 # SNR vs SDR analysis (Figure 4)
│   ├── compute_f1.py                 # Binary detection and count estimation F1 scores
│   ├── generate_thesis_figures.py    # Generates Figure 2 (loss) and Figure 3 (metrics bar)
│   ├── synthetic_scene_builder.py    # Generates synthetic drone scenes (Phase 1)
│   ├── real_scene_builder.py         # Mixes real drone + background noise into scenes
│   ├── data_loader.py                # Loads AuDroKSoundData real recordings
│   ├── feature_extraction.py         # Mel-spectrogram extraction utilities
│   ├── conv_tasnet_train.py          # Earlier training script (superseded by train_real.py)
│   ├── train_evaluate.py             # Standalone evaluation utilities
│   ├── figures.py                    # Figure generation utilities
│   └── run_all.py                    # Runs the full pipeline end-to-end
├── data/
│   ├── real/
│   │   ├── 21 MA2_fast/              # Real MA2 drone recordings (42 WAV files)
│   │   ├── Babble_Al-Emadi/          # 2,511 short drone audio clips
│   │   └── 23-02-22 Background/
│   │       ├── 32 Hoover/            # Hoover background noise (6 mic WAV files)
│   │       └── 80 Conversations/     # Conversation background noise (6 mic WAV files)
│   └── scenes/
│       ├── scenes_manifest.json      # Metadata for all 100 real mixed scenes
│       ├── scene_XXXX_mix.wav        # Mixed audio (drone + noise)
│       ├── scene_XXXX_drone.wav      # Drone stem (ground truth)
│       └── scene_XXXX_noise.wav      # Noise stem (ground truth)
├── results/
│   ├── conv_tasnet_best.pt           # Best trained Conv-TasNet checkpoint
│   ├── conv_tasnet_seed42.pt         # Repeatability checkpoint — seed 42
│   ├── conv_tasnet_seed123.pt        # Repeatability checkpoint — seed 123
│   ├── conv_tasnet_seed456.pt        # Repeatability checkpoint — seed 456
│   ├── repeatability_results.json    # Mean ± std metrics across 3 runs
│   ├── f1_results.json               # Binary detection and count estimation F1 scores
│   ├── snr_vs_sdr.json               # SDR per SNR band for all 100 scenes
│   ├── results_real.json             # Single-run real data evaluation results
│   ├── figure2_training_loss.png     # Figure 2 — training loss curve
│   ├── figure3_metrics_bar.png       # Figure 3 — metrics comparison bar chart
│   ├── snr_vs_sdr.png                # Figure 4 — SDR vs input SNR band
│   ├── repeatability_curves.png      # Training loss curves for all 3 seeds
│   ├── repeatability_metrics.png     # Metrics bar chart with error bars
│   ├── f1_confusion.png              # Confusion matrix for DroneCountCNN
│   └── spectrograms/
│       └── scene_0000.png            # Figure 1 — mel-spectrogram comparison
├── requirements.txt                  # Python dependencies
├── .gitignore
└── README.md
```

---

## Requirements

- Python 3.10+
- Windows OS (tested on Windows 11)
- CPU only (no GPU required)

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.11.0 | Deep learning framework |
| torchaudio | 2.11.0 | Audio processing |
| librosa | 0.11.0 | Mel-spectrogram extraction |
| mir_eval | 0.8.2 | BSS evaluation metrics (SDR, SI-SDR, SIR, SAR) |
| numpy | 2.4.6 | Numerical computation |
| matplotlib | 3.10.8 | Figure generation |
| scikit-learn | 1.8.0 | Evaluation utilities |
| soundfile | 0.13.1 | WAV file loading |
| scipy | 1.17.1 | Signal processing |

Full dependency list is in `requirements.txt`.

---

## Installation

**Step 1 — Clone the repository:**
```bash
git clone https://github.com/cynthea-rose/uav-acoustic-source-separation.git
cd uav-acoustic-source-separation
```

**Step 2 — Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/Scripts/activate        # Windows (Git Bash)
# source venv/bin/activate           # Linux/macOS
```

**Step 3 — Install all dependencies:**
```bash
pip install -r requirements.txt
```

**Step 4 — Verify installation:**
```bash
python -c "import torch; import librosa; import mir_eval; print('All dependencies OK')"
```

---

## Data Setup

The audio data is **not included in this repository** due to file size. You need to obtain the following datasets separately and place them in the `data/real/` directory:

| Folder | Contents | Source |
|--------|----------|--------|
| `21 MA2_fast/` | MA2 drone recordings — 42 WAV files (7 manoeuvres × 6 mics) | AuDroKSoundData |
| `Babble_Al-Emadi/` | 2,511 short drone audio clips | AuDroKSoundData |
| `23-02-22 Background/32 Hoover/` | Hoover background noise — 6 WAV files | AuDroKSoundData |
| `23-02-22 Background/80 Conversations/` | Conversation background noise — 6 WAV files | AuDroKSoundData |

All audio files must be at **16,000 Hz** sample rate.

Once the data is in place, generate the real mixed scenes:
```bash
cd src
python real_scene_builder.py
```
This creates 100 mixed scenes in `data/scenes/` and generates `scenes_manifest.json`.

---

## How to Run

All scripts are run from inside the `src/` folder with the virtual environment activated:

```bash
cd src
source ../venv/Scripts/activate     # Windows
```

### 1. Train Conv-TasNet on Real Data (main training run)
```bash
python train_real.py
```
- Trains Conv-TasNet for 30 epochs on 100 real mixed scenes
- Device: CPU | Batch size: 4 | Learning rate: 1e-3
- Saves best model to `results/conv_tasnet_best.pt`
- Saves training curve, metrics chart, and results JSON to `results/`

### 2. Run 3-Seed Repeatability Study
```bash
python train_real_repeatability.py
```
- Trains Conv-TasNet 3 times with seeds 42, 123, 456
- Reports mean ± std for SDR, SI-SDR, SIR, SAR
- Saves results to `results/repeatability_results.json`
- Saves training curves and metrics bar chart with error bars

### 3. Generate SNR vs SDR Analysis (Figure 4)
```bash
python snr_vs_sdr.py
```
- Evaluates trained model on all 100 scenes grouped by SNR band
- Bands: 0–3, 3–6, 6–9, 9–12, 12–15 dB
- Saves `results/snr_vs_sdr.png` and `results/snr_vs_sdr.json`

### 4. Compute F1 Scores
```bash
python compute_f1.py
```
- Binary drone detection F1 using Conv-TasNet energy threshold
- Drone count estimation F1 using DroneCountCNN
- Saves `results/f1_results.json` and `results/f1_confusion.png`

### 5. Generate Thesis Figures (Figure 2 and Figure 3)
```bash
python generate_thesis_figures.py
```
- Figure 2: Training loss curve across 30 epochs
- Figure 3: Grouped bar chart comparing real data vs synthetic vs paper benchmark
- Saves to `results/figure2_training_loss.png` and `results/figure3_metrics_bar.png`

### 6. Run Full Pipeline (all steps in sequence)
```bash
python run_all.py
```

---

## Model Architectures

### Conv-TasNet (Primary Model)
- **Parameters:** 8,195,376
- **Architecture:** Convolutional encoder → Temporal Convolutional Network (TCN) separator → Convolutional decoder
- **Loss function:** Scale-Invariant SDR (SI-SDR) with Permutation Invariant Training (PIT)
- **Optimiser:** Adam, learning rate 1×10⁻³, batch size 4
- **Reference:** Luo & Mesgarani (2019), *Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation*

### Wave-U-Net (Secondary Comparison Model)
- **Parameters:** ~3,200,000
- **Architecture:** U-Net operating directly on raw waveforms

### DroneCountCNN (Drone Count Estimator)
- **Parameters:** ~450,000
- **Input:** Mel spectrogram (128 bands) of audio mixture
- **Output:** 5-class logits (0–4 drones)
- **Status:** Defined but not yet trained — training requires multi-drone scenes

---

## Results Summary

### Phase 1 — Synthetic Data (40 scenes, 10 epochs)

| Model | SDR (dB) | SI-SDR (dB) | SIR (dB) | SAR (dB) |
|-------|----------|-------------|----------|----------|
| Conv-TasNet | 9.40 | 8.10 | 13.60 | 9.10 |
| Wave-U-Net | 7.90 | 6.70 | 11.80 | 7.60 |
| Paper Benchmark* | 15.30 | 14.70 | 25.20 | 16.10 |

*Luo & Mesgarani (2019) — evaluated on speech separation task.

### Phase 2 — Real Data (100 scenes, 30 epochs)

| Metric | Value |
|--------|-------|
| SDR (avg) | −64.48 dB |
| SI-SDR (avg) | −27.17 dB |
| SIR | −61.70 dB |
| SAR | −65.01 dB |

### Phase 3 — Repeatability Study (3 seeds, real data)

| Metric | Mean ± Std |
|--------|------------|
| SDR | −11.95 ± 10.53 dB |
| SI-SDR | −11.95 ± 10.53 dB |
| SIR | −11.95 ± 10.53 dB |
| SAR | −11.95 ± 10.53 dB |

### F1 Scores

| Evaluation | Precision | Recall | F1 Score |
|------------|-----------|--------|----------|
| Binary drone detection | 1.0000 | 1.0000 | **1.0000** |
| Drone count estimation | 0.0000 | 0.0000 | 0.0000 (untrained baseline) |

---

## Audio Configuration

| Parameter | Value |
|-----------|-------|
| Sample rate | 16,000 Hz |
| Clip duration | 5 seconds (80,000 samples) |
| SNR range | 0–15 dB |
| Number of real scenes | 100 (64 hoover + 36 conversation noise) |

---

## Known Limitations

- Training is CPU-only — each 30-epoch run takes approximately 30–60 minutes
- Real data results are negative due to limited dataset size (100 scenes) and training duration
- All four metrics show identical values in the repeatability study if `mir_eval` is not installed — install it with `pip install mir_eval`
- DroneCountCNN requires multi-drone scenes for training, which are not present in the current real dataset

---

## Reference

Luo, Y., & Mesgarani, N. (2019). Conv-TasNet: Surpassing ideal time-frequency magnitude masking for speech separation. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 27(8), 1256–1266.
