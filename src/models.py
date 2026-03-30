"""
Deep Learning Models for Acoustic Source Separation
Implements Conv-TasNet and Wave-U-Net (simplified) in PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Conv-TasNet ──────────────────────────────────────────────────────────────

class ConvTasNetEncoder(nn.Module):
    """1D convolutional encoder: maps waveform → latent representation."""
    def __init__(self, enc_dim: int = 256, kernel_size: int = 20, stride: int = 10):
        super().__init__()
        self.conv = nn.Conv1d(
            1, enc_dim, kernel_size=kernel_size, stride=stride, bias=False
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T)
        return self.relu(self.conv(x))   # → (B, enc_dim, L)


class TemporalConvBlock(nn.Module):
    """Depthwise-separable dilated temporal convolutional block."""
    def __init__(self, in_ch: int, hidden_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden_ch, 1),
            nn.PReLU(),
            nn.GroupNorm(1, hidden_ch),
            nn.Conv1d(
                hidden_ch, hidden_ch, kernel_size,
                dilation=dilation, padding=padding, groups=hidden_ch,
            ),
            nn.PReLU(),
            nn.GroupNorm(1, hidden_ch),
            nn.Conv1d(hidden_ch, in_ch, 1),
        )
        self.skip_conv = nn.Conv1d(in_ch, in_ch, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        return x + out, self.skip_conv(out)


class TemporalConvNet(nn.Module):
    """Stack of dilated TCN blocks with skip connections."""
    def __init__(
        self,
        in_ch: int       = 256,
        hidden_ch: int   = 512,
        kernel_size: int = 3,
        n_blocks: int    = 8,
        n_repeats: int   = 3,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(n_repeats):
            for i in range(n_blocks):
                dilation = 2 ** i
                self.blocks.append(
                    TemporalConvBlock(in_ch, hidden_ch, kernel_size, dilation)
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_sum = 0
        for block in self.blocks:
            x, skip = block(x)
            skip_sum = skip_sum + skip
        return skip_sum


class ConvTasNet(nn.Module):
    """
    Simplified Conv-TasNet for UAV source separation.
    Luo & Mesgarani (2019) — time-domain convolutional encoder-decoder.
    """
    def __init__(
        self,
        n_sources: int   = 2,
        enc_dim: int     = 256,
        hidden_ch: int   = 256,
        kernel_size: int = 3,
        n_blocks: int    = 8,
        n_repeats: int   = 3,
    ):
        super().__init__()
        self.n_sources = n_sources

        self.encoder = ConvTasNetEncoder(enc_dim=enc_dim, kernel_size=20, stride=10)
        self.layer_norm = nn.GroupNorm(1, enc_dim)
        self.bottleneck = nn.Conv1d(enc_dim, hidden_ch, 1)
        self.tcn = TemporalConvNet(
            in_ch=hidden_ch, hidden_ch=hidden_ch * 2,
            kernel_size=kernel_size, n_blocks=n_blocks, n_repeats=n_repeats,
        )
        self.mask_net = nn.Sequential(
            nn.Conv1d(hidden_ch, enc_dim * n_sources, 1),
            nn.ReLU(),
        )
        self.decoder = nn.ConvTranspose1d(enc_dim, 1, kernel_size=20, stride=10, bias=False)

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mixture: (B, 1, T) — raw waveform mixture
        Returns:
            sources: (B, n_sources, T) — separated waveforms
        """
        enc   = self.encoder(mixture)              # (B, enc_dim, L)
        normd = self.layer_norm(enc)
        feat  = self.bottleneck(normd)             # (B, hidden, L)
        skip  = self.tcn(feat)                     # (B, hidden, L)
        masks = self.mask_net(skip)                # (B, enc_dim*S, L)

        B, _, L = enc.shape
        masks = masks.view(B, self.n_sources, -1, L)  # (B, S, enc_dim, L)
        enc   = enc.unsqueeze(1)                       # (B, 1, enc_dim, L)

        masked = masks * enc                           # (B, S, enc_dim, L)
        masked = masked.view(B * self.n_sources, -1, L)

        decoded = self.decoder(masked)                 # (B*S, 1, T')
        T_out   = mixture.shape[-1]
        decoded = decoded[..., :T_out]

        sources = decoded.view(B, self.n_sources, T_out)
        return sources


# ─── Wave-U-Net ───────────────────────────────────────────────────────────────

class WaveUNetDownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 15):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.norm = nn.InstanceNorm1d(out_ch)
        self.act  = nn.LeakyReLU(0.2)
        self.pool = nn.AvgPool1d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.act(self.norm(self.conv(x)))
        return self.pool(skip), skip


class WaveUNetUpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, kernel_size: int = 15):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.conv = nn.Conv1d(in_ch + skip_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.norm = nn.InstanceNorm1d(out_ch)
        self.act  = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Crop/pad to match skip length
        if x.shape[-1] > skip.shape[-1]:
            x = x[..., :skip.shape[-1]]
        elif x.shape[-1] < skip.shape[-1]:
            skip = skip[..., :x.shape[-1]]
        x = torch.cat([x, skip], dim=1)
        return self.act(self.norm(self.conv(x)))


class WaveUNet(nn.Module):
    """
    Simplified Wave-U-Net for UAV source separation.
    Stoller et al. (2018) — multi-scale encoder-decoder with skip connections.
    """
    def __init__(self, n_sources: int = 2, base_ch: int = 24, n_levels: int = 5):
        super().__init__()
        self.n_sources = n_sources

        # Encoder (downsampling path)
        self.down_blocks = nn.ModuleList()
        in_ch = 1
        channels = []
        for i in range(n_levels):
            out_ch = base_ch * (i + 1)
            self.down_blocks.append(WaveUNetDownBlock(in_ch, out_ch))
            channels.append(out_ch)
            in_ch = out_ch

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_ch, in_ch * 2, 15, padding=7),
            nn.LeakyReLU(0.2),
        )
        in_ch = in_ch * 2

        # Decoder (upsampling path)
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(n_levels)):
            out_ch = channels[i]
            self.up_blocks.append(WaveUNetUpBlock(in_ch, channels[i], out_ch))
            in_ch = out_ch

        # Output conv → n_sources channels, tanh activation
        self.out_conv = nn.Conv1d(in_ch, n_sources, 1)

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mixture: (B, 1, T)
        Returns:
            sources: (B, n_sources, T)
        """
        skips = []
        x = mixture
        for block in self.down_blocks:
            x, skip = block(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for block, skip in zip(self.up_blocks, reversed(skips)):
            x = block(x, skip)

        # Trim/pad to input length
        T = mixture.shape[-1]
        out = self.out_conv(x)
        if out.shape[-1] > T:
            out = out[..., :T]
        elif out.shape[-1] < T:
            out = F.pad(out, (0, T - out.shape[-1]))

        return torch.tanh(out)


# ─── Drone Count CNN ──────────────────────────────────────────────────────────

class DroneCountCNN(nn.Module):
    """
    Lightweight CNN classifier that estimates the number of active drones
    (0 – MAX_DRONES) from a mel-spectrogram.
    """
    def __init__(self, n_mels: int = 128, n_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, T_frames)
        return self.classifier(self.features(x))


# ─── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    B, T = 2, 16000 * 5   # batch of 2, 5-second clips
    x = torch.randn(B, 1, T)

    print("─" * 55)
    print("Conv-TasNet forward pass …")
    model_ct = ConvTasNet(n_sources=2)
    out_ct = model_ct(x)
    n_params = sum(p.numel() for p in model_ct.parameters() if p.requires_grad)
    print(f"  Input:  {tuple(x.shape)}")
    print(f"  Output: {tuple(out_ct.shape)}")
    print(f"  Params: {n_params:,}")

    print("─" * 55)
    print("Wave-U-Net forward pass …")
    model_wu = WaveUNet(n_sources=2)
    out_wu = model_wu(x)
    n_params = sum(p.numel() for p in model_wu.parameters() if p.requires_grad)
    print(f"  Input:  {tuple(x.shape)}")
    print(f"  Output: {tuple(out_wu.shape)}")
    print(f"  Params: {n_params:,}")

    print("─" * 55)
    print("DroneCountCNN forward pass …")
    mel = torch.randn(B, 1, 128, 157)   # typical mel-spectrogram shape
    model_cnn = DroneCountCNN()
    logits = model_cnn(mel)
    print(f"  Input:  {tuple(mel.shape)}")
    print(f"  Output: {tuple(logits.shape)}  (logits over 0-4 drones)")
    print("─" * 55)
    print("All model sanity checks passed.")
