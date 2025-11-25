import math
from typing import List, Union

import numpy as np
import torch
from torch import nn
from audiotools.ml import BaseModel

from dac.model.base import CodecMixin
from dac.nn.layers import Snake1d, WNConv1d, WNConvTranspose1d
from dac.nn.quantize import ResidualVectorQuantize


class SimpleEncoder(nn.Module):
    """Lightweight 1D CNN encoder for audio."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        strides: List[int] = [4, 4, 4],
        latent_dim: int = None,
    ):
        super().__init__()
        layers = [WNConv1d(in_channels, base_channels, kernel_size=7, padding=3)]
        channels = base_channels

        # Downsampling blocks
        for stride in strides:
            out_channels = channels * 2
            layers += [
                Snake1d(channels),
                WNConv1d(
                    channels,
                    out_channels,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=math.ceil(stride / 2),
                ),
            ]
            channels = out_channels

        if latent_dim is None:
            latent_dim = channels

        layers += [
            Snake1d(channels),
            WNConv1d(channels, latent_dim, kernel_size=3, padding=1),
        ]

        self.model = nn.Sequential(*layers)
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, T]
        return self.model(x)


class SimpleDecoder(nn.Module):
    """Mirror decoder for SimpleEncoder."""

    def __init__(
        self,
        latent_dim: int,
        base_channels: int = 32,
        strides: List[int] = [4, 4, 4],
        out_channels: int = 1,
    ):
        super().__init__()

        # Compute channels forward to know final encoder channels
        channels = base_channels
        for _ in strides:
            channels *= 2

        layers = [WNConv1d(latent_dim, channels, kernel_size=7, padding=3)]

        # Upsampling blocks (reverse of encoder)
        for i, stride in enumerate(reversed(strides)):
            in_ch = channels
            out_ch = channels // 2
            layers += [
                Snake1d(in_ch),
                WNConvTranspose1d(
                    in_ch,
                    out_ch,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=math.ceil(stride / 2),
                ),
            ]
            channels = out_ch

        layers += [
            Snake1d(channels),
            WNConv1d(channels, out_channels, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, D, T']
        return self.model(z)


def init_weights(m: nn.Module):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class SimpleRVQAutoencoder(BaseModel, CodecMixin):
    """
    Lightweight RVQ audio auto-encoder (no GAN).

    Interface is compatible with dac.model.DAC:
    forward(audio_data, sample_rate) -> dict with keys:
        "audio", "z", "codes", "latents",
        "vq/commitment_loss", "vq/codebook_loss"
    """

    def __init__(
        self,
        encoder_dim: int = 32,
        encoder_rates: List[int] = [4, 4, 4],
        latent_dim: int = None,
        n_codebooks: int = 4,
        codebook_size: int = 512,
        codebook_dim: Union[int, List[int]] = 8,
        sample_rate: int = 44100,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            # similar heuristic as DAC: double channels each stage
            latent_dim = encoder_dim * (2 ** len(encoder_rates))
        self.latent_dim = latent_dim

        # Effective hop length of encoder
        self.hop_length = int(np.prod(encoder_rates))

        # Encoder / quantizer / decoder
        self.encoder = SimpleEncoder(
            in_channels=1,
            base_channels=encoder_dim,
            strides=encoder_rates,
            latent_dim=latent_dim,
        )

        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=False,
        )

        self.decoder = SimpleDecoder(
            latent_dim=latent_dim,
            base_channels=encoder_dim,
            strides=encoder_rates,
            out_channels=1,
        )

        self.apply(init_weights)
        # Required by CodecMixin (used for chunked compress/decompress)
        self.delay = self.get_delay()

    # -------- core API (similar to dac.model.dac.DAC) --------

    def preprocess(self, audio_data: torch.Tensor, sample_rate: int = None):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert (
            sample_rate == self.sample_rate
        ), f"Expected sample_rate={self.sample_rate}, got {sample_rate}"

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))
        return audio_data

    def encode(
        self,
        audio_data: torch.Tensor,
        n_quantizers: int = None,
    ):
        """
        Encode audio into quantized latents.

        audio_data: [B, 1, T]
        """
        z = self.encoder(audio_data)
        z_q, codes, latents, commitment_loss, codebook_loss = self.quantizer(
            z, n_quantizers
        )
        return z_q, codes, latents, commitment_loss, codebook_loss

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode quantized latents to waveform."""
        return self.decoder(z)

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = None,
        n_quantizers: int = None,
    ):
        """
        Forward pass.

        Returns
        -------
        {
            "audio": reconstructed audio [B, 1, T],
            "z": quantized latent [B, D, T'],
            "codes": [B, N, T'],
            "latents": [B, N*D, T'],
            "vq/commitment_loss": Tensor[1],
            "vq/codebook_loss": Tensor[1],
        }
        """
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)

        z, codes, latents, commitment_loss, codebook_loss = self.encode(
            audio_data, n_quantizers
        )

        x_hat = self.decode(z)

        return {
            "audio": x_hat[..., :length],
            "z": z,
            "codes": codes,
            "latents": latents,
            "vq/commitment_loss": commitment_loss,
            "vq/codebook_loss": codebook_loss,
        }


# ...existing code...


if __name__ == "__main__":

    import torch

    # Instantiate model
    model = SimpleRVQAutoencoder(
        encoder_dim=32,
        encoder_rates=[4, 4, 4],
        n_codebooks=4,
        codebook_size=512,
        codebook_dim=8,
        sample_rate=44100,
    )
    model.eval()

    print("Model:\n", model)
    print("\nNumber of parameters:",
          sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Dummy input: 1 second of mono audio at 44.1 kHz
    dummy_audio = torch.randn(1, 1, 44100)

    with torch.no_grad():
        out = model(dummy_audio, sample_rate=44100)

    print("\nForward pass successful.")
    print("Input shape:       ", dummy_audio.shape)
    print("Reconstructed shape", out["audio"].shape)
    print("Latent z shape:    ", out["z"].shape)
    print("Codes shape:       ", out["codes"].shape)
    print("Latents shape:     ", out["latents"].shape)
    print("Commitment loss:   ", out["vq/commitment_loss"].item())
    print("Codebook loss:     ", out["vq/codebook_loss"].item())