import argparse
import os
from pathlib import Path

import torch
from audiotools import AudioSignal

from dac.model.dac import DAC


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_dac_model(checkpoint_path: str, device: torch.device) -> DAC:
    """
    Load the trained DAC model from a weights.pth checkpoint.
    """
    print(f"Loading DAC checkpoint from: {checkpoint_path}")
    model = DAC().to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Handle both pure state_dict or dict with "state_dict"
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Remove possible "module." prefix from DistributedDataParallel
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned_state_dict[k[len("module."):]] = v
        else:
            cleaned_state_dict[k] = v

    model.load_state_dict(cleaned_state_dict, strict=False)
    model.eval()
    return model


def run_inference(
    model: DAC,
    input_wav: str,
    output_wav: str,
    device: torch.device,
):
    # Load input audio
    print(f"Loading input audio: {input_wav}")
    sig = AudioSignal(input_wav)

    # Ensure sample rate matches model
    if sig.sample_rate != model.sample_rate:
        raise ValueError(
            f"Input sample rate {sig.sample_rate} != model.sample rate {model.sample_rate}. "
            "Resample your audio to the model's sample rate before running inference."
        )

    # Get audio tensor [B, C, T] and move to device
    audio = sig.audio_data.to(device)  # shape: [batch, channels, time]

    # Downmix to mono if needed
    if audio.shape[1] > 1:
        audio = audio.mean(dim=1, keepdim=True)

    with torch.no_grad():
        out = model(
            audio,
            sample_rate=sig.sample_rate,
            n_quantizers=None,  # use all codebooks
        )
        recon = out["audio"]  # [B, 1, T]

    # Move back to CPU for saving
    recon = recon.detach().cpu()

    # Wrap in AudioSignal and save
    recon_sig = AudioSignal(recon, sig.sample_rate)

    out_path = Path(output_wav)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving reconstructed audio to: {output_wav}")
    recon_sig.write(output_wav)


def parse_args():
    parser = argparse.ArgumentParser(description="DAC inference: encode+decode a WAV file.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best/dac/weights.pth",
        help="Path to DAC weights checkpoint (weights.pth).",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="audio_in/test.wav",
        help="Path to input WAV file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="audio_out/test_recon.wav",
        help="Path to output reconstructed WAV file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cuda', 'mps', or 'cpu'. Default: auto",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device is None:
        device = get_device()
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Resolve paths relative to project root
    project_root = Path(__file__).parent
    checkpoint_path = str((project_root / args.checkpoint).resolve())
    input_wav = str((project_root / args.input).resolve())
    output_wav = str((project_root / args.output).resolve())

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(input_wav):
        raise FileNotFoundError(f"Input WAV not found: {input_wav}")

    model = load_dac_model(checkpoint_path, device)
    run_inference(model, input_wav, output_wav, device)


if __name__ == "__main__":
    main()