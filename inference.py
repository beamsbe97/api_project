
import os
import json
import argparse
from pathlib import Path

import torch
from audiotools import AudioSignal

from dac.model.dac import DAC
from scripts.evaluate import State
from scripts.train import losses


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
            cleaned_state_dict[k[len("module.") :]] = v
        else:
            cleaned_state_dict[k] = v

    model.load_state_dict(cleaned_state_dict, strict=False)
    model.eval()
    return model


# Lazy-created global metrics state (same as scripts/evaluate.py)
_METRICS_STATE = None


def get_metrics_state():
    """
    Create and cache the metrics State (uses the same losses as training/evaluate).
    """
    global _METRICS_STATE
    if _METRICS_STATE is None:
        waveform_loss = losses.L1Loss()
        stft_loss = losses.MultiScaleSTFTLoss()
        mel_loss = losses.MelSpectrogramLoss()
        sisdr_loss = losses.SISDRLoss()

        _METRICS_STATE = State(
            waveform_loss=waveform_loss,
            stft_loss=stft_loss,
            mel_loss=mel_loss,
            sisdr_loss=sisdr_loss,
        )
    return _METRICS_STATE



def compute_file_metrics(signal_path: str, recons_path: str):
    """
    Compute metrics between two WAV files using the same losses as training,
    on a single sample rate (44.1 kHz).

    Returns a dict of plain Python floats:
      mel-44k, stft-44k, waveform-44k, sisdr-44k
    """
    state = get_metrics_state()

    # Load both files
    signal = AudioSignal(signal_path)
    recons = AudioSignal(recons_path)

    # Resample both to a common SR (model SR, usually 44.1 kHz)
    target_sr = 44100
    x = signal.clone().resample(target_sr)
    y = recons.clone().resample(target_sr)

    # Downmix to mono to ensure shapes match for SISDR
    if x.num_channels > 1:
        x = x.to_mono()
    if y.num_channels > 1:
        y = y.to_mono()

    # Ensure exactly the same number of samples using underlying tensors
    x_len = x.audio_data.shape[-1]
    y_len = y.audio_data.shape[-1]
    min_len = min(x_len, y_len)
    if x_len != min_len:
        x.audio_data = x.audio_data[..., :min_len]
    if y_len != min_len:
        y.audio_data = y.audio_data[..., :min_len]

    out = {}
    with torch.no_grad():
        out["mel-44k"] = float(state.mel_loss(x, y).item())
        out["stft-44k"] = float(state.stft_loss(x, y).item())
        out["waveform-44k"] = float(state.waveform_loss(x, y).item())
        out["sisdr-44k"] = float(state.sisdr_loss(x, y).item())

    return out



def infer_and_evaluate(
    checkpoint_path: str,
    input_wav: str,
    output_wav: str,
    metrics_json: str | None = None,
    device: torch.device | None = None,
):
    """
    Full pipeline:
      - load model
      - run encode+decode
      - save output_wav
      - compute metrics
      - optionally save metrics to JSON
      - return metrics dict (floats)
    """
    if device is None:
        device = get_device()

    model = load_dac_model(checkpoint_path, device)

    # Load input audio
    sig = AudioSignal(input_wav)

    if sig.sample_rate != model.sample_rate:
        raise ValueError(
            f"Input sample rate {sig.sample_rate} != model sample rate {model.sample_rate}. "
            "Resample your audio to the model's sample rate before running inference."
        )

    audio = sig.audio_data.to(device)  # [B, C, T]
    if audio.shape[1] > 1:
        audio = audio.mean(dim=1, keepdim=True)

    original_length = audio.shape[-1]

    # Encode + decode using model.forward (which already does encode+decode)
    with torch.no_grad():
        out = model(
            audio,
            sample_rate=sig.sample_rate,
            n_quantizers=None,
        )
        recon = out["audio"][..., :original_length]

    # Save reconstructed audio
    recon = recon.detach().cpu()
    recon_sig = AudioSignal(recon, sig.sample_rate)

    out_path = Path(output_wav)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    recon_sig.write(str(out_path))

    # Compute metrics using the two WAV paths
    metrics = compute_file_metrics(input_wav, str(out_path))

    # Optionally save metrics to JSON
    if metrics_json is not None:
        mj = Path(metrics_json)
        mj.parent.mkdir(parents=True, exist_ok=True)
        with open(mj, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {mj}")

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DAC inference + metrics on a WAV file."
    )
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
        "--metrics-json",
        type=str,
        default="audio_out/test_metrics.json",
        help="Where to save metrics JSON (empty to skip).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: 'cuda', 'mps', or 'cpu'. Default: auto",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device is None:
        device = get_device()
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    project_root = Path(__file__).parent
    checkpoint_path = str((project_root / args.checkpoint).resolve())
    input_wav = str((project_root / args.input).resolve())
    output_wav = str((project_root / args.output).resolve())
    metrics_json = (
        str((project_root / args.metrics_json).resolve())
        if args.metrics_json
        else None
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(input_wav):
        raise FileNotFoundError(f"Input WAV not found: {input_wav}")

    metrics = infer_and_evaluate(
        checkpoint_path=checkpoint_path,
        input_wav=input_wav,
        output_wav=output_wav,
        metrics_json=metrics_json,
        device=device,
    )

    print(f"Reconstructed audio saved to: {output_wav}")
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
