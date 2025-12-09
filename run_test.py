import json
import time
import argparse
from pathlib import Path

import torch
from audiotools import AudioSignal

from inference import get_device, load_dac_model, compute_file_metrics


# Map logical model names to checkpoints (same convention as in app.py)
MODEL_CHECKPOINTS = {
    "dac_baseline": Path("best") / "dac" / "weights.pth",
    "dac_v1":       Path("best_v1") / "dac" / "weights.pth",
    "dac_v2":       Path("best_v2") / "dac" / "weights.pth",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DAC inference on a test dataset and compute metrics."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dac_baseline",
        choices=list(MODEL_CHECKPOINTS.keys()),
        help="Which logical model to use (mapped to a checkpoint).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="api_data/test",
        help="Directory containing test WAV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="static/output/test_cli",
        help="Where to write reconstructed WAVs and metrics JSON.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device to use: 'cuda', 'mps', or 'cpu'. Default: auto",
    )
    parser.add_argument(
        "--metrics-json",
        type=str,
        default=None,
        help=(
            "Path to metrics JSON file. "
            "Default: <output-dir>/metrics_<model>.json"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    project_root = Path(__file__).parent.resolve()

    # Resolve paths relative to project root
    data_dir = (project_root / args.data_dir).resolve()
    out_root = (project_root / args.output_dir).resolve()

    if args.model not in MODEL_CHECKPOINTS:
        raise ValueError(f"Unknown model '{args.model}'. Available: {list(MODEL_CHECKPOINTS.keys())}")

    ckpt_rel = MODEL_CHECKPOINTS[args.model]
    checkpoint_path = (project_root / ckpt_rel).resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for model '{args.model}': {checkpoint_path}")

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    # Device
    if args.device is None:
        device = get_device()
    else:
        device = torch.device(args.device)
    print(f"[TEST-CLI] Using device: {device}")

    # Load model once
    model = load_dac_model(str(checkpoint_path), device)
    model.eval()
    print(f"[TEST-CLI] Loaded model '{args.model}' from {checkpoint_path}")

    # Collect test WAV files
    wav_files = sorted(data_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {data_dir}")

    model_out_dir = out_root / args.model
    model_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[TEST-CLI] Found {len(wav_files)} test files in {data_dir}")
    print(f"[TEST-CLI] Writing reconstructions to {model_out_dir}")

    # Aggregation structures
    per_file_metrics = []  # list of dicts: filename + metrics
    sums = {
        "mel-44k": 0.0,
        "stft-44k": 0.0,
        "waveform-44k": 0.0,
        "sisdr-44k": 0.0,
    }
    counts = {
        "mel-44k": 0,
        "stft-44k": 0,
        "waveform-44k": 0,
        "sisdr-44k": 0,
    }

    start_all = time.perf_counter()
    processed = 0
    skipped = 0

    for wav_path in wav_files:
        try:
            sig_orig = AudioSignal(str(wav_path))

            # For the model: make sure audio is at model.sample_rate
            sig_for_model = sig_orig
            if sig_for_model.sample_rate != model.sample_rate:
                print(
                    f"[TEST-CLI] {wav_path.name}: resampling "
                    f"{sig_for_model.sample_rate} -> {model.sample_rate}"
                )
                sig_for_model = sig_for_model.resample(model.sample_rate)

            audio = sig_for_model.audio_data.to(device)
            if audio.shape[1] > 1:
                audio = audio.mean(dim=1, keepdim=True)
            original_length = audio.shape[-1]

            with torch.no_grad():
                z, codes, latents, commitment_loss, codebook_loss = model.encode(audio)
                recon = model.decode(z)
                recon = recon[..., :original_length]

            recon = recon.detach().cpu()
            recon_sig = AudioSignal(recon, sig_for_model.sample_rate)

            out_name = f"{args.model}_{wav_path.name}"
            recon_path = model_out_dir / out_name
            recon_sig.write(str(recon_path))

            # Metrics: use original file vs reconstructed;
            # compute_file_metrics will resample both to 44.1 kHz internally.
            m = compute_file_metrics(str(wav_path), str(recon_path))

            per_file_metrics.append(
                {
                    "file": wav_path.name,
                    "mel-44k": m.get("mel-44k"),
                    "stft-44k": m.get("stft-44k"),
                    "waveform-44k": m.get("waveform-44k"),
                    "sisdr-44k": m.get("sisdr-44k"),
                }
            )

            for k in sums.keys():
                v = m.get(k)
                if v is not None:
                    sums[k] += float(v)
                    counts[k] += 1

            processed += 1

        except Exception as e:
            print(f"[TEST-CLI] Error processing {wav_path.name}: {e}")
            skipped += 1
            continue

    elapsed_all = time.perf_counter() - start_all

    # Compute averages
    avg_metrics = {}
    for k in sums.keys():
        if counts[k] > 0:
            avg_metrics[k] = sums[k] / counts[k]
        else:
            avg_metrics[k] = None

    print(
        f"[TEST-CLI] Done. files_total={len(wav_files)}, "
        f"processed={processed}, skipped={skipped}, "
        f"time={elapsed_all:.3f} s"
    )
    print("[TEST-CLI] Average metrics:")
    for k, v in avg_metrics.items():
        if v is None:
            print(f"  {k}: None")
        else:
            print(f"  {k}: {v:.6f}")

    # Build JSON result
    result = {
        "model_name": args.model,
        "checkpoint": str(checkpoint_path),
        "data_dir": str(data_dir),
        "output_dir": str(model_out_dir),
        "device": str(device),
        "files_total": len(wav_files),
        "files_processed": processed,
        "files_skipped": skipped,
        "elapsed_seconds": elapsed_all,
        "avg_metrics": avg_metrics,
        "files": per_file_metrics,
    }

    # Decide metrics JSON path
    if args.metrics_json:
        metrics_path = (project_root / args.metrics_json).resolve()
    else:
        metrics_path = model_out_dir / f"metrics_{args.model}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[TEST-CLI] Metrics JSON written to {metrics_path}")


if __name__ == "__main__":
    main()