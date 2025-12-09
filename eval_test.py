from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import torch

from inference import infer_and_evaluate, get_device


PROJECT_ROOT = Path(__file__).parent
API_DATA_DIR = PROJECT_ROOT / "api_data"
TEST_DIR = API_DATA_DIR / "test"
CHECKPOINT_PATH = PROJECT_ROOT / "best" / "dac" / "weights.pth"

# Where to put reconstructed test files + metrics
OUTPUT_ROOT = PROJECT_ROOT / "api_data" / "test_out"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

RESULTS_JSON = OUTPUT_ROOT / "test_metrics_per_file.json"
AVG_JSON = OUTPUT_ROOT / "test_metrics_avg.json"


def find_wav_files(test_dir: Path) -> List[Path]:
    """Recursively find all .wav files under test_dir."""
    return sorted(test_dir.rglob("*.wav"))


def evaluate_dataset(
    test_dir: Path,
    checkpoint_path: Path,
    output_root: Path,
    device: torch.device | None = None,
) -> Dict[str, float]:
    """
    Run inference + metrics on all WAV files in test_dir.

    Returns:
        avg_metrics: dict with dataset-averaged metrics.
    """
    if device is None:
        device = get_device()
    print(f"Using device: {device}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    wav_files = find_wav_files(test_dir)
    if not wav_files:
        print(f"No .wav files found under {test_dir}")
        return {}

    print(f"Found {len(wav_files)} test files under {test_dir}")

    all_metrics: List[Dict[str, float]] = []
    per_file_results: Dict[str, Dict[str, float]] = {}

    for idx, wav_path in enumerate(wav_files, start=1):
        rel = wav_path.relative_to(test_dir)
        out_dir = output_root / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_wav = out_dir / f"{wav_path.stem}_recon.wav"

        print(f"[{idx}/{len(wav_files)}] {wav_path}")

        try:
            metrics = infer_and_evaluate(
                checkpoint_path=str(checkpoint_path),
                input_wav=str(wav_path),
                output_wav=str(out_wav),
                metrics_json=None,  # we aggregate ourselves
                device=device,
            )
        except Exception as e:
            print(f"  ! Skipping {wav_path} due to error: {e}")
            continue

        all_metrics.append(metrics)
        per_file_results[str(rel)] = metrics

    if not all_metrics:
        print("No metrics computed (all files failed?).")
        return {}

    # Compute dataset averages
    keys = all_metrics[0].keys()
    avg_metrics: Dict[str, float] = {}
    for k in keys:
        vals = [m[k] for m in all_metrics if k in m]
        if not vals:
            continue
        avg_metrics[k] = float(sum(vals) / len(vals))

    # Save per-file and averaged metrics
    with RESULTS_JSON.open("w") as f:
        json.dump(per_file_results, f, indent=2)
    with AVG_JSON.open("w") as f:
        json.dump(avg_metrics, f, indent=2)

    print("\nPer-file metrics saved to:", RESULTS_JSON)
    print("Average metrics saved to:", AVG_JSON)

    print("\nAverage metrics over test set:")
    for k, v in avg_metrics.items():
        print(f"  {k}: {v:.6f}")

    return avg_metrics


if __name__ == "__main__":
    device = get_device()
    evaluate_dataset(
        test_dir=TEST_DIR,
        checkpoint_path=CHECKPOINT_PATH,
        output_root=OUTPUT_ROOT,
        device=device,
    )