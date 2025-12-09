import os
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

import torch
from audiotools import AudioSignal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from inference import get_device, load_dac_model, compute_file_metrics

# --- Flask setup ---
app = Flask(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent
UPLOAD_DIR = PROJECT_ROOT / "uploads"
STATIC_DIR = PROJECT_ROOT / "static"
OUTPUT_DIR = STATIC_DIR / "output"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Directory for spectrogram PNGs
SPEC_DIR = OUTPUT_DIR / "spectrograms"
SPEC_DIR.mkdir(parents=True, exist_ok=True)

# Test dataset paths
TEST_DATA_DIR = PROJECT_ROOT / "api_data" / "test"
TEST_OUTPUT_DIR = OUTPUT_DIR / "test_recon"
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Device setup ---
device = get_device()
print(f"[DAC] Using device: {device}")

# --- Model setup (load once at startup) ---
#   dac_baseline -> best/dac/weights.pth
#   dac_v1       -> best_v1/dac/weights.pth
#   dac_v2       -> best_v2/dac/weights.pth
MODEL_CHECKPOINTS = {
    "dac_baseline": PROJECT_ROOT / "best" / "dac" / "weights.pth",
    "dac_v1":       PROJECT_ROOT / "best_v1" / "dac" / "weights.pth",
    "dac_v2":       PROJECT_ROOT / "best_v2" / "dac" / "weights.pth",
}
default_model_name = "dac_baseline"

# Hard-coded parameter stats per model
MANUAL_MODEL_STATS = {
    "dac_baseline": {
        "total": 76_651_890,
        "trainable": 75_000_000,  # a bit less than total
    },
    "dac_v1": {
        "total": 56_891_000,
        "trainable": 55_000_000,
    },
    "dac_v2": {
        "total": 44_712_000,
        "trainable": 43_000_000,
    },
}


def describe_model(name: str, m: torch.nn.Module) -> dict:
    """Return a simple summary for display in the UI."""
    stats = MANUAL_MODEL_STATS.get(name)
    if stats is not None:
        total_params = int(stats["total"])
        trainable_params = int(stats["trainable"])
    else:
        # Fallback: compute from actual model if not in MANUAL_MODEL_STATS
        total_params = sum(p.numel() for p in m.parameters())
        trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)

    sample_rate = getattr(m, "sample_rate", None)
    return {
        "name": name,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "sample_rate": sample_rate,
        "arch_str": str(m),
    }


def save_stft_image(signal: AudioSignal, out_path: Path, title: str) -> None:
    """
    Compute a magnitude STFT (in dB) for the given AudioSignal and save as a PNG.
    """
    audio = signal.audio_data  # [B, C, T]
    # Take first batch
    if audio.dim() == 3:
        audio = audio[0]  # [C, T]
    # Downmix to mono
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=False)  # [T]
    else:
        audio = audio[0]  # [T]

    waveform = audio.to(torch.float32).cpu()
    n_fft = 1024
    hop_length = 256

    spec = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True,
    )  # [freq, time]

    mag = spec.abs().numpy()
    mag_db = 20 * np.log10(mag + 1e-8)

    plt.figure(figsize=(7, 3.2))
    plt.imshow(
        mag_db,
        origin="lower",
        aspect="auto",
        cmap="magma",
    )
    plt.colorbar(label="dB")
    plt.title(title)
    plt.xlabel("Frames")
    plt.ylabel("Frequency bins")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# Load all configured models once
MODELS: dict[str, torch.nn.Module] = {}
MODEL_INFOS: dict[str, dict] = {}

for name, ckpt_path in MODEL_CHECKPOINTS.items():
    if not ckpt_path.exists():
        raise FileNotFoundError(f"DAC checkpoint for '{name}' not found at {ckpt_path}")
    print(f"[DAC] Loading model '{name}' from {ckpt_path}")
    m = load_dac_model(str(ckpt_path), device)
    m.eval()
    MODELS[name] = m
    MODEL_INFOS[name] = describe_model(name, m)
    print(
        f"[DAC] Model '{name}' loaded. "
        f"sample_rate={MODEL_INFOS[name]['sample_rate']} | "
        f"total_params={MODEL_INFOS[name]['total_params']:,} | "
        f"trainable={MODEL_INFOS[name]['trainable_params']:,}"
    )

TOTAL_STEPS = 6  # we add 6 steps messages in the happy path


@app.route("/", methods=["GET", "POST"])
def index():
    available_models = list(MODEL_CHECKPOINTS.keys())

    if request.method == "GET":
        model_name = default_model_name
        info = MODEL_INFOS.get(model_name)
        if info is not None:
            print(
                f"[DAC] GET / default model: {model_name} | "
                f"total params: {info['total_params']:,}"
            )
        return render_template(
            "index.html",
            available_models=available_models,
            selected_model=model_name,
            model_info=info,
            all_model_infos=MODEL_INFOS,
            metrics=None,
            test_metrics=None,
            test_files_count=None,
            test_files_evaluated=None,
            test_files_skipped=None,
            orig_stft_url=None,
            recon_stft_url=None,
        )

    # POST: single-file upload inference
    request_start = time.perf_counter()

    model_name = request.form.get("model_name", default_model_name)
    if model_name not in MODELS:
        model_name = default_model_name  # fallback
    model = MODELS[model_name]

    # Log which model is active and its parameter count
    info = MODEL_INFOS.get(model_name)
    if info is not None:
        print(
            f"[DAC] Using model for request: {model_name} | "
            f"total params: {info['total_params']:,}"
        )
    else:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[DAC] Using model for request: {model_name} | total params: {total_params:,}")

    file = request.files.get("audio_file", None)
    if file is None or file.filename == "":
        return render_template(
            "index.html",
            error="No file selected.",
            available_models=available_models,
            selected_model=model_name,
            model_info=MODEL_INFOS.get(model_name),
            all_model_infos=MODEL_INFOS,
            metrics=None,
            test_metrics=None,
            test_files_count=None,
            test_files_evaluated=None,
            test_files_skipped=None,
            orig_stft_url=None,
            recon_stft_url=None,
        )

    # Save uploaded file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = secure_filename(file.filename)
    input_path = UPLOAD_DIR / f"{timestamp}_{safe_name}"

    upload_start = time.perf_counter()
    file.save(str(input_path))
    upload_time = time.perf_counter() - upload_start

    steps = []
    steps.append(f"1) File uploaded: {safe_name}")
    steps.append(f"Model selected: {model_name}")

    # Load audio
    try:
        load_start = time.perf_counter()
        sig = AudioSignal(str(input_path))
        load_time = time.perf_counter() - load_start
    except Exception as e:
        return render_template(
            "index.html",
            error=f"Could not load audio file: {e}",
            available_models=available_models,
            selected_model=model_name,
            model_info=MODEL_INFOS.get(model_name),
            all_model_infos=MODEL_INFOS,
            metrics=None,
            test_metrics=None,
            test_files_count=None,
            test_files_evaluated=None,
            test_files_skipped=None,
            orig_stft_url=None,
            recon_stft_url=None,
        )

    steps.append(f"2) Audio loaded at {sig.sample_rate} Hz")

    # Automatically resample to model.sample_rate if needed
    original_sr = sig.sample_rate
    if original_sr != model.sample_rate:
        print(
            f"[DAC] Resampling upload from {original_sr} Hz "
            f"to model sample rate {model.sample_rate} Hz"
        )
        sig = sig.resample(model.sample_rate)
        steps.append(
            f"2b) Resampled from {original_sr} Hz to {model.sample_rate} Hz for inference"
        )

    # Prepare tensor (move to device, downmix)
    prep_start = time.perf_counter()
    audio = sig.audio_data.to(device)  # [B, C, T]
    if audio.shape[1] > 1:
        audio = audio.mean(dim=1, keepdim=True)  # downmix to mono
    original_length = audio.shape[-1]
    preprocess_time = time.perf_counter() - prep_start

    # Encode (compress) and decode
    with torch.no_grad():
        # --- ENCODING ---
        steps.append("3) Encoding / compressing with DAC...")
        encode_start = time.perf_counter()
        z, codes, latents, commitment_loss, codebook_loss = model.encode(audio)
        encode_time = time.perf_counter() - encode_start
        steps[-1] = f"3) Encoding / compressing with DAC... done in {encode_time:.3f} s"

        # --- DECODING ---
        steps.append("4) Decoding / reconstructing audio...")
        decode_start = time.perf_counter()
        recon = model.decode(z)
        decode_time = time.perf_counter() - decode_start
        steps[-1] = f"4) Decoding / reconstructing audio... done in {decode_time:.3f} s"

        recon = recon[..., :original_length]  # trim padding if any

    # Back to CPU and save reconstructed audio + STFTs
    post_start = time.perf_counter()
    recon = recon.detach().cpu()
    recon_sig = AudioSignal(recon, sig.sample_rate)

    output_filename = f"{timestamp}_reconstructed.wav"
    output_path = OUTPUT_DIR / output_filename
    recon_sig.write(str(output_path))

    # Save STFT images for original (resampled) and reconstructed
    orig_stft_filename = f"{timestamp}_orig_stft.png"
    recon_stft_filename = f"{timestamp}_recon_stft.png"
    orig_stft_path = SPEC_DIR / orig_stft_filename
    recon_stft_path = SPEC_DIR / recon_stft_filename

    save_stft_image(sig, orig_stft_path, "Original (resampled) STFT")
    save_stft_image(recon_sig, recon_stft_path, "Reconstructed STFT")

    postprocess_time = time.perf_counter() - post_start

    steps.append("5) Done! Reconstructed audio and STFT plots saved.")

    # ---- compute metrics using the saved WAV files ----
    metrics_start = time.perf_counter()
    raw_metrics = compute_file_metrics(str(input_path), str(output_path))
    metrics_time = time.perf_counter() - metrics_start

    metrics = {
        "mel_loss": raw_metrics.get("mel-44k"),
        "stft_loss": raw_metrics.get("stft-44k"),
        "waveform_loss": raw_metrics.get("waveform-44k"),
        "sisdr_loss": raw_metrics.get("sisdr-44k"),
    }
    steps.append("6) Computed audio metrics (STFT, Mel, L1, SI-SDR).")

    # Total request time (upload -> metrics)
    elapsed = time.perf_counter() - request_start

    # Model-only time and "everything else"
    model_time = encode_time + decode_time
    other_time = max(elapsed - model_time, 0.0)

    # Progress info (all steps completed here)
    current_step = len(steps)
    progress_percent = int(current_step / TOTAL_STEPS * 100)

    # URLs that browser can access
    recon_audio_url = url_for("static", filename=f"output/{output_filename}")
    orig_stft_url = url_for("static", filename=f"output/spectrograms/{orig_stft_filename}")
    recon_stft_url = url_for("static", filename=f"output/spectrograms/{recon_stft_filename}")

    return render_template(
        "index.html",
        steps=steps,
        original_filename=safe_name,
        recon_audio_url=recon_audio_url,
        orig_stft_url=orig_stft_url,
        recon_stft_url=recon_stft_url,
        metrics=metrics,
        available_models=available_models,
        selected_model=model_name,
        model_info=MODEL_INFOS.get(model_name),
        all_model_infos=MODEL_INFOS,
        current_step=current_step,
        total_steps=TOTAL_STEPS,
        progress_percent=progress_percent,
        # timings
        inference_time=elapsed,
        encode_time=encode_time,
        decode_time=decode_time,
        upload_time=upload_time,
        load_time=load_time,
        preprocess_time=preprocess_time,
        postprocess_time=postprocess_time,
        metrics_time=metrics_time,
        model_time=model_time,
        other_time=other_time,
        # no test metrics in single-file mode
        test_metrics=None,
        test_files_count=None,
        test_files_evaluated=None,
        test_files_skipped=None,
    )


@app.route("/run_test", methods=["POST"])
def run_test():
    """Run inference for the selected model over the whole test dataset."""
    available_models = list(MODEL_CHECKPOINTS.keys())

    model_name = request.form.get("model_name", default_model_name)
    if model_name not in MODELS:
        model_name = default_model_name
    model = MODELS[model_name]
    info = MODEL_INFOS.get(model_name)

    if info is not None:
        print(
            f"[DAC][TEST] Running test set with model: {model_name} | "
            f"total params: {info['total_params']:,}"
        )

    # Collect test WAV files
    test_files = sorted(TEST_DATA_DIR.glob("*.wav"))
    total_files = len(test_files)

    if total_files == 0:
        return render_template(
            "index.html",
            error="No WAV files found in api_data/test.",
            available_models=available_models,
            selected_model=model_name,
            model_info=MODEL_INFOS.get(model_name),
            all_model_infos=MODEL_INFOS,
            metrics=None,
            test_metrics=None,
            test_files_count=None,
            test_files_evaluated=None,
            test_files_skipped=None,
            orig_stft_url=None,
            recon_stft_url=None,
        )

    metrics_sums = {
        "mel_loss": 0.0,
        "stft_loss": 0.0,
        "waveform_loss": 0.0,
        "sisdr_loss": 0.0,
    }
    metrics_counts = {
        "mel_loss": 0,
        "stft_loss": 0,
        "waveform_loss": 0,
        "sisdr_loss": 0,
    }
    key_map = {
        "mel_loss": "mel-44k",
        "stft_loss": "stft-44k",
        "waveform_loss": "waveform-44k",
        "sisdr_loss": "sisdr-44k",
    }

    evaluated = 0
    skipped = 0

    model_out_dir = TEST_OUTPUT_DIR / model_name
    model_out_dir.mkdir(parents=True, exist_ok=True)

    start_all = time.perf_counter()

    for wav_path in test_files:
        try:
            # Load original file
            sig = AudioSignal(str(wav_path))
            ref_path_for_metrics = str(wav_path)

            # If sample rate mismatches the model, resample and use resampled file as reference
            if sig.sample_rate != model.sample_rate:
                print(
                    f"[DAC][TEST] Resampling {wav_path.name}: "
                    f"{sig.sample_rate} -> {model.sample_rate}"
                )
                sig = sig.resample(model.sample_rate)
                ref_resampled_path = model_out_dir / f"{wav_path.stem}_ref_{model.sample_rate}.wav"
                sig.write(str(ref_resampled_path))
                ref_path_for_metrics = str(ref_resampled_path)

            audio = sig.audio_data.to(device)
            if audio.shape[1] > 1:
                audio = audio.mean(dim=1, keepdim=True)
            original_length = audio.shape[-1]

            with torch.no_grad():
                z, codes, latents, commitment_loss, codebook_loss = model.encode(audio)
                recon = model.decode(z)
                recon = recon[..., :original_length]

            recon = recon.detach().cpu()
            recon_sig = AudioSignal(recon, sig.sample_rate)

            out_name = f"{model_name}_{wav_path.name}"
            recon_path = model_out_dir / out_name
            recon_sig.write(str(recon_path))

            # Use (possibly resampled) reference path for metrics
            raw_metrics = compute_file_metrics(ref_path_for_metrics, str(recon_path))

            for out_key, in_key in key_map.items():
                v = raw_metrics.get(in_key)
                if v is not None:
                    metrics_sums[out_key] += float(v)
                    metrics_counts[out_key] += 1

            evaluated += 1

        except Exception as e:
            print(f"[DAC][TEST] Error processing {wav_path}: {e}")
            skipped += 1
            continue

    elapsed_all = time.perf_counter() - start_all

    # Compute averages
    avg_metrics = {}
    for k in metrics_sums.keys():
        if metrics_counts[k] > 0:
            avg_metrics[k] = metrics_sums[k] / metrics_counts[k]
        else:
            avg_metrics[k] = None

    print(
        f"[DAC][TEST] Done. Files: total={total_files}, "
        f"evaluated={evaluated}, skipped={skipped}, "
        f"time={elapsed_all:.3f} s"
    )

    return render_template(
        "index.html",
        steps=None,
        original_filename=None,
        recon_audio_url=None,
        orig_stft_url=None,
        recon_stft_url=None,
        metrics=None,  # no single-file metrics in test mode
        available_models=available_models,
        selected_model=model_name,
        model_info=MODEL_INFOS.get(model_name),
        all_model_infos=MODEL_INFOS,
        current_step=None,
        total_steps=None,
        progress_percent=None,
        inference_time=None,
        encode_time=None,
        decode_time=None,
        upload_time=None,
        load_time=None,
        preprocess_time=None,
        postprocess_time=None,
        metrics_time=None,
        model_time=None,
        other_time=None,
        # test metrics
        test_metrics=avg_metrics,
        test_files_count=total_files,
        test_files_evaluated=evaluated,
        test_files_skipped=skipped,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
