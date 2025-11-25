import os
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

import torch
from audiotools import AudioSignal

from inference import get_device, load_dac_model, compute_file_metrics  # <-- added compute_file_metrics

# --- Flask setup ---
app = Flask(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent
UPLOAD_DIR = PROJECT_ROOT / "uploads"
STATIC_DIR = PROJECT_ROOT / "static"
OUTPUT_DIR = STATIC_DIR / "output"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Model setup (load once at startup) ---
device = get_device()
print(f"[DAC] Using device: {device}")

CHECKPOINT_PATH = PROJECT_ROOT / "best" / "dac" / "weights.pth"
if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f"DAC checkpoint not found at {CHECKPOINT_PATH}")

model = load_dac_model(str(CHECKPOINT_PATH), device)
model.eval()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    # POST: file upload
    file = request.files.get("audio_file", None)
    if file is None or file.filename == "":
        return render_template("index.html", error="No file selected.")

    # Save uploaded file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = secure_filename(file.filename)
    input_path = UPLOAD_DIR / f"{timestamp}_{safe_name}"
    file.save(str(input_path))

    steps = []
    steps.append(f"1) File uploaded: {safe_name}")

    # Load audio
    try:
        sig = AudioSignal(str(input_path))
    except Exception as e:
        return render_template(
            "index.html",
            error=f"Could not load audio file: {e}",
        )

    steps.append(f"2) Audio loaded at {sig.sample_rate} Hz")

    # Ensure sample rate matches model (44100 Hz in your DAC by default)
    if sig.sample_rate != model.sample_rate:
        return render_template(
            "index.html",
            error=(
                f"Sample rate mismatch: file is {sig.sample_rate} Hz, "
                f"model expects {model.sample_rate} Hz. "
                "Please upload a 44.1 kHz WAV file."
            ),
        )

    # Prepare tensor
    audio = sig.audio_data.to(device)  # [B, C, T]
    if audio.shape[1] > 1:
        audio = audio.mean(dim=1, keepdim=True)  # downmix to mono

    original_length = audio.shape[-1]

    # Encode (compress) and decode
    with torch.no_grad():
        steps.append("3) Encoding / compressing with DAC...")
        z, codes, latents, commitment_loss, codebook_loss = model.encode(audio)

        steps.append("4) Decoding / reconstructing audio...")
        recon = model.decode(z)
        recon = recon[..., :original_length]  # trim padding if any

    # Back to CPU and save reconstructed audio
    recon = recon.detach().cpu()
    recon_sig = AudioSignal(recon, sig.sample_rate)

    output_filename = f"{timestamp}_reconstructed.wav"
    output_path = OUTPUT_DIR / output_filename
    recon_sig.write(str(output_path))

    steps.append("5) Done! Reconstructed audio saved.")

    # ---- NEW: compute metrics using the saved WAV files ----
    raw_metrics = compute_file_metrics(str(input_path), str(output_path))
    # Map 44k metrics to simple names for the template
    metrics = {
        "mel_loss": raw_metrics.get("mel-44k"),
        "stft_loss": raw_metrics.get("stft-44k"),
        "waveform_loss": raw_metrics.get("waveform-44k"),
        "sisdr_loss": raw_metrics.get("sisdr-44k"),
    }
    steps.append("6) Computed audio metrics (STFT, Mel, L1, SI-SDR).")

    # URL that browser can access
    recon_audio_url = url_for("static", filename=f"output/{output_filename}")

    return render_template(
        "index.html",
        steps=steps,
        original_filename=safe_name,
        recon_audio_url=recon_audio_url,
        metrics=metrics,  # <-- pass metrics to template
    )


if __name__ == "__main__":
    # For local dev only; use a proper WSGI server in production
    app.run(host="0.0.0.0", port=5001, debug=True)