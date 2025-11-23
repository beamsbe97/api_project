import librosa
import numpy as np

import dac
from audiotools import AudioSignal
import torch

# Download a model
model_path = dac.utils.download(model_type="44khz")
model = dac.DAC.load(model_path)

device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to( device_ )
print(device_)

file_name = "track03_rolling_stone_blues_end"
audio_input_wav_file = "../audio_in/"+ file_name + ".wav"
audio_compressed_file = audio_input_wav_file.split("/")[-1].split(".")[0]+"_compressed" + ".dac"
audio_outpu_wav_file =  "../audio_out/"+ audio_input_wav_file.split("/")[-1].split(".")[0] + "_out.wav"




# Load audio signal file
signal = AudioSignal(audio_input_wav_file)

# Encode audio signal as one long file
# (may run out of GPU memory on long files)
if signal.sample_rate != model.sample_rate:
    np_array_audio = np.array(signal.audio_data)
    print(f"Resampling input audio from {signal.sample_rate} to {model.sample_rate}")
    signal.audio_data = torch.Tensor(librosa.resample(np_array_audio, orig_sr=signal.sample_rate, target_sr=model.sample_rate))
    signal.sample_rate = model.sample_rate


if signal.audio_data.ndim == 2:
        signal.audio_data = signal.audio_data.mean(axis=0)   # convert to single  channel for model encoder


signal.to(model.device)
x = model.preprocess(signal.audio_data, signal.sample_rate)
z, codes, latents, _, _ = model.encode(x)

# Decode audio signal
y = model.decode(z)

# Alternatively, use the `compress` and `decompress` functions
# to compress long files.

signal = signal.cpu()
x = model.compress(signal)

# Save and load to and from disk
x.save(audio_compressed_file)
x = dac.DACFile.load(audio_compressed_file)

# Decompress it back to an AudioSignal
y = model.decompress(x)

# Write to file
y.write(audio_outpu_wav_file)