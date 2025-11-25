import os
import collections
from torch.serialization import add_safe_globals

import dac
from audiotools import AudioSignal
from audiotools.ml.decorators import default_list

# Allow loading checkpoints that use collections.defaultdict and default_list (PyTorch 2.6+)
add_safe_globals([collections.defaultdict, default_list])

# Load from the DAC folder; CPU-only on Mac
model, _ = dac.DAC.load_from_folder("best", map_location="cpu")
model.to("cpu")

# Path to your test file (relative to project root)
audio_path = "audio_in/audio_test_1.wav"

# Load audio signal file
signal = AudioSignal(audio_path)
signal.to(model.device)

# Mix stereo -> mono if needed (model expects 1 channel)
audio = signal.audio_data  # [batch, channels, time]
if audio.shape[1] > 1:
    audio = audio.mean(dim=1, keepdim=True)

x = model.preprocess(audio, signal.sample_rate)
z, codes, latents, _, _ = model.encode(x)

# Decode
#y = model.decode(z)

# Or use compress/decompress
#signal = signal.cpu()
#compressed = model.compress(signal)
#compressed.save("compressed.dac")

#from dac import DACFile
#compressed = DACFile.load("compressed.dac")
#y = model.decompress(compressed)

# Ensure output directory exists and write file
#os.makedirs("audio_out", exist_ok=True)
#y.write("audio_out/audio_test_1_out.wav")