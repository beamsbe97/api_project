import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from model import *

# --- Imports from your codebase environment ---
# We assume audiotools is installed as seen in your train.py
from audiotools import AudioSignal
from audiotools.data.datasets import AudioLoader, AudioDataset
from audiotools.data import transforms

# --- 1. The Lightweight Model (Updated API) ---
# Updated to return a dict matching train.py's expectation: out["audio"]
class RVQAudioAutoEncoder(nn.Module):
    def __init__(self, input_channels=1, latent_dim=128, n_codebook=1024, n_quantizers=4):
        super().__init__()
        self.encoder = Encoder(c_in=input_channels, c_out=latent_dim) # defined in prev response
        self.rvq = ResidualVQ(num_quantizers=n_quantizers, n_e=n_codebook, e_dim=latent_dim) # defined in prev response
        self.decoder = Decoder(c_in=latent_dim, c_out=input_channels) # defined in prev response

    def forward(self, x, sample_rate=None):
        # train.py passes sample_rate, we accept it to match the signature but ignore it
        latents = self.encoder(x)
        quantized_latents, vq_loss, _ = self.rvq(latents)
        x_hat = self.decoder(quantized_latents)
        
        # Return dictionary matching the structure used in train.py
        return {
            "audio": x_hat,
            "vq/commitment_loss": vq_loss
        }

# --- 2. The Exact Loss Function from Screenshot ---
class SpecificReconstructionLoss(nn.Module):
    def __init__(self, lambda_stft=1.0):
        super().__init__()
        self.lambda_stft = lambda_stft
        # Standard multi-res settings
        self.fft_sizes = [1024, 2048, 512]
        self.hop_sizes = [120, 240, 50]
        self.win_lengths = [600, 1200, 240]

    def forward(self, x, x_hat):
        # 1. Time Domain L1
        loss_time = F.l1_loss(x, x_hat)

        # 2. Multi-Res STFT L1 Magnitude Loss
        loss_stft = 0.0
        for n_fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            window = torch.hann_window(win).to(x.device)
            
            # We must use return_complex=True and calculate mag manually for consistency
            stft_x = torch.stft(x.squeeze(1), n_fft=n_fft, hop_length=hop, win_length=win, window=window, return_complex=True)
            stft_hat = torch.stft(x_hat.squeeze(1), n_fft=n_fft, hop_length=hop, win_length=win, window=window, return_complex=True)
            
            # L1 distance of magnitudes
            loss_stft += F.l1_loss(torch.abs(stft_x), torch.abs(stft_hat))

        total_loss = loss_time + (self.lambda_stft * loss_stft)
        return total_loss, loss_time, loss_stft

# --- 3. Data Loading (Adapted from train.py) ---
def build_dataloader(folder_path, sample_rate=16000, batch_size=8, num_workers=4):
    # Logic adapted from build_dataset in train.py
    
    # 1. Find files
    loader = AudioLoader(sources=[folder_path])
    
    # 2. Define transforms (e.g. random crop to 1 sec)
    # In train.py, this is done via argbind, here we do it explicitly
    transform = transforms.Compose(
        transforms.RescaleAudio(sample_rate),
        transforms.RandomCrop(sample_rate), # Crop to 1 second
    )
    
    # 3. Create Dataset
    dataset = AudioDataset(loader, sample_rate, transform=transform)
    
    # 4. Create DataLoader
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn=dataset.collate)

# --- 4. Main Training Loop ---
def train(folder_path, save_path="ckpt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize
    model = RVQAudioAutoEncoder().to(device)
    optimizer = AdamW(model.parameters(), lr=3e-4)
    criterion = SpecificReconstructionLoss(lambda_stft=1.0).to(device)
    
    dataloader = build_dataloader(folder_path)
    
    print("Starting training without GAN...")
    
    for epoch in range(10): # Example loop
        model.train()
        for i, batch in enumerate(dataloader):
            # audiotools loading logic from train_loop in train.py
            # batch is a dict containing 'signal'
            
            # Move signals to device
            signal = batch['signal'].to(device)
            
            # Forward Pass
            # Note: model.forward accepts sample_rate to match train.py signature
            out = model(signal.audio_data, signal.sample_rate)
            
            x_hat = out["audio"]
            vq_loss = out["vq/commitment_loss"]
            
            # Calculate Reconstruction Loss (The screenshot equation)
            rec_loss, l1_time, l1_stft = criterion(signal.audio_data, x_hat)
            
            # Final Loss = Rec Loss + VQ Loss (No GAN/Discriminator)
            loss = rec_loss + vq_loss
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f"Ep {epoch} It {i} | Total: {loss.item():.4f} | Time: {l1_time.item():.4f} | STFT: {l1_stft.item():.4f} | VQ: {vq_loss.item():.4f}")

        # Save checkpoint logic
        torch.save(model.state_dict(), f"{save_path}/model_ep{epoch}.pth")

if __name__ == "__main__":
    # Replace with your actual audio folder path
    audio_folder = "./my_audio_data" 
    os.makedirs(audio_folder, exist_ok=True) # safety check
    
    train(audio_folder)