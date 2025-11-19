import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class MultiResolutionSTFTLoss(nn.Module):
    """
    Implements the reconstruction loss from the screenshot:
    L = ||x - x_hat||_1 + lambda * Sum(|STFT(x)| - |STFT(x_hat)|)_1
    """
    def __init__(self, 
                 fft_sizes=[1024, 2048, 512], 
                 hop_sizes=[120, 240, 50], 
                 win_lengths=[600, 1200, 240], 
                 time_domain_weight=1.0,
                 stft_weight=1.0): # This is the lambda in the equation
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.time_weight = time_domain_weight
        self.stft_weight = stft_weight # Lambda

    def stft(self, x, fft_size, hop_size, win_length):
        window = torch.hann_window(win_length).to(x.device)
        return torch.stft(x.squeeze(1), n_fft=fft_size, hop_length=hop_size, 
                          win_length=win_length, window=window, 
                          return_complex=True)

    def forward(self, x, x_hat):
        # 1. Time Domain L1 Loss: ||x - x_hat||_1
        loss_time = F.l1_loss(x, x_hat)

        # 2. Multi-Resolution STFT Loss
        loss_stft = 0.0
        for n_fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            # Calculate STFTs
            stft_x = self.stft(x, n_fft, hop, win)
            stft_hat = self.stft(x_hat, n_fft, hop, win)
            
            # Get Magnitudes: |STFT(x)|
            mag_x = torch.abs(stft_x)
            mag_hat = torch.abs(stft_hat)
            
            # Spectral Convergence / Magnitude Loss
            # The equation asks for L1 norm of the difference of magnitudes
            loss_stft += F.l1_loss(mag_x, mag_hat)

        # Total Loss formulation
        total_loss = (self.time_weight * loss_time) + (self.stft_weight * loss_stft)
        
        return total_loss, loss_time, loss_stft
    

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, stride=1, 
                               padding=((kernel_size - 1) * dilation) // 2, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, stride=1, 
                               padding=((kernel_size - 1) * dilation) // 2, dilation=dilation)
        self.act = nn.ELU()

    def forward(self, x):
        skip = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return self.act(x + skip)

class Encoder(nn.Module):
    def __init__(self, c_in=1, c_h=32, c_out=128, strides=[2, 4, 5, 8]):
        super().__init__()
        layers = [nn.Conv1d(c_in, c_h, 7, padding=3), nn.ELU()]
        
        current_channels = c_h
        # Downsampling layers
        for stride in strides:
            layers += [
                nn.Conv1d(current_channels, current_channels * 2, 2 * stride, stride=stride, padding=stride // 2),
                nn.ELU()
            ]
            current_channels *= 2
            
        layers += [nn.Conv1d(current_channels, c_out, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, c_in=128, c_h=32, c_out=1, strides=[8, 5, 4, 2]):
        super().__init__()
        # Reverse channel expansion logic for reconstruction
        current_channels = c_h * (2 ** len(strides))
        
        layers = [nn.Conv1d(c_in, current_channels, 7, padding=3), nn.ELU()]
        
        # Upsampling layers (Transpose Conv)
        for stride in strides:
            layers += [
                nn.ConvTranspose1d(current_channels, current_channels // 2, 2 * stride, stride=stride, padding=stride // 2),
                nn.ELU(),
                ResBlock(current_channels // 2) # Add residual block for audio quality
            ]
            current_channels //= 2
            
        layers += [nn.Conv1d(current_channels, c_out, 7, padding=3), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim):
        super().__init__()
        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

    def forward(self, z):
        # z shape: [B, C, T] -> [B, T, C]
        z_permuted = z.permute(0, 2, 1).contiguous()
        
        # Calculate distances
        d = torch.sum(z_permuted ** 2, dim=2, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * torch.matmul(z_permuted, self.embedding.weight.t())
            
        # Find closest encodings
        min_encoding_indices = torch.argmin(d, dim=2)
        z_q = self.embedding(min_encoding_indices) # [B, T, C]
        
        # Codebook loss (commitment loss)
        loss = torch.mean((z_q.detach() - z_permuted) ** 2) + \
               0.25 * torch.mean((z_q - z_permuted.detach()) ** 2)
        
        # Straight-through estimator
        z_q = z_permuted + (z_q - z_permuted).detach()
        
        # Reshape back to [B, C, T]
        z_q = z_q.permute(0, 2, 1).contiguous()
        return z_q, loss, min_encoding_indices

class ResidualVQ(nn.Module):
    def __init__(self, num_quantizers, n_e, e_dim):
        super().__init__()
        self.layers = nn.ModuleList([VectorQuantizer(n_e, e_dim) for _ in range(num_quantizers)])

    def forward(self, z):
        quantized_out = 0
        residual = z
        total_loss = 0
        
        all_indices = []
        
        for layer in self.layers:
            quantized, loss, indices = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            total_loss += loss
            all_indices.append(indices)
            
        return quantized_out, total_loss, all_indices
    


class RVQAudioAutoEncoder(nn.Module):
    def __init__(self, 
                 input_channels=1, 
                 latent_dim=128, 
                 n_codebook=1024, 
                 n_quantizers=4):
        super().__init__()
        self.encoder = Encoder(c_in=input_channels, c_out=latent_dim)
        self.rvq = ResidualVQ(num_quantizers=n_quantizers, n_e=n_codebook, e_dim=latent_dim)
        self.decoder = Decoder(c_in=latent_dim, c_out=input_channels)

    def forward(self, x):
        # Encode
        latents = self.encoder(x)
        
        # Quantize (with residual steps)
        quantized_latents, vq_loss, indices = self.rvq(latents)
        
        # Decode
        x_hat = self.decoder(quantized_latents)
        
        return x_hat, vq_loss

# ==========================================
# Example Usage / Training Loop Logic
# ==========================================

if __name__ == "__main__":
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = 3e-4
    lambda_stft = 1.0 # The lambda in your screenshot
    
    # Initialize Models
    model = RVQAudioAutoEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize the Loss Module based on screenshot
    criterion = MultiResolutionSTFTLoss(stft_weight=lambda_stft).to(device)

    # Dummy Audio Input (Batch: 4, Channels: 1, Samples: 16000)
    x = torch.randn(4, 1, 16000).to(device)

    # --- Training Step ---
    model.train()
    optimizer.zero_grad()

    # 1. Forward Pass
    x_hat, vq_loss = model(x)

    # 2. Compute Reconstruction Loss (The equation in the image)
    # Total Rec Loss = L1_Time + lambda * L1_STFT_Mag
    rec_loss_total, l1_time, l1_stft = criterion(x, x_hat)

    # 3. Combine losses (Reconstruction + VQ Commitment Loss)
    final_loss = rec_loss_total + vq_loss

    # 4. Backprop
    final_loss.backward()
    optimizer.step()

    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {x_hat.shape}")
    print(f"Total Loss: {final_loss.item():.4f}")
    print(f" -> Time L1 Loss: {l1_time.item():.4f}")
    print(f" -> STFT Loss: {l1_stft.item():.4f}")
    print(f" -> VQ Commitment Loss: {vq_loss.item():.4f}")