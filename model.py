import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
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