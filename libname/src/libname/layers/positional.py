import torch.nn as nn
import torch
import torch.nn.functional as F

device = ("cuda" if torch.cuda.is_available() else "cpu")

class Positional_encoding(nn.Module):
    def __init__(self ,context_length, d_model):
        super().__init__()
        self.embeddings = nn.Embedding(context_length , d_model)

    def forward(self , x):
        B , T = x.shape
        positions = torch.arange(0 , T , device = x.device).unsqueeze(0).expand(B , T)
        return self.embeddings(positions)



def compute_thetas(seq_len: int, head_dim: int, theta: float = 10000.0, device=None):
    assert head_dim % 2 == 0

    numerators = torch.arange(0, head_dim, 2, dtype=torch.float32,device=device)

    thetas = 1.0 / (theta ** (numerators / head_dim))

    positions = torch.arange(seq_len, dtype=torch.float32, device=device)

    frequencies = torch.outer(positions, thetas)

    freqs_complex = torch.polar(torch.ones_like(frequencies), frequencies)

    return freqs_complex


def rope(x: torch.Tensor, theta: float = 10000.0):
    # Expected input: [B, H, T, D]

    B, H, T, D = x.shape

    assert D % 2 == 0

    # Convert pairs of real dimensions into complex numbers
    x_complex = torch.view_as_complex(x.float().reshape(B, H, T, D // 2, 2))

    # Compute rotation frequencies
    freqs_complex = compute_thetas(seq_len=T,head_dim=D,theta=theta,device=x.device)

    # Shape: [1, 1, T, D//2]
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(0)

    # Apply rotations
    rotated = x_complex * freqs_complex

    # Convert complex numbers back to real numbers
    rotated = torch.view_as_real(rotated)

    # Restore original shape
    return rotated.reshape(B, H, T, D).to(x.dtype)





    
