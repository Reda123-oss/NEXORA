import torch.nn as nn
import torch
import torch.nn.functional as F

device = ("cuda" if torch.cuda.is_available() else "cpu")

class Positional_encoding(nn.Module):
    def __init__(self ,context_length, d_model):
        super().__init__()
        self.embeddings = nn.Embedding(context_length , d_model)

    def forward(self , x):
        B , T = x.shape[ :2]
        positions = torch.arange(0 , T , device = x.device).unsqueeze(0).expand(B , T)
        return self.embeddings(positions)



def compute_thetas(
    seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
    device=None,
    start_pos: int = 0
):
    assert head_dim % 2 == 0

    # Calculate inverse frequencies
    numerators = torch.arange(
        0, head_dim, 2,
        dtype=torch.float32,
        device=device
    )

    thetas = 1.0 / (theta ** (numerators / head_dim))

    # Generate absolute token positions
    positions = torch.arange(
        start_pos,
        start_pos + seq_len,
        dtype=torch.float32,
        device=device
    )

    # Calculate position-frequency products
    frequencies = torch.outer(positions, thetas)

    # Convert angles into complex rotations
    freqs_complex = torch.polar(
        torch.ones_like(frequencies),
        frequencies
    )

    return freqs_complex


def rope(x: torch.Tensor, start_pos: int = 0, theta: float = 10000.0):

    # Expected input: [B, H, T, D]
    B, H, T, D = x.shape

    assert D % 2 == 0

    x_complex = torch.view_as_complex(
        x.float().reshape(B, H, T, D // 2, 2)
    )

    freqs_complex = compute_thetas(
        seq_len=T,
        head_dim=D,
        theta=theta,
        device=x.device,
        start_pos=start_pos
    )

    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(0)

    rotated = x_complex * freqs_complex

    rotated = torch.view_as_real(rotated)

    return rotated.reshape(B, H, T, D).to(x.dtype)





    
