import torch 
import torch.nn as nn


class BoxNormalizer(nn.Module):
    """Maps a known coordinate box [lo, hi] <-> [-1, 1]."""
    def __init__(self, lo: float = 0.0, hi: float = 512.0) -> None:
        super().__init__()
        scale = 2.0 / (hi - lo)
        self.register_buffer("scale", torch.tensor(scale))
        self.register_buffer("offset", torch.tensor(-1.0 - lo * scale))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.offset

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.offset) / self.scale