import torch
from torch import nn

class AutoGroupNorm(nn.Module):
  def __init__(self, num_channels: int, channels_per_group: int = 8, max_groups: int = 32, affine: bool = True, eps: float = 1e-5):
    super(AutoGroupNorm, self).__init__()
    groups = num_channels // channels_per_group
    groups = max(1, min(groups, max_groups, num_channels))
    self.gn = nn.GroupNorm(groups, num_channels, affine=affine, eps = eps)
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.gn(x)