
import torch
from torch import nn
import warnings

def EnsureWithinRange(val, min_val, max_val):
    return max(min_val, min(val, max_val))

class AutoGroupNorm(nn.Module):
    def __init__(self, num_channels, num_groups = None, eps = 1e-5, affine=True, dtype = torch.float32, pref_channels_per_group = 8, min_groups = 1, max_groups = 64):
        super().__init__()
        num_groups = num_groups if num_groups is not None else self.__select_group_num(num_channels, pref_channels_per_group, min_groups, max_groups)
        self.norm = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=affine, dtype=dtype)
    def forward(self, x):
        return self.norm(x)        
    @staticmethod
    def __validate_groups(num_channels, num_groups):
        return (num_channels % num_groups) == 0
    @staticmethod
    def __select_group_num(num_channels, pref_channels_per_group, min_groups, max_groups):
        # Trying optimal case: num_channels divisible by pref_channels_per_group, groups within min-max range
        natural = num_channels // pref_channels_per_group
        opt = EnsureWithinRange(natural, min_groups, max_groups)
        if AutoGroupNorm.__validate_groups(num_channels, opt): return opt
        # Relax min-max constraint
        if AutoGroupNorm.__validate_groups(num_channels, natural):
            warnings.warn(f"AutoGroupNorm: pref_channels_per_group={pref_channels_per_group} divides num_channels={num_channels} cleanly, but out of range=[{min_groups}..{max_groups}]. Ignoring range restriction. Using num_groups={natural}", UserWarning)
            return natural
        # Scan all available group numbers, searching for one that works
        pos = [ i for i in range(min_groups, max_groups+1) if AutoGroupNorm.__validate_groups(num_channels, i) ]
        pos.sort( key = lambda v: abs(v - natural) )
        if len(pos): 
            warnings.warn(f"AutoGroupNorm: pref_channels_per_group={pref_channels_per_group} doesn't divide num_channels={num_channels} cleanly. Using num_groups={pos[0]}")
            return pos[0]
        # AutoGroupNorm unable to find good amount of groups
        raise RuntimeError(f"AutoGroupNorm: With num_channels={num_channels}, there is no valid num_groups within range [{min_groups}..{max_groups}]")
            
class NormActConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode="zeros", dtype = torch.float32, 
            act_factory = None, # lambda: nn.SiLU(inplace=True)
            norm_factory = None): # lambda channels: nn.BatchNorm2d(channels)
        super().__init__()
        norm = norm_factory(in_channels) if callable(norm_factory) else (norm_factory if isinstance(norm_factory, nn.Module) else AutoGroupNorm(in_channels, dtype=dtype))
        act = act_factory() if callable(act_factory) else (act_factory if isinstance(act_factory, nn.Module) else nn.SiLU(inplace=True))
        self.net = nn.Sequential(
            norm,
            act,
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, dtype=dtype)
        )
    def forward(self, x):
        return self.net(x)

class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode="zeros", dtype = torch.float32, 
            act_factory = None, # lambda: nn.SiLU(inplace=True)
            norm_factory = None): # lambda channels: nn.BatchNorm2d(channels)
        super().__init__()
        norm = norm_factory(out_channels) if callable(norm_factory) else (norm_factory if isinstance(norm_factory, nn.Module) else AutoGroupNorm(out_channels, dtype=dtype))
        act = act_factory() if callable(act_factory) else (act_factory if isinstance(act_factory, nn.Module) else nn.SiLU(inplace=True))
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, dtype=dtype),
            norm,
            act
        )
    def forward(self, x):
        return self.net(x)

class SqueezeExciteBlock(nn.Module):
    def __init__(self, channels: int, rd_ratio: float = 0.25, dtype=torch.float32):
        super().__init__()
        rd = max(1, int(round(channels * rd_ratio)))
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, rd, 1, bias=True, dtype=dtype),
            nn.SiLU(inplace=True),            # or ReLU per original SE
            nn.Conv2d(rd, channels, 1, bias=True, dtype=dtype),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.avg(x))  # B×C×1×1
        return x * w               # broadcast across H×W
        
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dtype=torch.float32, main_branch=None, act_factory=None):
        super().__init__()
        self.act = act_factory() if callable(act_factory) else (act_factory if isinstance(act_factory, nn.Module) else nn.SiLU(inplace=True))
        self.skip = nn.Identity() if (in_channels==out_channels and stride==1) else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False, dtype=dtype)
        if main_branch is None: raise RuntimeError(f"ResidualBlock requires callable factory (in_channels, out_channels, stride, dtype) passed as main branch, got {str(type(main_branch))}")
        self.main = main_branch(in_channels, out_channels, stride, dtype)
    def forward(self, x):
        residual = self.skip(x)
        x = self.main(x)
        return self.act(x + residual)
"""
# Easy residual bottleneck: 
ResidualBlock(128, 128, main_branch = lambda ic,oc,s,dt: nn.Sequential( 
    ConvNormAct(ic, int(ic*0.5), kernel_size=3, padding=1, norm_factory=AutoGroupNorm), 
    #ConvNormAct(int(ic*0.5), int(ic*0.5), kernel_size=3, padding=1, norm_factory=AutoGroupNorm), 
    ConvNormAct(int(ic*0.5), oc, kernel_size=3, padding=1, norm_factory=AutoGroupNorm, act_factory=nn.Identity))
)
"""
    