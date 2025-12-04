
import torch
from torch import nn
import warnings

# Best inputs for factories
# act_factory = lambda: nn.SiLU(inplace=True)
# norm_factory = lambda channels: nn.BatchNorm2d(channels)

# norm defaults to AutoGroupNorm
# act defaults to SiLU inplace

def _make_norm(norm_factory, channels):
    if norm_factory is None: return AutoGroupNorm(channels)
    if callable(norm_factory) and not isinstance(norm_factory, nn.Module): return norm_factory(channels)
    raise TypeError(f"norm_factory must be a function like 'lambda channels: nn.BatchNorm2d(channels)', got type={str(type(norm_factory))}")

def _make_act(act_factory):
    if act_factory is None: return nn.SiLU(inplace=True)
    if callable(act_factory) and not isinstance(act_factory, nn.Module): return act_factory()
    raise TypeError(f"act_factory must be a function like 'lambda: nn.SiLU(inplace=True)', got type={str(type(act_factory))}")

def _ensure_within_range(val, min_val, max_val):
    return max(min_val, min(val, max_val))

class AutoGroupNorm(nn.Module):
    def __init__(self, num_channels: int, num_groups: int | None = None, eps:float = 1e-5, affine:bool=True, dtype: torch.dtype = torch.float32, pref_channels_per_group:int = 8, min_groups:int = 1, max_groups:int = 64):
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
        opt = _ensure_within_range(natural, min_groups, max_groups)
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
           
class PureConv(nn.Module): # 
    def __init__(self, in_channels: int, out_channels:int, kernel_size, stride:int=1, padding:int=0, dilation:int=1, groups:int=1, bias:bool=False, padding_mode="zeros", dtype: torch.dtype = torch.float32, 
            act_factory = None, # ignored
            norm_factory = None): # ignored
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, dtype=dtype)
    def forward(self, x):
        return self.conv(x)
           
class NormActConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride:int=1, padding:int=0, dilation:int=1, groups:int=1, bias:bool=False, padding_mode="zeros", dtype: torch.dtype = torch.float32, 
            act_factory = None, # lambda: nn.SiLU(inplace=True)
            norm_factory = None): # lambda channels: nn.BatchNorm2d(channels)
        super().__init__()
        norm = _make_norm(norm_factory, in_channels)
        act = _make_act(act_factory)
        self.net = nn.Sequential(
            norm,
            act,
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, dtype=dtype)
        )
    def forward(self, x):
        return self.net(x)

class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride:int=1, padding:int=0, dilation:int=1, groups:int=1, bias:bool=False, padding_mode="zeros", dtype: torch.dtype = torch.float32, 
            act_factory = None, # lambda: nn.SiLU(inplace=True)
            norm_factory = None): # lambda channels: nn.BatchNorm2d(channels)
        super().__init__()
        norm = _make_norm(norm_factory, out_channels)
        act = _make_act(act_factory)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, dtype=dtype),
            norm,
            act
        )
    def forward(self, x):
        return self.net(x)

class SqueezeExciteBlock(nn.Module):
    def __init__(self, channels: int, rd_ratio: float = 0.25, dtype=torch.float32, act_factory = None):
        super().__init__()
        rd = max(1, int(round(channels * rd_ratio)))
        act = _make_act(act_factory)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, rd, 1, bias=True, dtype=dtype),
            act,            
            nn.Conv2d(rd, channels, 1, bias=True, dtype=dtype),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.avg(x))  # B×C×1×1
        return x * w               # broadcast across H×W

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    - Channel attention (avg + max pool) with a tiny MLP (two 1x1 convs)
    - Spatial attention (avg + max across channels) with a 7x7 conv
    Factories:
      - act_factory: activation used inside the channel MLP (like your SE block)
    Notes:
      - No norm layers used inside CBAM (typical CBAM design).
      - Sigmoid gates are standard for attention.
    """
    def __init__(
        self,
        channels: int,
        rd_ratio: float = 0.25,            # reduction ratio for channel MLP
        spatial_kernel: int = 7,           # 7x7 in the paper
        dtype: torch.dtype = torch.float32,
        act_factory=None,                  # lambda: nn.SiLU(inplace=True) by default through _make_act
        use_channel: bool = True,
        use_spatial: bool = True,
    ):
        super().__init__()
        self.use_channel = use_channel
        self.use_spatial = use_spatial

        # ---- Channel attention (like SE, but with both avg & max pooling) ----
        rd = max(1, int(round(channels * rd_ratio)))
        act = _make_act(act_factory)
        self.ca_mlp = nn.Sequential(
            nn.Conv2d(channels, rd, kernel_size=1, bias=True, dtype=dtype),
            act,
            nn.Conv2d(rd, channels, kernel_size=1, bias=True, dtype=dtype),
        )
        self.ca_gate = nn.Sigmoid()

        # ---- Spatial attention (avg+max over channels -> 7x7 conv) ----------
        padding = spatial_kernel // 2
        self.sa_conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=padding, bias=False, dtype=dtype)
        self.sa_gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x

        # Channel attention ----------------------------------------------------
        if self.use_channel:
            # global avg & max pool over spatial dims -> [B, C, 1, 1]
            avg = torch.mean(out, dim=(2, 3), keepdim=True)
            mx  = torch.amax(out, dim=(2, 3), keepdim=True)
            # shared MLP on both, then sum and gate
            att_c = self.ca_gate(self.ca_mlp(avg) + self.ca_mlp(mx))
            out = att_c * out

        # Spatial attention ----------------------------------------------------
        if self.use_spatial:
            # avg & max over channels -> [B, 1, H, W], then concat -> [B, 2, H, W]
            avg = torch.mean(out, dim=1, keepdim=True)
            mx, _ = torch.max(out, dim=1, keepdim=True)
            att_s = self.sa_gate(self.sa_conv(torch.cat([avg, mx], dim=1)))
            out = att_s * out

        return out

 
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride:int=1, dtype=torch.float32, main_branch=None, act_factory=None, norm_factory=None, kernel_size=3):
        super().__init__()
        self.act = _make_act(act_factory)
        self.skip = nn.Identity() if (in_channels==out_channels and stride==1) else nn.Sequential(
                nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding) if stride > 1 else nn.Identity(),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False, dtype=dtype) if in_channels != out_channels else nn.Identity(), 
                norm_factory(out_channels) if in_channels != out_channels else nn.Identity() )
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

class Bottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int | None = None, reduction:float=0.5, 
            X:int = 1, stride:int = 1, padding_mode="zeros", dtype: torch.dtype = torch.float32, norm_factory = None, act_factory = None, conv_class=ConvNormAct):
        super().__init__()
        out_channels = out_channels if out_channels else in_channels
        working_channels = int(out_channels*reduction)
        
        # Reduction
        reduction_layer = conv_class(in_channels, working_channels, kernel_size=1, norm_factory=norm_factory, act_factory=act_factory, dtype=dtype)
        
        # Processing layers
        bottleneck_layers = []
        if stride > 1 and X <= 0:
            warnings.warn(f"Bottleneck: You've set stride={stride} but X={X}, meaning no processing layers will be created. Stride is applied on first processing layer, meaning it is silently ignored there.", UserWarning)
        elif X<=0:
            warnings.warn(f"Bottleneck: Created with X={X}, meaning no processing layers will be created", UserWarning)
        elif stride > 1 and X >= 1:
            bottleneck_layers.append(conv_class(working_channels, working_channels, kernel_size=3, padding=1, padding_mode=padding_mode, stride=stride, norm_factory=norm_factory, act_factory=act_factory, dtype=dtype))            
            X = X - 1
        bottleneck_layers.extend([ conv_class(working_channels, working_channels, kernel_size=3, padding=1, padding_mode=padding_mode, norm_factory=norm_factory, act_factory=act_factory, dtype=dtype) for _ in range(X) ])
        
        # Expansion
        expansion_layer = conv_class(working_channels, out_channels, kernel_size=1, norm_factory=norm_factory, act_factory=lambda: nn.Identity(), dtype=dtype)
        if conv_class is NormActConv: expansion_layer = nn.Conv2d(working_channels, out_channels, kernel_size=1, dtype=dtype, bias=False)
        
        # And finally, create residual block
        self.net = ResidualBlock(in_channels, out_channels, stride=stride, dtype = dtype, act_factory=act_factory, 
            main_branch = lambda ic, oc, s, dt: nn.Sequential(
                reduction_layer,
                *bottleneck_layers,
                expansion_layer,
            ))
        
    def forward(self, x):
        return self.net(x)

class PointwiseConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride:int=1, dtype=torch.float32, bias:bool=False, norm_factory=None, act_factory=None, conv_class=ConvNormAct):
        super().__init__()
        self.conv = conv_class(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias, norm_factory=norm_factory, act_factory=act_factory, dtype=dtype)
    def forward(self, x):
        return self.conv(x)

class DepthwiseConv(nn.Module):
    def __init__(self, channels: int, kernel_size=3, padding:int=1, padding_mode = "zeros", X:int=1, stride:int=1, dtype=torch.float32, bias:bool=False, norm_factory=None, act_factory=None, conv_class=ConvNormAct):
        super().__init__()
        convs = []
        for i in range(X):
            s = stride if i == 0 else 1
            convs.append(conv_class(channels, channels, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, stride=s, bias=bias, norm_factory=norm_factory, act_factory=act_factory, groups = channels, dtype=dtype))
        self.conv = nn.Sequential(
            *convs
        )
    def forward(self, x):
        return self.conv(x)

class CSP1_X(nn.Module):
    def __init__(self, in_channels: int, working_channels: int, out_channels: int | None = None, padding_mode="zeros",
            X: int = 2, bottleneck_X: int = 1, reduction: float = 0.5, norm_factory=None, act_factory=None, dtype=torch.float32, conv_class=ConvNormAct):
        super().__init__()
        # Channels
        out_channels = out_channels if out_channels else in_channels
        concat_channels =  working_channels*2
        # Main branch
        main = [ ConvNormAct(in_channels, working_channels, kernel_size=1, norm_factory=norm_factory, act_factory=act_factory, dtype=dtype) ]
        main.extend( [Bottleneck( working_channels, working_channels, reduction=reduction, X = bottleneck_X, stride = 1, padding_mode=padding_mode, dtype=dtype, norm_factory=norm_factory, act_factory=act_factory, conv_class=conv_class ) for _ in range(X) ])
        main.append( PureConv(working_channels, working_channels, kernel_size=1, dtype=dtype, bias=False) )
        self.main_branch = nn.Sequential(*main)
        # Skip branch
        self.skip_branch = PureConv(in_channels, working_channels, kernel_size=1, dtype=dtype, bias=False)
        # Post-concat branch
        norm = _make_norm(norm_factory, concat_channels)
        act = _make_act(act_factory)
        self.post_concat_branch = nn.Sequential(
            norm,
            act,
            ConvNormAct(concat_channels, out_channels, kernel_size=1, norm_factory=norm_factory, act_factory=act_factory, dtype=dtype)
        )
    def forward(self, x):
        residual = self.skip_branch(x)
        x = self.main_branch(x)
        x = torch.cat([x, residual], dim=1)
        return self.post_concat_branch(x)
        
class CSP2_X(nn.Module):
    def __init__(self, in_channels: int, working_channels: int, out_channels: int | None = None, padding_mode="zeros",
            X: int = 2, bottleneck_X: int = 1, reduction: float = 0.5, norm_factory=None, act_factory=None, dtype=torch.float32, conv_class=ConvNormAct):
        super().__init__()
        # Channels and Residuals
        out_channels = out_channels if out_channels else in_channels
        concat_channels =  working_channels*2
        R = 2*X
        # Main branch
        main = [ ConvNormAct(in_channels, working_channels, kernel_size=1, norm_factory=norm_factory, act_factory=act_factory, dtype=dtype),
                 ConvNormAct(working_channels, working_channels, kernel_size=3, padding=1, padding_mode=padding_mode, norm_factory=norm_factory, act_factory=act_factory, dtype=dtype) ]
        main.extend( [Bottleneck( working_channels, working_channels, reduction=reduction, X = bottleneck_X, stride = 1, dtype=dtype, norm_factory=norm_factory, act_factory=act_factory, conv_class=conv_class ) for _ in range(R) ])
        main.append( PureConv(working_channels, working_channels, kernel_size=1, dtype=dtype, bias=False) )
        self.main_branch = nn.Sequential(*main)
        # Skip branch
        self.skip_branch = PureConv(in_channels, working_channels, kernel_size=1, dtype=dtype, bias=False)
        # Post-concat branch
        norm = _make_norm(norm_factory, concat_channels)
        act = _make_act(act_factory)
        self.post_concat_branch = nn.Sequential(
            norm,
            act,
            ConvNormAct(concat_channels, out_channels, kernel_size=1, norm_factory=norm_factory, act_factory=act_factory, dtype=dtype)
        )
    def forward(self, x):
        residual = self.skip_branch(x)
        x = self.main_branch(x)
        x = torch.cat([x, residual], dim=1)
        return self.post_concat_branch(x)

