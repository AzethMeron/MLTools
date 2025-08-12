import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable, List, Optional, Sequence, Union

# ---- tiny helpers ------------------------------------------------------------

def _to_list(v: Union[int, Sequence[int]], n: int) -> List[int]:
    if isinstance(v, int):
        return [v] * n
    v = list(v)
    if len(v) != n:
        raise ValueError(f"Length mismatch: expected {n}, got {len(v)}")
    return v

# Default lateral 1x1 conv (matches your original: no activation on laterals)
def default_lateral_factory(in_c: int, out_c: int) -> nn.Module:
    return PointwiseConvolutionalBlock(in_c, out_c, activation=nn.Identity())

# Default “reduce after concat” block: simple 3×3 ConvNormAct
def default_reduce_factory(in_c: int, out_c: int) -> nn.Module:
    return ConvolutionalBlock(in_c, out_c, kernel_size=3, padding=1)

# Optional smoothing after fusion (3×3)
def default_smooth_factory(ch: int) -> nn.Module:
    return ConvolutionalBlock(ch, ch, kernel_size=3, padding=1)

# ---- Generic FPN -------------------------------------------------------------

class GenericFPN(nn.Module):
    """
    A flexible Feature Pyramid Network.

    Args:
        in_channels: list of input feature widths (shallow -> deep).
        out_channels: single int or list[int]; defaults to in_channels.
        fuse_type: 'sum' | 'concat' | 'csp'
            - 'sum': elementwise sum (with per-level projection if widths differ).
            - 'concat': concat + reduce via reduce_factory.
            - 'csp': concat then CSP1_X( in=cat , work=out[i] , out=out[i] ).
        lateral_factory: fn(in_c, out_c) -> nn.Module (default: 1×1 conv, no act).
        reduce_factory: fn(in_c, out_c) -> nn.Module used when fuse_type='concat'.
        smooth: None | '3x3' to add a post-fusion smoothing conv per level.
        upsample_mode: 'nearest' or 'bilinear'.
        align_corners: bool for bilinear.
        extra_levels: how many extra pyramid levels to add by strided 3×3 on the last map.
        extra_conv_channels: channels for extra levels; int or list[int]. If None, uses the last out_channels.
    """
    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: Optional[Union[int, Sequence[int]]] = None,
        *,
        fuse_type: str = "csp",
        lateral_factory: Callable[[int, int], nn.Module] = default_lateral_factory,
        reduce_factory: Callable[[int, int], nn.Module] = default_reduce_factory,
        smooth: Optional[str] = None,
        upsample_mode: str = "bilinear",
        align_corners: Optional[bool] = False,
        extra_levels: int = 0,
        extra_conv_channels: Optional[Union[int, Sequence[int]]] = None,
    ):
        super().__init__()

        if out_channels is None:
            out_channels = list(in_channels)
        self.in_channels: List[int] = list(in_channels)
        self.out_channels: List[int] = _to_list(out_channels, len(self.in_channels))

        if len(self.in_channels) != len(self.out_channels):
            raise ValueError("in_channels and out_channels length must match.")

        self.num_levels = len(self.in_channels)
        self.fuse_type = fuse_type.lower()
        if self.fuse_type not in {"sum", "concat", "csp"}:
            raise ValueError(f"Unsupported fuse_type: {fuse_type}")

        self.upsample_mode = upsample_mode
        self.align_corners = align_corners if upsample_mode == "bilinear" else None
        self.smooth_flag = smooth

        # Laterals: project each input to desired out_channels[i]
        self.lateral_convs = nn.ModuleList([
            lateral_factory(self.in_channels[i], self.out_channels[i])
            for i in range(self.num_levels)
        ])

        # Top-down fusion modules for levels 0..L-2 (each fuses from level i+1 into level i)
        self.fuse_modules = nn.ModuleList()
        self.proj_from_top = nn.ModuleList()  # used by 'sum' when channel dims differ
        self.post_smooth = nn.ModuleList()

        for i in range(self.num_levels - 1):
            ch_i = self.out_channels[i]
            ch_top = self.out_channels[i + 1]

            if self.fuse_type == "sum":
                # If channels differ, project top to ch_i before summation.
                self.proj_from_top.append(
                    nn.Identity() if ch_top == ch_i else PointwiseConvolutionalBlock(ch_top, ch_i, activation=nn.Identity())
                )
                self.fuse_modules.append(nn.Identity())  # fusion is just + ; optional smoothing below
            elif self.fuse_type == "concat":
                self.proj_from_top.append(nn.Identity())
                self.fuse_modules.append(reduce_factory(ch_i + ch_top, ch_i))
            else:  # 'csp'
                self.proj_from_top.append(nn.Identity())
                self.fuse_modules.append(
                    CSP1_X(input_channels=ch_i + ch_top, working_channels=ch_i, output_channels=ch_i, X=2)
                )

            if self.smooth_flag == "3x3":
                self.post_smooth.append(default_smooth_factory(ch_i))
            else:
                self.post_smooth.append(nn.Identity())

        # Extra pyramid levels (e.g., P6/P7)
        self.extra_levels = int(extra_levels)
        if self.extra_levels > 0:
            if extra_conv_channels is None:
                extra_ch = [self.out_channels[-1]] * self.extra_levels
            else:
                extra_ch = _to_list(extra_conv_channels, self.extra_levels)

            self.extra_downsamples = nn.ModuleList()
            self.extra_out_channels = extra_ch
            in_ch = self.out_channels[-1]
            for j in range(self.extra_levels):
                out_ch = extra_ch[j]
                # 3x3 stride-2 conv to go down one level
                self.extra_downsamples.append(
                    ConvolutionalBlock(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
                )
                in_ch = out_ch
        else:
            self.extra_downsamples = nn.ModuleList()
            self.extra_out_channels = []

    def _upsample_to(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode=self.upsample_mode, align_corners=self.align_corners)

    def forward(self, features: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if len(features) != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} feature maps, got {len(features)}.")

        # 1) Laterals
        outs = [l(f) for l, f in zip(self.lateral_convs, features)]
        # 2) Top-down
        for i in range(self.num_levels - 1, 0, -1):
            top = self._upsample_to(outs[i], outs[i - 1])
            top = self.proj_from_top[i - 1](top)

            if self.fuse_type == "sum":
                fused = outs[i - 1] + top
            else:
                fused = self.fuse_modules[i - 1](torch.cat([outs[i - 1], top], dim=1))

            outs[i - 1] = self.post_smooth[i - 1](fused)

        # 3) Optional extra levels
        if self.extra_levels > 0:
            x = outs[-1]
            for ds in self.extra_downsamples:
                x = ds(x)
                outs.append(x)

        return outs

from typing import List, Sequence, Optional, Union, Callable
from torch import nn

# If you kept the helpers from the GenericFPN snippet:
# default_lateral_factory, default_reduce_factory

class ClassicFPN(GenericFPN):
    """
    Classic FPN (Lin et al. 2017):
      - Laterals: 1x1 conv to a unified width (default 256)
      - Fusion: elementwise sum (projects top path if widths differ)
      - Smoothing: optional 3x3 after fusion (enabled by default)
      - Upsample: nearest by default
      - Extra levels: optional (P6/P7 via stride-2 3x3 convs)
    """
    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: Union[int, Sequence[int]] = 256,
        *,
        smooth: bool = True,
        upsample_mode: str = "nearest",
        align_corners: Optional[bool] = None,
        extra_levels: int = 0,
        extra_conv_channels: Optional[Union[int, Sequence[int]]] = None,
        lateral_factory: Callable[[int, int], nn.Module] = default_lateral_factory,
        reduce_factory: Callable[[int, int], nn.Module] = default_reduce_factory,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            fuse_type="sum",
            lateral_factory=lateral_factory,
            reduce_factory=reduce_factory,
            smooth="3x3" if smooth else None,
            upsample_mode=upsample_mode,
            align_corners=align_corners,
            extra_levels=extra_levels,
            extra_conv_channels=extra_conv_channels,
        )


class CspFPN(GenericFPN):
    """
    CSP-style FPN (matches your current design):
      - Laterals: 1x1 conv (no activation)
      - Fusion: concat + CSP1_X(working=out[i], out=out[i], X=2)
      - Smoothing: off by default
      - Upsample: bilinear by default (align_corners=False)
      - Channels: keep per-level widths by default (out_channels=None)
    """
    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: Optional[Union[int, Sequence[int]]] = None,
        *,
        upsample_mode: str = "bilinear",
        align_corners: Optional[bool] = False,
        smooth: bool = False,
        extra_levels: int = 0,
        extra_conv_channels: Optional[Union[int, Sequence[int]]] = None,
        lateral_factory: Callable[[int, int], nn.Module] = default_lateral_factory,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,           # defaults to in_channels widths
            fuse_type="csp",
            lateral_factory=lateral_factory,
            reduce_factory=default_reduce_factory,  # unused for 'csp' but kept for API symmetry
            smooth="3x3" if smooth else None,
            upsample_mode=upsample_mode,
            align_corners=align_corners,
            extra_levels=extra_levels,
            extra_conv_channels=extra_conv_channels,
        )
