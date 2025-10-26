import torch
import torch.nn.functional as F
from torch import nn

class Conv2dWithFeatures(nn.Conv2d):
    """
    Identical conv output to nn.Conv2d, plus returns shared receptive-field features.
    Forward returns out or (out, shared_feats) if include_features=True:
      out: [B, C_out, H_out, W_out]
      shared_feats: [B, H_out, W_out, C_in*kH*kW]  (detached, CPU)
    """

    def forward(self, x: torch.Tensor, include_features: bool = False):
        # Keep gradients for the real convolution
        out = F.conv2d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )
        
        if not include_features: return out

        # Everything below runs without gradient tracking
        with torch.no_grad():
            # These are already tuples in nn.Conv2d, but normalize anyway for safety
            kH, kW = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
            dH, dW = self.dilation if isinstance(self.dilation, tuple) else (self.dilation, self.dilation)
            pH, pW = self.padding if isinstance(self.padding, tuple) else (self.padding, self.padding)
            sH, sW = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)

            patches = F.unfold(
                x, kernel_size=(kH, kW), dilation=(dH, dW),
                padding=(pH, pW), stride=(sH, sW)
            )  # [B, C_in*kH*kW, L]

            B, Ckk, L = patches.shape
            H_out, W_out = out.shape[-2:]
            assert L == H_out * W_out, "unfold length must match conv spatial size"

            shared_feats = (
                patches.detach().transpose(1, 2)     # [B, L, C_in*kH*kW]
                .reshape(B, H_out, W_out, Ckk)       # [B, H_out, W_out, C_in*kH*kW]
                .cpu()
            )

        return out, shared_feats

    @torch.no_grad()
    def features_for_class_from_shared(
        self,
        shared_feats: torch.Tensor,          # [B, H, W, C_in*kH*kW] (detached, CPU)
        class_id: int
    ):
        """
        Get features used to compute score for class with class_id, extracting
        from 'shared_feats'. Handles grouped convs by slicing the correct input slice.
        """
        assert shared_feats.ndim == 4, "shared_feats must have shape [B, H, W, Ckk]."

        B, H, W, Ckk = shared_feats.shape
        kH, kW = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
        G = self.groups
        C_in = self.in_channels
        Co = self.out_channels
        assert 0 <= class_id < Co, "class_id out of range."

        Cg_in = C_in // G
        Cg_vec = Cg_in * kH * kW
        Co_per_g = Co // G

        # Identify group of this output channel and slice the matching part of shared_feats
        g = class_id // Co_per_g
        start = g * Cg_vec
        end   = (g + 1) * Cg_vec
        assert end <= Ckk, "shared_feats last dim does not match expected grouped feature size."

        feats_g = shared_feats[..., start:end]     # [B, H, W, Cg_vec]

        return feats_g
