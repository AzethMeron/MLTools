import torch
import torch.nn.functional as F
from torch import nn

class FastConv2d(nn.Conv2d):
    """
    Identical conv output to nn.Conv2d, plus returns shared receptive-field features.
    Forward returns:
      out: [B, C_out, H_out, W_out]
      shared_feats: [B, H_out, W_out, C_in_per_group*kH*kW]  (detached, CPU)
    """

    def forward(self, x: torch.Tensor):
        out = F.conv2d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )

        # Unfold to collect receptive fields for all (y,x)
        kH, kW = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
        dH, dW = self.dilation if isinstance(self.dilation, tuple) else (self.dilation, self.dilation)
        pH, pW = self.padding if isinstance(self.padding, tuple) else (self.padding, self.padding)
        sH, sW = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)

        patches = F.unfold(
            x, kernel_size=(kH, kW), dilation=(dH, dW),
            padding=(pH, pW), stride=(sH, sW)   # [B, C_in*kH*kW, L]
        )

        B, Ckk, L = patches.shape
        H_out, W_out = out.shape[2], out.shape[3]

        shared_feats = (patches.transpose(1, 2)         # [B, L, Ckk]
                                .contiguous()
                                .view(B, H_out, W_out, Ckk)
                                .detach()
                                .cpu())
        return out, shared_feats

    @torch.no_grad()
    def class_from_shared(
        self,
        shared_feats: torch.Tensor,          # [B, H, W, Ckk]  (detached, CPU)
        class_id: int,
        return_contributions: bool = False,  # False -> scores [B,H,W]; True -> per-element contribs [B,H,W,Ckk_group]
        add_bias: bool = True
    ):
        """
        Compute either logits (dot + bias) or per-element contributions for a given class_id
        from CPU 'shared_feats'. Handles grouped convs by slicing the correct input slice.
        """
        assert shared_feats.device.type == "cpu", "shared_feats must be on CPU (as produced by forward)."
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

        # Get that class's weight vector for the same group slice
        w = (self.weight[class_id]                # [Cg_in, kH, kW]
             .detach().cpu().reshape(-1))        # [Cg_vec]

        if return_contributions:
            # Elementwise contributions per position (no bias):
            # [B,H,W,Cg_vec] where sum over last dim equals the pre-bias logit
            return feats_g * w                   # stays on CPU

        # Compute logits efficiently: flatten spatial, matmul, reshape
        FH = feats_g.reshape(B * H * W, Cg_vec)   # [BHW, Cg_vec]
        scores = FH @ w                            # [BHW]
        scores = scores.view(B, H, W)              # [B,H,W]

        if add_bias and (self.bias is not None):
            scores += float(self.bias[class_id].detach().cpu())

        return scores
