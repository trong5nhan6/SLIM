"""Custom head for idea "slim": LSCDetect — Lightweight Shared-Convolution Detect head.

Motivation (measured on this repo's own models, not guessed): with a P2 head, the stock
ultralytics Detect head is roughly *half* the parameters of a sub-1M model (star-yolo11n:
~0.3M of 0.647M total) and a large GFLOPs slice, because cv2/cv3 run a private 2-conv stack
per scale and the P2 scale pays it at 4x/16x the resolution of P3/P4. Literature (YOLOv8-PD,
Sci Reports 2024; LEAD-YOLO, 2025; RSNet, arXiv:2410.23073) converges on the same fix,
usually called LSCD:

  1. A cheap per-scale 1x1 ConvGN projects every scale to one shared hidden width `hidc`.
  2. ONE shared stack of two 3x3 ConvGN layers + ONE shared box conv + ONE shared cls conv
     replace the per-scale cv2/cv3 stacks (params divided by ~nl, FLOPs also cut because
     hidc is thinner than the stock per-scale stacks).
  3. A learnable per-scale scalar (`Scale`) on the box branch compensates for the shared
     regressor seeing different strides.
  4. GroupNorm instead of BatchNorm in the head: FCOS ablations (and the LSCD papers) show
     GN in detection heads is *more* accurate, and it is batch-size independent — relevant
     here because P2@960 forces batch=8, where BN statistics are noisy.

Reported effect in those papers: params -19..-27%, GFLOPs ~-10%, mAP flat to slightly up.

LSCDetect subclasses ultralytics Detect so everything downstream (v8DetectionLoss,
DetectionModel stride/bias init, val/predict/export paths, NWD loss patch) keeps working;
only __init__/forward/bias_init are overridden. ConvGN is deliberately a plain nn.Module
(NOT ultralytics Conv): DetectionModel.fuse() fuses anything `isinstance(m, Conv) and
hasattr(m, "bn")` via fuse_conv_and_bn, which is BatchNorm-only math — GroupNorm must not
be fused, so we stay out of fuse()'s isinstance net.
"""

import math

import torch
import torch.nn as nn

from ultralytics.nn.modules.block import C2f
from ultralytics.nn.modules.conv import Conv, RepConv
from ultralytics.nn.modules.head import Detect

__all__ = ["ConvGN", "Scale", "LSCDetect", "LSCDetectBN", "RepBottleneck", "C3k2Rep"]


class ConvGN(nn.Module):
    """Conv2d + GroupNorm + SiLU. GN group count adapts to stay a divisor of c2.

    `g` is the *convolution* group count (g=c1=c2 gives a depthwise conv); `gn_groups`
    is GroupNorm's group count (16 per the LSCD papers, clamped to a divisor of c2).
    """

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, g: int = 1, gn_groups: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, groups=g, bias=False)
        self.gn = nn.GroupNorm(math.gcd(gn_groups, c2), c2)  # GN needs num_groups | c2
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class Scale(nn.Module):
    """Learnable scalar multiplier (FCOS-style), one per detect scale on the box branch."""

    def __init__(self, value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(value, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class LSCDetect(Detect):
    """Detect head with shared GroupNorm convs across scales.

    YAML usage (registered via models.slim.register_slim()):
        - [[16, 19, 22], 1, LSCDetect, [nc, 64]]   # 64 = hidc (shared hidden width)

    parse_model appends the per-scale input channel list, so the real signature is
    (nc, hidc, ch) — same convention as Detect's (nc, ch).
    """

    def __init__(self, nc: int = 80, hidc: int = 64, ch: tuple = ()):
        super().__init__(nc, ch)  # sets nl/reg_max/no/stride + dfl; cv2/cv3 replaced below
        assert not self.end2end, "LSCDetect does not implement the end2end (v10) path"
        self.conv = nn.ModuleList(ConvGN(x, hidc, 1) for x in ch)  # per-scale align to hidc
        # Shared stack: dense 3x3 + depthwise 3x3. The papers use two dense 3x3s, but the
        # shared stack runs at EVERY scale's resolution — at P2 that dominated the whole
        # model (measured: 5.57 of 10.86 GFLOPs@640 with two dense 64ch convs). Depth stays
        # 2 (align -> dense -> dw), cost drops to ~1x dense conv + epsilon.
        self.share_conv = nn.Sequential(ConvGN(hidc, hidc, 3), ConvGN(hidc, hidc, 3, g=hidc))
        # cv2/cv3 keep Detect's attribute names (box/cls) but are single shared convs,
        # not per-scale ModuleLists.
        self.cv2 = nn.Conv2d(hidc, 4 * self.reg_max, 1)
        self.cv3 = nn.Conv2d(hidc, self.nc, 1)
        self.scale = nn.ModuleList(Scale(1.0) for _ in ch)

    def forward(self, x: list[torch.Tensor]):
        """Same output contract as Detect.forward (training list / inference tuple)."""
        for i in range(self.nl):
            feat = self.share_conv(self.conv[i](x[i]))
            x[i] = torch.cat((self.scale[i](self.cv2(feat)), self.cv3(feat)), 1)
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def bias_init(self):
        """Shared preds can't take Detect's per-stride bias init; use the stride mean.

        Detect.bias_init() sets the cls prior per scale from its stride; with one shared
        cls conv there is only one bias vector, so we initialize it for the geometric
        middle of the pyramid. The per-scale Scale modules absorb the remaining per-level
        magnitude differences during the first epochs.
        """
        s = float(self.stride.prod() ** (1.0 / self.nl))  # geometric mean stride
        self.cv2.bias.data[:] = 1.0  # box
        self.cv3.bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)  # cls prior


class LSCDetectBN(LSCDetect):
    """Fuse-able variant of LSCDetect (idea "slim", ablation B1).

    Identical shared-conv structure to LSCDetect, but every ConvGN is replaced by the stock
    ultralytics `Conv` (Conv2d + BatchNorm + SiLU). Motivation: GroupNorm cannot be folded
    into its preceding conv, so LSCDetect keeps 3 separate kernels (conv, GN, act) at EVERY
    detect scale at inference; the P2 scale pays that at 4x resolution. A BN head fuses via
    DetectionModel.fuse() (fuse_conv_and_bn), collapsing each conv+bn into a single kernel,
    which is where slim's real (non-FLOP) latency hides.

    Trade-off measured by a real run, not assumed: the shared 3x3 stack now carries ONE
    BatchNorm seen by all scales sequentially, so its running stats are averaged across P2/
    P3/P4 (this is exactly why GN was chosen — batch=8 makes per-scale BN noisy). The
    per-scale align convs + learnable Scale absorb part of that. If mAP drops materially,
    the fallback is shared conv weights with per-scale BN (fuses to nl separate convs).

    forward()/bias_init() are inherited from LSCDetect unchanged (they only reference
    self.conv / self.share_conv / self.cv2 / self.cv3 / self.scale).
    """

    def __init__(self, nc: int = 80, hidc: int = 64, ch: tuple = ()):
        Detect.__init__(self, nc, ch)  # skip LSCDetect.__init__ (it builds ConvGN)
        assert not self.end2end, "LSCDetectBN does not implement the end2end (v10) path"
        self.conv = nn.ModuleList(Conv(x, hidc, 1) for x in ch)  # per-scale align (fuse-able)
        self.share_conv = nn.Sequential(Conv(hidc, hidc, 3), Conv(hidc, hidc, 3, g=hidc))
        self.cv2 = nn.Conv2d(hidc, 4 * self.reg_max, 1)
        self.cv3 = nn.Conv2d(hidc, self.nc, 1)
        self.scale = nn.ModuleList(Scale(1.0) for _ in ch)


class RepBottleneck(nn.Module):
    """Two RepConv 3x3s with an optional residual (idea "slim", ablation B2).

    RepConv trains as a multi-branch (3x3 + 1x1) block and reparameterizes to a single dense
    3x3 conv at inference (RepConv.fuse_convs, triggered by DetectionModel.fuse()). This
    trades GhostBottleneck's low-FLOP-but-memory-bound blocks for dense convs that keep the
    GPU busy — higher FLOPs on paper, but lower wall-clock latency at batch=1, which is the
    slim latency problem. `c3k`-style depth stays at the C2f level.
    """

    def __init__(self, c: int, shortcut: bool = True):
        super().__init__()
        self.cv1 = RepConv(c, c, 3)
        self.cv2 = RepConv(c, c, 3)
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3k2Rep(C2f):
    """C3k2 with RepBottleneck inner blocks — drop-in for C3k2Ghost (same YAML signature
    C3k2(c1, c2, n, c3k, e, g, shortcut); `c3k` accepted but ignored, Rep blocks always used).
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(RepBottleneck(self.c, shortcut) for _ in range(n))
