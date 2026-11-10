"""Shape/param sanity tests for the shared blocks and the cfg/exp*.yaml models.

Run: pytest tests/ -v
"""

from pathlib import Path

import pytest
import torch

from models.lwso.losses import nwd
from models.lwso.modules import BiFPNCat, C3k2Ghost, DySample, ECA, EMA, SPDConv, SPDConvGroup
from models.slim.modules import ConvGN, LSCDetect, Scale

ROOT = Path(__file__).resolve().parents[1]


def n_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------- modules

def test_spdconv_halves_spatial_no_pixel_loss():
    m = SPDConv(16, 32)
    y = m(torch.randn(2, 16, 64, 64))
    assert y.shape == (2, 32, 32, 32)


def test_spdconv_odd_input_pads_to_even():
    # 240->120 etc. are even in our pipeline, but degrade gracefully via padding
    y = SPDConv(8, 16)(torch.randn(1, 8, 63, 63))
    assert y.shape[-2:] == (32, 32)  # 63 padded to 64, then halved


def test_spdconvgroup_halves_spatial_no_pixel_loss():
    m = SPDConvGroup(16, 32)
    y = m(torch.randn(2, 16, 64, 64))
    assert y.shape == (2, 32, 32, 32)


def test_spdconvgroup_lighter_than_dense_spdconv():
    dense = SPDConv(16, 32)
    grouped = SPDConvGroup(16, 32, groups=8)
    assert n_params(grouped) < n_params(dense), (
        f"grouped {n_params(grouped)} should be < dense {n_params(dense)}"
    )


def test_spdconvgroup_falls_back_when_not_divisible():
    # c1*4=12, c2=7 (prime, coprime with 12): no group >1 divides both -> falls back to g=1
    m = SPDConvGroup(3, 7, groups=8)
    y = m(torch.randn(1, 3, 8, 8))
    assert y.shape == (1, 7, 4, 4)


def test_c3k2ghost_shape_and_lighter_than_c3k2():
    from ultralytics.nn.modules.block import C3k2

    ghost = C3k2Ghost(64, 128, n=2)
    stock = C3k2(64, 128, n=2)
    y = ghost(torch.randn(2, 64, 32, 32))
    assert y.shape == (2, 128, 32, 32)
    assert n_params(ghost) < n_params(stock), (
        f"ghost {n_params(ghost)} should be < stock {n_params(stock)}"
    )


def test_ema_preserves_shape():
    x = torch.randn(2, 128, 30, 30)
    assert EMA(128, 128)(x).shape == x.shape


def test_ema_rejects_channel_change():
    with pytest.raises(AssertionError):
        EMA(128, 64)


def test_eca_preserves_shape():
    x = torch.randn(2, 64, 30, 30)
    assert ECA(64)(x).shape == x.shape


def test_eca_rejects_channel_change():
    with pytest.raises(AssertionError):
        ECA(64, 32)


def test_eca_kernel_size_is_odd_and_positive():
    for c in (16, 32, 48, 64, 128, 256):
        k = ECA(c).conv.kernel_size[0]
        assert k >= 1 and k % 2 == 1, f"c={c}: kernel {k} should be odd"


def test_dysample_upscales_2x():
    m = DySample(64, 64, scale=2)
    y = m(torch.randn(2, 64, 32, 32))
    assert y.shape == (2, 64, 64, 64)


def test_dysample_fresh_init_close_to_nearest_upsample():
    # offset conv init ~0 => output should start near plain upsampling (stable start)
    torch.manual_seed(0)
    x = torch.randn(1, 8, 16, 16)
    y = DySample(8, 8, scale=2)(x)
    ref = torch.nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
    assert (y - ref).abs().mean() < 0.2


def test_bifpncat_sums_channels_and_weights_normalized():
    m = BiFPNCat(2)
    a, b = torch.randn(2, 64, 16, 16), torch.randn(2, 128, 16, 16)
    y = m([a, b])
    assert y.shape == (2, 192, 16, 16)
    w = torch.nn.functional.relu(m.w)
    assert abs((w / (w.sum() + m.eps)).sum().item() - 1.0) < 1e-3


# ---------------------------------------------------------------- loss

def test_nwd_identical_boxes_is_one():
    boxes = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
    assert nwd(boxes, boxes).item() == pytest.approx(1.0, abs=1e-3)


def test_nwd_smooth_for_small_shift_where_iou_collapses():
    a = torch.tensor([[100.0, 100.0, 108.0, 108.0]])  # 8x8 box
    b = a + 6.0  # 6px shift: IoU ~= 0.02, NWD should stay clearly positive
    from ultralytics.utils.metrics import bbox_iou

    iou = bbox_iou(a, b, xywh=False).item()
    sim = nwd(a, b).item()
    assert iou < 0.1
    assert sim > 0.4


# ---------------------------------------------------------------- slim (LSCDetect head)

def test_convgn_shape_and_uses_groupnorm_not_batchnorm():
    m = ConvGN(32, 64, 3)
    y = m(torch.randn(2, 32, 16, 16))
    assert y.shape == (2, 64, 16, 16)
    assert isinstance(m.gn, torch.nn.GroupNorm)
    assert not any(isinstance(x, torch.nn.BatchNorm2d) for x in m.modules())


def test_convgn_gn_groups_adapt_to_odd_channels():
    # 40 channels: gcd(16, 40) = 8 groups; must build and forward without error
    y = ConvGN(16, 40, 3)(torch.randn(1, 16, 8, 8))
    assert y.shape == (1, 40, 8, 8)


def test_convgn_depthwise_variant():
    m = ConvGN(48, 48, 3, g=48)
    assert m.conv.groups == 48
    assert n_params(m) < n_params(ConvGN(48, 48, 3))


def test_scale_multiplies_and_is_learnable():
    s = Scale(2.0)
    assert torch.allclose(s(torch.ones(3)), torch.full((3,), 2.0))
    assert sum(1 for _ in s.parameters()) == 1


def test_lscdetect_train_and_eval_forward():
    ch = (48, 96, 128)
    m = LSCDetect(nc=10, hidc=48, ch=ch)
    m.stride = torch.tensor([4.0, 8.0, 16.0])
    m.bias_init()  # must not need per-scale cv2/cv3 ModuleLists like stock Detect does
    feats = [torch.randn(2, c, 64 // (2**i), 64 // (2**i)) for i, c in enumerate(ch)]
    m.train()
    outs = m([f.clone() for f in feats])
    assert [o.shape[1] for o in outs] == [m.no] * 3  # 4*reg_max + nc per scale
    m.eval()
    with torch.no_grad():
        y, raw = m([f.clone() for f in feats])
    assert y.shape[1] == 4 + 10  # decoded boxes + class scores


def test_lscdetect_much_lighter_than_stock_detect():
    from ultralytics.nn.modules.head import Detect

    ch = (48, 96, 128)
    shared = LSCDetect(nc=10, hidc=48, ch=ch)
    stock = Detect(nc=10, ch=ch)
    assert n_params(shared) < 0.5 * n_params(stock), (
        f"LSCDetect {n_params(shared)} should be <50% of stock Detect {n_params(stock)}"
    )


def test_slim_yaml_stays_under_baseline_budget():
    # The whole point of idea slim: params < 2.58M AND GFLOPs@640 < 6.5 (stock YOLO11n,
    # unfused) while keeping the P2 head. Regression-guard both.
    from models.slim.register import register_slim

    register_slim()
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    det = YOLO(str(ROOT / "cfg/exp01-slim-yolo11n.yaml")).model
    assert n_params(det) < 2_580_000
    gflops = get_flops(det, 640)
    assert 0 < gflops < 6.5, f"slim GFLOPs@640 = {gflops:.2f}, must stay below baseline 6.5"


def test_slim_fuse_preserves_groupnorm_head():
    # DetectionModel.fuse() folds Conv+BatchNorm pairs; ConvGN must never be caught by
    # that isinstance net (GroupNorm can't be folded into a conv the same way).
    from models.slim.register import register_slim

    register_slim()
    from ultralytics import YOLO

    det = YOLO(str(ROOT / "cfg/exp01-slim-yolo11n.yaml")).model
    n_gn_before = sum(1 for m in det.modules() if isinstance(m, torch.nn.GroupNorm))
    det.fuse()
    n_gn_after = sum(1 for m in det.modules() if isinstance(m, torch.nn.GroupNorm))
    assert n_gn_before == n_gn_after > 0
    with torch.no_grad():
        det.eval()(torch.zeros(1, 3, 256, 256))


# ---------------------------------------------------------------- full models

# One row per experiment in results/SOTA-SLIM.xlsx. `params_m` is the UNFUSED parameter
# count measured when the file was written; the tolerance is tight (2%) on purpose, so an
# accidental edit to a cfg shows up here instead of silently invalidating a table row.
# Rows 5 and 11 of the spreadsheet reuse EXP04's model, and row 14 reuses EXP04's cfg with
# only `use_nwd` flipped, so neither gets its own entry.
@pytest.mark.parametrize(
    "cfg,register_fn,expected_strides,params_m",
    [
        ("cfg/exp01-slim-yolo11n.yaml", "slim", [4, 8, 16], 0.560),
        ("cfg/exp02-b2a-yolo11n.yaml", "slim", [4, 8, 16], 0.945),
        ("cfg/exp03-b2b-yolo11n.yaml", "slim", [4, 8, 16], 0.553),
        ("cfg/exp04-ours-yolo11n.yaml", "slim", [4, 8, 16], 0.938),
        ("cfg/exp06-ours-yolo11s.yaml", "slim", [4, 8, 16], 2.373),
        ("cfg/exp07-ours-yolo11m.yaml", "slim", [4, 8, 16], 4.208),
        ("cfg/exp08-base-yolo11n.yaml", None, [8, 16, 32], 2.592),
        ("cfg/exp09-base-yolo11s.yaml", None, [8, 16, 32], 9.432),
        ("cfg/exp10-base-yolo11m.yaml", None, [8, 16, 32], 20.061),
        ("cfg/exp12-nohead-yolo11n.yaml", "slim", [4, 8, 16], 1.362),
        ("cfg/exp13-nop2-yolo11n.yaml", "slim", [8, 16], 0.812),  # P2 removed -> 2 scales
        ("cfg/exp15-noattn-yolo11n.yaml", "slim", [4, 8, 16], 0.872),
        ("cfg/exp16-nospd-yolo11n.yaml", "slim", [4, 8, 16], 1.108),
        ("cfg/exp17-bnhead-yolo11n.yaml", "slim", [4, 8, 16], 0.938),
        # P2 removed AND a P5/32 stage added -> the conventional YOLO pyramid
        ("cfg/exp18-p345-yolo11n.yaml", "slim", [8, 16, 32], 3.001),
        # EXP04 + a P5/32 stage -> four-level P2-P5 pyramid (Reviewer 1 control)
        ("cfg/exp19-p2345-yolo11n.yaml", "slim", [4, 8, 16, 32], 3.127),
    ],
)
def test_model_builds_and_forwards(cfg, register_fn, expected_strides, params_m):
    if register_fn == "slim":
        from models.slim.register import register_slim

        register_slim()  # calls register_lwso() for the shared blocks
    # register_fn is None for cfg files that only use stock ultralytics modules

    from ultralytics import YOLO

    model = YOLO(str(ROOT / cfg))
    det = model.model  # DetectionModel
    total = n_params(det)
    assert total / 1e6 == pytest.approx(params_m, rel=0.02), (
        f"{cfg}: {total/1e6:.3f}M params, expected {params_m:.3f}M -- did the cfg change?"
    )

    det.eval()
    with torch.no_grad():
        preds = det(torch.zeros(1, 3, 256, 256))
    assert preds is not None
    strides = sorted(int(s) for s in det.stride.tolist())
    assert strides == expected_strides, f"{cfg}: unexpected strides {strides}"
    print(f"{cfg}: {total/1e6:.3f}M params, strides {strides}")
