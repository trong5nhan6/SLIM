"""Shared building blocks for the Ours (slim) family — NOT an idea of its own.

This package used to back `--idea lwso`. That idea is gone; what stayed is the block
library everything else is built on, so the package remains as a plain dependency of
models/slim/:

    modules.py   - SPDConv(Group), C3k2Ghost, EMA, ECA, DySample, BiFPNCat (nn.Module blocks)
    register.py  - register_lwso(): runtime registration into ultralytics parse_model,
                   called by models.slim.register.register_slim()
    losses.py    - NWDBboxLoss / patch_nwd_loss(): CIoU-NWD blend loss, used by every
                   config with `use_nwd: true`

Nothing is re-exported here on purpose (import-light: no torch/ultralytics at package
init) so `from models import MODEL_REGISTRY` stays cheap for train.py's argparse setup.
Import what you need directly, e.g. `from models.lwso.modules import SPDConvGroup`.
"""

__all__: list[str] = []
