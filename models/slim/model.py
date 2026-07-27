"""Idea: slim — the "Ours" family: LSCDetect shared GroupNorm head + thinner P2 neck +
RepConv blocks, with an optional CIoU-NWD blend loss (`use_nwd`). Every cfg/exp*.yaml that
is not a stock YOLOv11 baseline is built by this class. See modules.py/register.py in this
package; this file is just the idea class train.py drives via models.build_model().
"""

from __future__ import annotations

from models.base_model import BaseModel


class SlimModel(BaseModel):
    def build(self) -> None:
        # Imported here, not at module top, so `from models.slim import SlimModel`
        # (needed just to populate MODEL_REGISTRY / --idea choices) stays cheap.
        from models.lwso.losses import patch_nwd_loss

        from .register import register_slim

        register_slim()  # must run before YOLO(...) so parse_model knows LSCDetect
        if self.cfg.get("use_nwd", False):
            ratio = float(self.cfg.get("nwd_ratio", 0.5))
            constant = float(self.cfg.get("nwd_constant", 12.8))
            patch_nwd_loss(ratio=ratio, constant=constant)
            print(f"[slim] NWD blend loss active: ratio={ratio}, C={constant}")

        self._yolo = self._build_yolo()
