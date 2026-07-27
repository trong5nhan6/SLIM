"""Idea: baseline — stock YOLO11n, no custom modules, no NWD loss.

Reference point cho họ Ours. Dùng cho 3 dòng YOLOv11 gốc của Bảng 3:
cfg/exp08-base-yolo11n.yaml, exp09 (s), exp10 (m) — đều train from-scratch cùng
recipe với họ Ours để so sánh công bằng (xem configs/exp08.yaml).
"""

from __future__ import annotations

from models.base_model import BaseModel


class BaselineModel(BaseModel):
    def build(self) -> None:
        self._yolo = self._build_yolo()
