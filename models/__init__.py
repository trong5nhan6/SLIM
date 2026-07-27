"""Model registry — map idea -> BaseModel subclass.

Thêm idea mới (3 bước, không sửa train.py):
  1. Tạo models/<ten>.py định nghĩa 1 class kế thừa BaseModel (xem baseline.py, models/slim/) —
     implement build(), override get_callbacks() nếu cần callback riêng.
  2. Thêm entry vào MODEL_REGISTRY bên dưới.
  3. Tạo configs/<ten>.yaml kế thừa configs/base.yaml (xem configs/exp01.yaml).

train.py lấy --idea choices trực tiếp từ MODEL_REGISTRY, nên không cần sửa gì ở đó.

Chỉ còn 2 idea, phủ đúng 17 dòng trong results/SOTA-SLIM.xlsx:
  * slim     — họ Ours (EXP01-04, 06, 07, 12-17)
  * baseline — YOLOv11 gốc (EXP08, 09, 10)
Các idea cũ (lwso, fap, star, pd) đã gỡ. models/lwso/ vẫn còn nhưng nay là THƯ VIỆN dùng
chung chứ không phải idea: slim xây trên SPDConvGroup/C3k2Ghost/EMA/ECA/DySample/BiFPNCat
trong lwso/modules.py và NWD blend loss trong lwso/losses.py.
"""

from __future__ import annotations

from models.base_model import BaseModel
from models.baseline import BaselineModel
from models.slim import SlimModel

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "baseline": BaselineModel,
    "slim": SlimModel,
}


def build_model(idea: str, cfg) -> BaseModel:
    key = idea.lower()
    if key not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY)
        raise KeyError(f"idea={idea!r} chưa được đăng ký trong MODEL_REGISTRY. Có sẵn: {available}")
    return MODEL_REGISTRY[key](cfg)


__all__ = ["MODEL_REGISTRY", "build_model", "BaseModel", "BaselineModel", "SlimModel"]
