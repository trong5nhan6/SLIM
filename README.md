# Ours-YOLO — lightweight small-object YOLO cho VisDrone2019

Cải tiến YOLO11n cho small object trên ảnh UAV: **thêm đầu detect P2 (stride 4), bỏ tầng P5**,
SPDConv nhóm (downsample không mất thông tin), khối RepConv reparameterize được, EMA/ECA/C2PSA
attention, neck BiFPN-style, head `LSCDetect` chia sẻ conv + GroupNorm, loss CIoU⊕NWD.
Không fork ultralytics — module custom được đăng ký runtime qua `models.slim.register_slim()`.

Repo này đã được gọt lại đúng phạm vi bài báo: **mỗi file config = đúng một dòng trong
`results/SOTA-SLIM.xlsx`**. Các idea cũ (`lwso`, `fap`, `star`, `pd`) đã gỡ bỏ.

## Bảng thí nghiệm

| # | Config | Nội dung | Params | GFLOPs@640 |
|---|---|---|---|---|
| | | **Bảng 2 — đổi block (2x2: upsampler × block)** | | |
| 1 | `configs/exp01.yaml` | DySample + C3k2Ghost (SLIM) | 0.560M | 6.31 |
| 2 | `configs/exp02.yaml` | DySample + C3k2Rep | 0.945M | 9.78 |
| 3 | `configs/exp03.yaml` | bilinear + C3k2Ghost | 0.553M | 6.33 |
| 4 | `configs/exp04.yaml` | bilinear + C3k2Rep = **Ours-N** | 0.938M | 9.79 |
| | | **Bảng 3 — scale** | | |
| 5 | *(= exp04)* | Ours-N | 0.938M | 9.79 |
| 6 | `configs/exp06.yaml` | Ours-S | 2.373M | 23.65 |
| 7 | `configs/exp07.yaml` | Ours-M *(tùy chọn)* | 4.208M | 41.59 |
| 8 | `configs/exp08.yaml` | YOLOv11n from-scratch | 2.592M | 6.45 |
| 9 | `configs/exp09.yaml` | YOLOv11s from-scratch | 9.432M | 21.57 |
| 10 | `configs/exp10.yaml` | YOLOv11m from-scratch *(tùy chọn)* | 20.06M | 68.23 |
| | | **Bảng 4 — gỡ từng thành phần khỏi Ours-N** | | |
| 11 | *(= exp04)* | Ours-N (mốc so sánh) | 0.938M | 9.79 |
| 12 | `configs/exp12.yaml` | − LSCDetect (stock Detect) | 1.362M | 16.30 |
| 13 | `configs/exp13.yaml` | − nhánh P2 (head P3+P4) | 0.812M | 5.86 |
| 14 | `configs/exp14.yaml` | − NWD (CIoU thuần, chỉ đổi loss) | 0.938M | 9.79 |
| 15 | `configs/exp15.yaml` | − attention (EMA+C2PSA+ECA) | 0.872M | 9.33 |
| 16 | `configs/exp16.yaml` | − SPDConv (Conv stride-2) | 1.108M | 10.83 |
| 17 | `configs/exp17.yaml` | head GroupNorm → BatchNorm | 0.938M | 9.83 |
| 18 | `configs/exp18.yaml` | − P2, **+ P5** → pyramid P3/P4/P5 kiểu YOLO | 3.001M | 7.77 |

Hai dòng 13 và 18 trả lời hai câu hỏi khác nhau về P2, đừng gộp làm một:

* **13** = bỏ P2 và không thêm gì → head 2 scale P3+P4 (khung này vốn không có P5). Đây là
  ablation đúng chuẩn "đổi một biến", đo riêng chi phí/đóng góp của nhánh stride-4.
* **18** = bỏ P2 **và** thêm lại tầng stride-32 → pyramid P3/P4/P5 giống hệt YOLOv11 ships.
  Đây là mốc reviewer sẽ đòi: "w/o P2" trong literature nghĩa là FPN ba tầng tiêu chuẩn,
  không phải head hai tầng. Kết quả đo: 3.001M / 7.77G — **nặng gấp 3.2× Ours-N dù đã bỏ
  P2**, vì tầng P5 rộng 256ch và rung bottom-up P4→P5 là conv to nhất đồ thị. Nếu mAP vẫn
  giảm thì luận điểm mạnh hơn hẳn: không chỉ "P2 có ích" mà là "P2 đổi được nhiều mAP
  small-object trên mỗi FLOP hơn P5".

Số ở trên là bản **chưa fuse**, đo bằng `ultralytics.utils.torch_utils.get_flops` @640.
`tests/test_modules.py` regression-guard đúng những con số này (sai lệch >2% là fail).
Sau fuse/reparam, ví dụ Ours-N còn **0.893M / 9.23 GFLOPs**.

## Cấu trúc

```
models/                 mỗi idea = 1 module riêng, chọn qua --idea (xem "Thêm idea mới")
  base_model.py          BaseModel (ABC): build()/train()/get_callbacks() dùng chung mọi idea
                          + callback test-set monitoring mặc định (test_every)
                          + _compute_efficiency_metrics(): params/GFLOPs/size/latency/FPS
  baseline.py            idea "baseline": stock YOLO11n/s/m, không module/loss custom
  slim/                   idea "slim": họ Ours — LSCDetect head + neck P2 mỏng + RepConv
    model.py               SlimModel: register_slim() + patch_nwd_loss() rồi build YOLO
    modules.py              ConvGN, Scale, LSCDetect (head dùng chung conv giữa 3 scale),
                          LSCDetectBN (bản fuse được), RepBottleneck/C3k2Rep
    register.py              register_lwso() + patch parse_model nhận LSCDetect/C3k2Rep
  lwso/                   THƯ VIỆN dùng chung (không còn là idea) — slim xây trên đây
    modules.py              SPDConv(+Group), C3k2Ghost, EMA, ECA, DySample, BiFPNCat
    register.py              patch parse_model của ultralytics (pin 8.3.x)
    losses.py                 NWD blend loss (thay BboxLoss), dùng khi use_nwd: true
  _patch_utils.py         helper patch parse_model cộng dồn + patch đăng ký cho DDP
  __init__.py             MODEL_REGISTRY {idea: class} + build_model(idea, cfg)
cfg/                    kiến trúc model (.yaml theo format ultralytics parse_model)
                          1 file cho mỗi dòng bảng — xem "Bảng thí nghiệm" ở trên
configs/                hyperparameter theo thí nghiệm (train.py load)
  base.yaml             default cho mọi run — mọi config exp kế thừa file này
  exp<NN>.yaml          1 file cho mỗi dòng bảng
data/
  visdrone.yaml        dataset config (sửa `path:` nếu chuyển máy)
  scripts/convert_visdrone.py     VisDrone → YOLO format
train.py / val.py      entry point train & eval
tests/                 pytest shape/param sanity tests
results/               SOTA-SLIM.xlsx (bảng chính) + result.txt, slim_idea.txt (log thô)
logs/                   text log mỗi run (<name>.log, xem "Ghi chú quan trọng") — gitignored
```

## Bắt đầu

```bash
pip install -r requirements.txt
pytest tests/ -v                      # sanity check trước khi train

# 1) Tải VisDrone2019-DET (train/val/test-dev) rồi convert:
python data/scripts/convert_visdrone.py --src <thư mục chứa VisDrone2019-DET-*> --mask-ignored

# 2) Chạy 1 dòng bất kỳ trong bảng thí nghiệm:
python train.py --config configs/exp02.yaml

# 3) Override hyperparameter từ CLI (chỉ áp khi thực sự truyền):
python train.py --config configs/exp04.yaml --imgsz 640 --batch 16

# Đánh giá — in đủ 10 metric của bảng + 1 dòng CSV dán thẳng vào Excel:
python val.py --weights runs/detect/exp04/weights/best.pt --split val
```

`val.py` in ra: RUN INFO (epochs/imgsz/batch/seed đọc từ `train_args` trong checkpoint),
P/R/mAP50/mAP50-95, khối EFFICIENCY METRICS (params, GFLOPs, model size **fp32 lý thuyết**
và **dung lượng file `.pt` thật**, latency, FPS), rồi khối `PAPER TABLE ROW`.

`--imgsz` mặc định là **640** ở cả `configs/base.yaml` lẫn `val.py` — đúng imgsz của mọi dòng
trong bảng, nên không cần truyền tay. Nếu vẫn lỡ eval ở size khác size lúc train, `val.py`
cảnh báo ngay (GFLOPs và latency đều phụ thuộc bậc hai vào imgsz).

## Cấu hình train (`configs/` + `models/`)

`train.py` không hardcode hyperparameter hay kiến trúc — 2 tầng tách biệt:

- **`configs/`**: hyperparameter, merge theo thứ tự ưu tiên tăng dần —
  **`configs/base.yaml`** (default chung) → **`configs/exp<NN>.yaml`** (trỏ bằng `--config`)
  → **CLI flag** (override cuối cùng, chỉ áp dụng khi thực sự truyền).
- **`models/`**: code build model, mỗi idea 1 class kế thừa `BaseModel`
  (`models/base_model.py`), đăng ký trong `models/__init__.py` (`MODEL_REGISTRY`).

Hiện có 2 idea: **`slim`** (họ Ours) và **`baseline`** (YOLOv11 gốc). Vì mỗi thí nghiệm là
1 file `configs/exp<NN>.yaml` chứ không phải 1 file per-idea, hãy luôn dùng `--config`;
`--idea` chỉ còn quyết định class nào build model (đọc từ khóa `idea:` trong chính file đó).

### Thêm idea mới

1. Tạo `models/<tên>.py` (hoặc `models/<tên>/`) định nghĩa 1 class kế thừa `BaseModel`,
   implement `build()` (set `self._yolo`); override `get_callbacks()` nếu cần callback riêng,
   gọi `super().get_callbacks()` để giữ test-set monitoring mặc định.
2. Thêm entry vào `MODEL_REGISTRY` trong `models/__init__.py`.
3. Tạo `configs/<tên>.yaml` kế thừa `configs/base.yaml`.

## Kiến trúc Ours

Khung `lwso-eff` (tổ hợp thắng trong các vòng thử trước) + các thay đổi sau, mỗi cái có
đúng 1 dòng ablation trong Bảng 4:

1. **`LSCDetect`** (`models/slim/modules.py`) thay stock `Detect`: mỗi scale căn về 48ch qua
   1x1 ConvGN, rồi **dùng chung** 1 stack (3x3 dense + 3x3 depthwise, GroupNorm) + 1 conv box
   + 1 conv cls cho cả 3 scale; mỗi scale chỉ thêm 1 hệ số `Scale` học được ở nhánh box.
   Head chạy ở **mọi** scale mà P2 có diện tích gấp 16 lần P4, nên mỗi kernel thừa trong head
   bị nhân lên 16 lần — đo được: dùng stock Detect thay vào làm model từ 9.23 lên **15.68
   GFLOPs (+70%)**. Đây là thứ khiến kiến trúc P2 khả thi trong ngân sách.
2. **Neck thu vừa phải**: P2 out 64→48, P3 top-down/bottom-up 128→96 — điểm giữa có chủ đích
   giữa 32ch (quá mỏng, mất mAP toàn bộ class) và 64/128ch (12.4 GFLOPs).
3. **`C3k2Rep`** thay `C3k2Ghost`: RepConv train dạng đa nhánh rồi reparameterize về 1 conv
   3x3 dense khi inference. FLOPs tăng nhưng latency batch=1 **giảm** (Ghost/DWConv là
   memory-bound, GPU không no việc).
4. **Bilinear** thay `DySample`: DySample học offset rồi gọi `grid_sample` — rẻ FLOPs, đắt
   wall-clock. Đo được: đổi upsampler gần như **miễn phí** về chi phí (0.557→0.550M params,
   6.16→6.18 GFLOPs), nên toàn bộ phần compute tăng của Ours-N đến từ RepConv.
5. **`SPDConvGroup`** thay Conv stride-2: rearrange 2x2 → 4x kênh (không mất pixel) rồi trộn
   bằng grouped conv. Rẻ hơn conv stride-2 đúng một nửa (`0.5·c1·c2·9` vs `c1·c2·9` MAC/pixel)
   — bỏ nó đi làm model **đắt hơn**: 9.23 → 10.26 GFLOPs.
6. **NWD blend loss** (`models/lwso/losses.py`): CIoU⊕NWD, bật bằng `use_nwd: true`.

## Ghi chú quan trọng

- **VRAM**: đầu P2 @960 rất tốn bộ nhớ. batch 8 ≈ 12GB. Thiếu VRAM → giảm `--imgsz 768`
  hoặc `--batch 4`. Không dùng `--multi-scale` nếu <16GB.
- **NWD constant** (`--nwd-constant`): mặc định 12.8 theo paper (đơn vị pixel tuyệt đối),
  nhưng trong loss của ultralytics box ở đơn vị stride-normalized → nên ablate C ∈ {2, 4, 8, 12.8}.
- **Pin ultralytics 8.3.x**: `models/lwso/register.py` và `models/slim/register.py` patch
  source `parse_model` bằng các phép thay thế có kiểm tra; sai version sẽ raise RuntimeError
  ngay khi gọi `register_slim()` (fail sớm, không hỏng ngầm).
- **Ignored regions**: luôn convert với `--mask-ignored` — vùng ignore được tô xám 114
  để model không bị phạt oan khi có object thật trong đó.
- **Anaconda trên Windows**: nếu gặp `OMP Error #15 (libiomp5md.dll)`, các script đã tự set
  `KMP_DUPLICATE_LIB_OK=TRUE`; fix triệt để là dùng venv/conda env riêng thay vì base env.
- **Model size**: ultralytics lưu checkpoint ở **fp16**, nên file `.pt` thật chỉ bằng ~một nửa
  con số `params × 4` (fp32 lý thuyết). Các paper SOTA report dung lượng file thật — dùng dòng
  `Checkpoint file size (.pt)` của `val.py` cho bảng, đừng dùng dòng fp32.
- **Epochs/imgsz của một checkpoint** nằm trong chính nó: `torch.load(best.pt)['train_args']`.
  `val.py` đã tự đọc và in ra. Mọi dòng trong cùng một bảng phải khớp epochs, nếu không bảng
  mất giá trị so sánh.
- **`logs/<name>.log` — một file log cho mỗi config**: `name` lấy từ `configs/exp<NN>.yaml`
  (`name: exp<NN>`), nên chạy `configs/exp02.yaml` là ra `logs/exp02.log`. File **luôn** được
  tạo, không phụ thuộc `--test-every`. Nội dung, theo thứ tự:

  | Khối | Khi nào ghi | Nội dung |
  |---|---|---|
  | `RUN <name>` | đầu train | config nào, idea, model cfg, số layer, params, GFLOPs, loss, dataset, recipe (epochs/imgsz/batch/seed/device), thư mục output |
  | `[EPOCH n/N]` | **mỗi epoch** | train loss từng thành phần + val P/R/mAP50/mAP50-95 + lr |
  | `[test-set eval @ epoch n]` | mỗi `--test-every N` (0 = tắt riêng khối này) | test-split mAP + khối **EFFICIENCY METRICS** (params/GFLOPs/size/latency/FPS) |
  | `SUMMARY <name>` | cuối train | số epoch đã chạy, best fitness, đường dẫn best.pt |
  | `EVAL <name> split=...` | mỗi lần chạy `val.py` | run info + P/R/mAP + per-class + efficiency + dòng `PAPER TABLE ROW` |

  `val.py` **append** vào đúng file đó (tên run suy ra từ đường dẫn `--weights`; đổi bằng
  `--log-name`, tắt bằng `--no-log`), nên một config có **một file log** chứa cả train, test
  và thông tin model. Song song, `runs/detect/<name>/test_metrics.csv` vẫn được ghi mỗi lần
  trigger `--test-every`.
- **Seed**: `configs/base.yaml` set `seed: 42`. Override qua `--seed <n>` nếu cần nhiều seed
  để lấy mean±std.
- **Multi-GPU (`--device 0,1`)**: ultralytics spawn process con hoàn toàn mới từ 1 file `.py`
  tạm — process đó không import gì từ project nên `register_slim()` không tự chạy, sẽ crash
  khi build model. `models/_patch_utils.py:patch_ddp_registration()` vá thêm bước đăng ký vào
  chính file tạm đó (được `BaseModel.train()` tự gọi). Đã verify DDP thật trên Kaggle T4x2.
- ⚠️ **`final report/assets/make_figures.py`** vẽ sơ đồ từ các cfg đã xóa (`star`, `fap`,
  `lwso`), nên **không regenerate hình được nữa**. Các file PDF đã sinh trong `assets/` vẫn còn.

## Kết quả

Xem `results/SOTA-SLIM.xlsx` (bảng chính, 4 bảng × 17 dòng) và `results/result.txt`,
`results/slim_idea.txt` (log thô). Trích các dòng đã có số thật, VisDrone val 548 ảnh,
imgsz 640, Tesla T4, latency batch=1 fp32 sau fuse:

| Model | Params | GFLOPs | FPS | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| YOLOv11n | 2.584M | 6.30 | — | 0.2960 | 0.1690 |
| YOLOv11s | 9.417M | 21.32 | 97.5 | 0.3679 | 0.2178 |
| SLIM (exp01) | 0.557M | 6.16 | 84.8 | 0.2943 | 0.1683 |
| **Ours-N (exp04)** | **0.893M** | 9.23 | **98.0** | **0.3284** | **0.1907** |

Ours-N nhanh hơn SLIM (98.0 vs 84.8 FPS) **dù** GFLOPs cao hơn 50% — FLOPs không phải proxy
của latency trên GPU với các khối memory-bound. Đó là kết quả chính mà Bảng 2 tách bạch.
