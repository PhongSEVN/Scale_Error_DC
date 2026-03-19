import argparse
import sys
from pathlib import Path

CFG = dict(
    model="yolo11s.pt",
    data="dataset_yolo.yaml",
    imgsz=1024,
    epochs=100,
    patience=20,
    batch=8,
    workers=4,
    optimizer="AdamW",
    lr0=1e-3,
    lrf=0.01,
    momentum=0.937,
    weight_decay=5e-4,
    warmup_epochs=3,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=6.0,
    translate=0.1,
    scale=0.5,
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
    close_mosaic=10,
    project="runs/yolo",
    name="scale_error_dc",
    exist_ok=True,
    save=True,
    save_period=10,
    plots=True,
    val=True,
    device=0,
    amp=True,
    cache=False,
    verbose=True,
)


def check_env():
    print("[CHECK] Môi trường...")

    try:
        import ultralytics
        print(f"  ultralytics : {ultralytics.__version__}")
    except ImportError:
        print("  [ERROR] Chưa cài ultralytics → pip install ultralytics")
        sys.exit(1)

    try:
        import torch
        cuda = torch.cuda.is_available()
        print(f"  torch       : {torch.__version__}  CUDA={cuda}", end="")
        if cuda:
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  ({gpu_name}, {vram:.1f} GB)")
            if vram < 4:
                print("  [WARN] VRAM < 4 GB → --model yolo11n.pt --batch 2 --imgsz 640")
            elif vram < 6:
                print("  [WARN] VRAM < 6 GB → thử --batch 4 nếu OOM")
        else:
            print()
            print("  [WARN] Không có GPU, training sẽ rất chậm")
            CFG["device"] = "cpu"
            CFG["amp"] = False
    except ImportError:
        print("  [ERROR] Chưa cài torch")
        sys.exit(1)


def build_oversampled_list(label_dir: Path, img_dir: Path, out_txt: Path, max_repeat: int = 8):
    class_counts: dict[int, int] = {}
    img_classes: dict[str, list[int]] = {}

    for lbl in label_dir.glob("*.txt"):
        classes_in_img = []
        for line in lbl.read_text().strip().splitlines():
            if not line:
                continue
            cls = int(line.split()[0])
            classes_in_img.append(cls)
            class_counts[cls] = class_counts.get(cls, 0) + 1
        img_classes[lbl.stem] = classes_in_img

    if not class_counts:
        print("  [WARN] Không đọc được labels, bỏ qua oversampling")
        return

    total_ann = sum(class_counts.values())
    class_names = {0: "DiVat", 1: "DiVatLoiLom", 2: "LoiChi", 3: "LoiNhua", 4: "LoiTray"}
    print("  Phân bố trước oversampling:")
    for cls_id, cnt in sorted(class_counts.items()):
        bar = "█" * int(cnt / total_ann * 30)
        print(f"    {class_names.get(cls_id, cls_id):15s}: {cnt:6d} ({cnt/total_ann*100:5.1f}%) {bar}")

    max_count = max(class_counts.values())
    img_repeat: dict[str, int] = {}
    for stem, classes in img_classes.items():
        if not classes:
            img_repeat[stem] = 1
            continue
        rarest_count = min(class_counts[c] for c in set(classes))
        repeat = round(max_count / rarest_count)
        img_repeat[stem] = min(repeat, max_repeat)

    lines = []
    for stem, repeat in img_repeat.items():
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            p = img_dir / f"{stem}{ext}"
            if p.exists():
                img_path = p
                break
        if img_path is None:
            continue
        lines.extend([str(img_path.resolve())] * repeat)

    out_txt.write_text("\n".join(lines))

    n_original = len(img_classes)
    n_oversampled = len(lines)
    print(f"\n  Oversampling: {n_original} → {n_oversampled} ảnh "
          f"(×{n_oversampled/n_original:.1f}, max_repeat={max_repeat})")


def check_data():
    if not Path(CFG["data"]).exists():
        print(f"  [ERROR] Không tìm thấy {CFG['data']}")
        sys.exit(1)

    dirs = [
        Path("data/train/images"),
        Path("data/valid/images"),
        Path("data/train/labels"),
        Path("data/valid/labels"),
    ]
    ok = True
    for p in dirs:
        exists = p.exists()
        print(f"  {'✓' if exists else '✗ MISSING'}  {p}")
        if not exists:
            ok = False

    if not ok:
        print("\n  Tải dataset từ Roboflow (format YOLOv8/YOLOv11) vào data/")
        sys.exit(1)

    train_img = dirs[0]
    valid_img = dirs[1]
    n_train = len(list(train_img.glob("*.jpg"))) + len(list(train_img.glob("*.png")))
    n_valid = len(list(valid_img.glob("*.jpg"))) + len(list(valid_img.glob("*.png")))
    print(f"  train: {n_train} ảnh | valid: {n_valid} ảnh")


def train(oversample: bool = True):
    from ultralytics import YOLO

    if oversample:
        print("\n[OVERSAMPLE] Xử lý class imbalance...")
        out_txt = Path("data/train_oversampled.txt")
        build_oversampled_list(Path("data/train/labels"), Path("data/train/images"), out_txt)
        tmp_yaml = Path("dataset_yolo_oversampled.yaml")
        tmp_yaml.write_text(
            f"path: {Path('data').resolve().as_posix()}\n"
            f"train: {out_txt.resolve().as_posix()}\n"
            f"val: valid/images\n"
            f"nc: 5\n"
            f"names:\n"
            f"  0: DiVat\n"
            f"  1: DiVatLoiLom\n"
            f"  2: LoiChi\n"
            f"  3: LoiNhua\n"
            f"  4: LoiTray\n"
        )
        cfg = {**CFG, "data": str(tmp_yaml)}
    else:
        cfg = CFG

    print(f"\n[TRAIN] {cfg['model']}  imgsz={cfg['imgsz']}  batch={cfg['batch']}  epochs={cfg['epochs']}")
    print(f"        output → {cfg['project']}/{cfg['name']}\n")

    model = YOLO(cfg["model"])
    model.train(**cfg)


def validate(weights: str):
    from ultralytics import YOLO

    w = Path(weights)
    if not w.exists():
        print(f"[SKIP] Không tìm thấy weights: {w}")
        return

    print(f"\n[VAL] {w}")
    model = YOLO(str(w))
    m = model.val(data=CFG["data"], imgsz=CFG["imgsz"], batch=CFG["batch"])
    print(f"\n  mAP50    : {m.box.map50:.4f}")
    print(f"  mAP50-95 : {m.box.map:.4f}")


def main():
    parser = argparse.ArgumentParser(description="YOLO Training - Scale Error DC")
    parser.add_argument("--model",         default=None)
    parser.add_argument("--batch",         type=int, default=None)
    parser.add_argument("--imgsz",         type=int, default=None)
    parser.add_argument("--epochs",        type=int, default=None)
    parser.add_argument("--device",        default=None)
    parser.add_argument("--no-oversample", action="store_true")
    parser.add_argument("--val-only",      metavar="WEIGHTS", default=None)
    args = parser.parse_args()

    for key, val in [("model", args.model), ("batch", args.batch),
                     ("imgsz", args.imgsz), ("epochs", args.epochs),
                     ("device", args.device)]:
        if val is not None:
            CFG[key] = val

    print("=" * 60)
    print("  YOLO Training - Scale Error DC")
    print("=" * 60)

    check_env()

    if args.val_only:
        validate(args.val_only)
        return

    print("\n[CHECK] Dataset...")
    check_data()

    train(oversample=not args.no_oversample)
    validate(f"runs/yolo/scale_error_dc/weights/best.pt")


if __name__ == "__main__":
    main()
