"""
================================================================================
  🔬 UNIVERSAL COMPUTER VISION DATASET EXPLORER
================================================================================
  Tự động phát hiện và khám phá MỌI loại dataset phổ biến cho Computer Vision:
    ✅ COCO Format          (JSON annotations)
    ✅ YOLO Format          (.txt labels, normalized xywh)
    ✅ Pascal VOC Format    (XML annotations)
    ✅ Image Classification (folder-based: class_name/image.jpg)
    ✅ Segmentation Masks   (images + masks folders)

  Cách dùng:
    1. Điền đường dẫn dataset vào biến DATASET_ROOT bên dưới
    2. Chạy: python explore_dataset.py
================================================================================
"""

import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         ĐIỀN ĐƯỜNG DẪN Ở ĐÂY                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
DATASET_ROOT = r"d:\IT\Projects\Scale_Error_DC\data"
# ==============================================================================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}
MASK_KEYWORDS = {"mask", "masks", "label", "labels", "seg", "segmentation",
                 "annotation", "annotations", "gt", "groundtruth", "ground_truth"}


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def fmt_size(n_bytes):
    """Format bytes thành KB/MB/GB dễ đọc."""
    if n_bytes >= 1024**3:
        return f"{n_bytes / 1024**3:.2f} GB"
    elif n_bytes >= 1024**2:
        return f"{n_bytes / 1024**2:.1f} MB"
    elif n_bytes >= 1024:
        return f"{n_bytes / 1024:.1f} KB"
    return f"{n_bytes} bytes"


def dir_size(path):
    """Tính tổng dung lượng thư mục."""
    total = 0
    for f in Path(path).rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def count_images(path):
    """Đếm số file ảnh trong thư mục (đệ quy)."""
    return sum(1 for f in Path(path).rglob("*") if f.suffix.lower() in IMG_EXTS)


def list_images(path, recursive=True):
    """Liệt kê tất cả file ảnh."""
    gen = Path(path).rglob("*") if recursive else Path(path).iterdir()
    return [f for f in gen if f.is_file() and f.suffix.lower() in IMG_EXTS]


def get_image_size_pil(path):
    """Lấy kích thước ảnh bằng đọc header (không cần PIL)."""
    try:
        # Thử đọc JPEG header
        with open(path, "rb") as f:
            head = f.read(32)
            if head[:2] == b'\xff\xd8':  # JPEG
                f.seek(0)
                data = f.read()
                idx = 2
                while idx < len(data) - 1:
                    if data[idx] != 0xFF:
                        break
                    marker = data[idx + 1]
                    if marker in (0xC0, 0xC1, 0xC2):
                        h = int.from_bytes(data[idx+5:idx+7], 'big')
                        w = int.from_bytes(data[idx+7:idx+9], 'big')
                        return w, h
                    else:
                        length = int.from_bytes(data[idx+2:idx+4], 'big')
                        idx += 2 + length
            elif head[:8] == b'\x89PNG\r\n\x1a\n':  # PNG
                w = int.from_bytes(head[16:20], 'big')
                h = int.from_bytes(head[20:24], 'big')
                return w, h
            elif head[:2] == b'BM':  # BMP
                w = int.from_bytes(head[18:22], 'little')
                h = abs(int.from_bytes(head[22:26], 'little', signed=True))
                return w, h
    except:
        pass
    return None, None


def print_bar(label, count, total, width=30):
    """In thanh biểu đồ ngang."""
    pct = (count / total * 100) if total > 0 else 0
    filled = int(pct / 100 * width)
    bar = "█" * filled + "░" * (width - filled)
    print(f"     {label:30s} : {count:6,} ({pct:5.1f}%) |{bar}|")


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_subheader(title):
    print(f"\n  {'─'*50}")
    print(f"  {title}")
    print(f"  {'─'*50}")


# ══════════════════════════════════════════════════════════════════════════════
#  FORMAT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_format(root):
    """
    Tự động phát hiện loại dataset.
    Returns: str - "coco", "yolo", "voc", "classification", "segmentation", "unknown"
    """
    root = Path(root)
    detected = []

    # --- Check COCO ---
    coco_jsons = list(root.rglob("*annotations*.json")) + list(root.rglob("*coco*.json"))
    # Cũng check trong các subfolder phổ biến
    for sub in ["train", "valid", "val", "test", "train2017", "val2017"]:
        sub_dir = root / sub
        if sub_dir.exists():
            coco_jsons.extend(sub_dir.glob("*.json"))
    coco_jsons = list(set(coco_jsons))
    # Verify it's actually COCO format
    for jf in coco_jsons:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "images" in data and "annotations" in data:
                detected.append("coco")
                break
        except:
            pass

    # --- Check YOLO ---
    txt_files = list(root.rglob("*.txt"))
    # Kiểm tra xem có file labels/ hoặc .txt song song với ảnh không
    for txt in txt_files[:20]:  # Sample 20 files
        if txt.name in ("classes.txt", "data.yaml", "README.txt",
                        "README.dataset.txt", "README.roboflow.txt", "notes.txt"):
            continue
        try:
            with open(txt, "r") as f:
                first_line = f.readline().strip()
            if first_line:
                parts = first_line.split()
                if len(parts) == 5:
                    int(parts[0])  # class_id
                    floats = [float(x) for x in parts[1:]]
                    if all(0 <= v <= 1 for v in floats):
                        detected.append("yolo")
                        break
        except:
            pass

    # --- Check Pascal VOC ---
    xml_files = list(root.rglob("*.xml"))
    for xf in xml_files[:10]:
        try:
            tree = ET.parse(xf)
            r = tree.getroot()
            if r.tag == "annotation" and r.find("object") is not None:
                detected.append("voc")
                break
        except:
            pass

    # --- Check Classification (folder-based) ---
    # Cấu trúc: root/split/class_name/image.jpg hoặc root/class_name/image.jpg
    subdirs = [d for d in root.iterdir() if d.is_dir()
               and d.name not in {"__pycache__", ".git", ".idea"}]
    
    if subdirs:
        # Check nếu các subfolder chứa trực tiếp ảnh (classification)
        folder_has_images = 0
        for sd in subdirs[:20]:
            imgs = [f for f in sd.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS]
            if imgs:
                folder_has_images += 1
        
        # Nếu >= 2 subfolder đều chứa ảnh → classification
        if folder_has_images >= 2:
            # Kiểm tra thêm: không phải là train/val split
            split_names = {"train", "valid", "val", "test", "train2017", "val2017"}
            non_split_dirs = [d for d in subdirs if d.name.lower() not in split_names]
            if len(non_split_dirs) >= 2:
                detected.append("classification")
            else:
                # Check bên trong train/val có dạng class folders không
                for sd in subdirs:
                    if sd.name.lower() in split_names:
                        inner_dirs = [d for d in sd.iterdir() if d.is_dir()]
                        inner_has_imgs = sum(1 for d in inner_dirs[:20]
                                             if any(f.suffix.lower() in IMG_EXTS 
                                                    for f in d.iterdir() if f.is_file()))
                        if inner_has_imgs >= 2:
                            detected.append("classification")
                            break

    # --- Check Segmentation (image + mask pairs) ---
    dir_names = {d.name.lower() for d in root.rglob("*") if d.is_dir()}
    if MASK_KEYWORDS & dir_names:
        detected.append("segmentation")

    if not detected:
        detected.append("unknown")

    return list(set(detected))


# ══════════════════════════════════════════════════════════════════════════════
#  COCO FORMAT EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

def find_coco_splits(root):
    """Tìm tất cả COCO annotation JSON files và cặp chúng với thư mục ảnh."""
    root = Path(root)
    splits = []

    # Tìm tất cả JSON files có thể là COCO annotations
    json_files = list(root.rglob("*.json"))
    
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "images" not in data or "annotations" not in data:
                continue
            
            # Xác định thư mục ảnh (cùng thư mục với JSON)
            img_dir = jf.parent
            split_name = jf.parent.name if jf.parent != root else jf.stem
            splits.append({
                "name": split_name,
                "json_path": jf,
                "image_dir": img_dir,
                "data": data
            })
        except:
            pass
    
    return splits


def explore_coco(root):
    """Khám phá dataset dạng COCO."""
    print_header("📦 COCO FORMAT DATASET")
    
    splits = find_coco_splits(root)
    if not splits:
        print("  ❌ Không tìm thấy file annotation COCO hợp lệ!")
        return

    all_results = {}

    for split in splits:
        data = split["data"]
        name = split["name"]
        img_dir = split["image_dir"]

        print_subheader(f"📂 Split: {name.upper()}")

        images = data.get("images", [])
        annotations = data.get("annotations", [])
        categories = data.get("categories", [])

        # Thống kê cơ bản
        print(f"\n  📊 Thống kê tổng quát:")
        print(f"     • Annotation file  : {split['json_path'].name}")
        print(f"     • Số lượng ảnh      : {len(images):,}")
        print(f"     • Số lượng annotation: {len(annotations):,}")
        print(f"     • Số lượng classes   : {len(categories)}")

        # Categories
        cat_id_to_name = {}
        print(f"\n  🏷️  Danh sách classes:")
        for cat in categories:
            cat_id_to_name[cat["id"]] = cat["name"]
            supcat = cat.get("supercategory", "")
            extra = f" (super: {supcat})" if supcat else ""
            print(f"     • ID={cat['id']:3d} | {cat['name']}{extra}")

        # Phân bố class
        class_counts = Counter()
        for ann in annotations:
            cat_name = cat_id_to_name.get(ann["category_id"], f"unknown_{ann['category_id']}")
            class_counts[cat_name] += 1

        if class_counts:
            print(f"\n  📈 Phân bố annotation theo class:")
            total_anns = sum(class_counts.values())
            for cls_name, count in class_counts.most_common():
                print_bar(cls_name, count, total_anns)

        # Annotations per image
        anns_per_image = Counter()
        for ann in annotations:
            anns_per_image[ann["image_id"]] += 1

        images_with = len(anns_per_image)
        images_without = len(images) - images_with
        counts = list(anns_per_image.values()) if anns_per_image else [0]

        print(f"\n  🔢 Annotations per image:")
        print(f"     • Có annotation     : {images_with:,}")
        print(f"     • Không annotation  : {images_without:,}")
        print(f"     • Trung bình / ảnh  : {sum(counts)/len(counts):.2f}")
        print(f"     • Min / Max         : {min(counts)} / {max(counts)}")

        # Phân bố objects/ảnh
        obj_dist = Counter(anns_per_image.values())
        print(f"\n  📊 Phân bố số objects/ảnh:")
        for n_obj in sorted(obj_dist.keys()):
            n_imgs = obj_dist[n_obj]
            bar_len = min(int(n_imgs / max(obj_dist.values()) * 30), 30) if obj_dist else 0
            print(f"     {n_obj:3d} obj → {n_imgs:5,} ảnh {'█' * bar_len}")

        # Kích thước ảnh
        widths = [img.get("width", 0) for img in images]
        heights = [img.get("height", 0) for img in images]
        if widths and heights:
            unique_sizes = set(zip(widths, heights))
            print(f"\n  📐 Kích thước ảnh:")
            if len(unique_sizes) <= 10:
                for w, h in sorted(unique_sizes):
                    c = sum(1 for iw, ih in zip(widths, heights) if iw == w and ih == h)
                    print(f"     • {w} × {h} : {c:,} ảnh")
            else:
                print(f"     • {len(unique_sizes)} loại kích thước khác nhau")
                print(f"     • Width  : {min(widths)} → {max(widths)} (avg: {sum(widths)/len(widths):.0f})")
                print(f"     • Height : {min(heights)} → {max(heights)} (avg: {sum(heights)/len(heights):.0f})")

        # Bounding Box stats
        bbox_ws, bbox_hs, bbox_areas = [], [], []
        has_segmentation = False
        has_keypoints = False
        for ann in annotations:
            if "bbox" in ann and len(ann["bbox"]) == 4:
                _, _, bw, bh = ann["bbox"]
                bbox_ws.append(bw)
                bbox_hs.append(bh)
                bbox_areas.append(bw * bh)
            if "segmentation" in ann and ann["segmentation"]:
                has_segmentation = True
            if "keypoints" in ann:
                has_keypoints = True

        if bbox_ws:
            print(f"\n  📏 Bounding Box (COCO: [x,y,w,h]):")
            print(f"     • Width  : {min(bbox_ws):.1f} → {max(bbox_ws):.1f} (avg: {sum(bbox_ws)/len(bbox_ws):.1f})")
            print(f"     • Height : {min(bbox_hs):.1f} → {max(bbox_hs):.1f} (avg: {sum(bbox_hs)/len(bbox_hs):.1f})")
            print(f"     • Area   : {min(bbox_areas):.0f} → {max(bbox_areas):.0f} (avg: {sum(bbox_areas)/len(bbox_areas):.0f})")
            
            small = sum(1 for a in bbox_areas if a < 32**2)
            medium = sum(1 for a in bbox_areas if 32**2 <= a < 96**2)
            large = sum(1 for a in bbox_areas if a >= 96**2)
            print(f"\n  📐 Size distribution (COCO standard):")
            print_bar("Small  (< 32²)", small, len(bbox_areas))
            print_bar("Medium (32² ~ 96²)", medium, len(bbox_areas))
            print_bar("Large  (≥ 96²)", large, len(bbox_areas))

        # Annotation types
        print(f"\n  🧩 Loại annotation có trong dataset:")
        print(f"     • Bounding Box    : {'✅' if bbox_ws else '❌'}")
        print(f"     • Segmentation    : {'✅' if has_segmentation else '❌'}")
        print(f"     • Keypoints       : {'✅' if has_keypoints else '❌'}")

        # File integrity
        if img_dir.is_dir():
            actual = {f.name for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXTS}
            expected = {img["file_name"] for img in images}
            missing = expected - actual
            print(f"\n  🔍 Kiểm tra file:")
            print(f"     • Ảnh thực tế   : {len(actual):,}")
            print(f"     • Ảnh trong JSON: {len(expected):,}")
            if missing:
                print(f"     • ⚠️  Thiếu {len(missing)} file!")
            else:
                print(f"     • ✅ Đầy đủ!")

        # Disk size
        if img_dir.is_dir():
            print(f"\n  💾 Dung lượng: {fmt_size(dir_size(img_dir))}")

        all_results[name] = {
            "n_images": len(images),
            "n_annotations": len(annotations),
            "class_counts": dict(class_counts),
        }

    # Tổng kết
    if len(all_results) > 1:
        print_subheader("📋 TỔNG KẾT COCO")
        total_imgs = sum(r["n_images"] for r in all_results.values())
        total_anns = sum(r["n_annotations"] for r in all_results.values())
        print(f"  • Tổng ảnh        : {total_imgs:,}")
        print(f"  • Tổng annotations: {total_anns:,}")
        for name, r in all_results.items():
            pct = r["n_images"] / total_imgs * 100 if total_imgs else 0
            print(f"  • {name:10s}: {r['n_images']:6,} ảnh ({pct:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
#  YOLO FORMAT EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

def find_yolo_splits(root):
    """Tìm splits cho YOLO dataset."""
    root = Path(root)
    splits = []
    
    # Check standard YOLO structure: images/train, labels/train
    # hoặc train/images, train/labels
    # hoặc flat: train/*.jpg + train/*.txt
    
    possible_splits = ["train", "valid", "val", "test"]
    
    for sp in possible_splits:
        # Pattern 1: root/images/split + root/labels/split
        img_dir = root / "images" / sp
        lbl_dir = root / "labels" / sp
        if img_dir.is_dir() and lbl_dir.is_dir():
            splits.append({"name": sp, "image_dir": img_dir, "label_dir": lbl_dir})
            continue
        
        # Pattern 2: root/split/images + root/split/labels
        img_dir = root / sp / "images"
        lbl_dir = root / sp / "labels"
        if img_dir.is_dir() and lbl_dir.is_dir():
            splits.append({"name": sp, "image_dir": img_dir, "label_dir": lbl_dir})
            continue
        
        # Pattern 3: root/split (flat - images + txt in same folder)
        sp_dir = root / sp
        if sp_dir.is_dir():
            has_imgs = any(f.suffix.lower() in IMG_EXTS for f in sp_dir.iterdir() if f.is_file())
            has_txts = any(f.suffix == ".txt" for f in sp_dir.iterdir() if f.is_file())
            if has_imgs and has_txts:
                splits.append({"name": sp, "image_dir": sp_dir, "label_dir": sp_dir})
                continue

    # Nếu không tìm thấy splits, check root level
    if not splits:
        img_dir = root / "images"
        lbl_dir = root / "labels"
        if img_dir.is_dir() and lbl_dir.is_dir():
            splits.append({"name": "all", "image_dir": img_dir, "label_dir": lbl_dir})
        else:
            has_imgs = any(f.suffix.lower() in IMG_EXTS for f in root.iterdir() if f.is_file())
            has_txts = any(f.suffix == ".txt" for f in root.iterdir() if f.is_file())
            if has_imgs and has_txts:
                splits.append({"name": "all", "image_dir": root, "label_dir": root})

    return splits


def explore_yolo(root):
    """Khám phá dataset dạng YOLO."""
    print_header("📦 YOLO FORMAT DATASET")
    
    root = Path(root)
    
    # Load classes
    classes = []
    for cls_file in [root / "classes.txt", root / "obj.names"]:
        if cls_file.exists():
            classes = [l.strip() for l in open(cls_file) if l.strip()]
            break
    
    # Check data.yaml
    yaml_file = None
    for yf in [root / "data.yaml", root / "dataset.yaml", root / "data.yml"]:
        if yf.exists():
            yaml_file = yf
            try:
                with open(yf, "r", encoding="utf-8") as f:
                    content = f.read()
                # Simple YAML parsing for names
                if "names:" in content:
                    import re
                    # Try list format: names: ['a', 'b']
                    match = re.search(r"names:\s*\[(.+?)\]", content)
                    if match:
                        classes = [n.strip().strip("'\"") for n in match.group(1).split(",")]
                    else:
                        # Try multi-line format
                        lines = content.split("\n")
                        in_names = False
                        for line in lines:
                            if line.strip().startswith("names:"):
                                in_names = True
                                continue
                            if in_names:
                                if line.strip().startswith("- "):
                                    classes.append(line.strip()[2:].strip("'\""))
                                elif line.strip().startswith(("nc:", "train:", "val:", "test:", "path:")):
                                    in_names = False
                                elif re.match(r"\s+\d+:\s+", line):
                                    classes.append(line.split(":", 1)[1].strip().strip("'\""))
            except:
                pass
            break

    if classes:
        print(f"\n  🏷️  Classes ({len(classes)}):")
        for i, c in enumerate(classes):
            print(f"     • {i}: {c}")
    if yaml_file:
        print(f"\n  📄 Config file: {yaml_file.name}")

    splits = find_yolo_splits(root)
    if not splits:
        print("  ❌ Không tìm thấy cấu trúc YOLO hợp lệ!")
        return

    all_results = {}

    for split in splits:
        name = split["name"]
        img_dir = split["image_dir"]
        lbl_dir = split["label_dir"]

        print_subheader(f"📂 Split: {name.upper()}")

        images = list_images(img_dir, recursive=False)
        label_files = list(lbl_dir.glob("*.txt"))
        label_files = [f for f in label_files
                       if f.name not in ("classes.txt", "data.yaml", "README.txt",
                                         "README.dataset.txt", "README.roboflow.txt")]

        print(f"\n  📊 Thống kê:")
        print(f"     • Số ảnh       : {len(images):,}")
        print(f"     • Số label file: {len(label_files):,}")

        # Match images ↔ labels
        img_stems = {f.stem for f in images}
        lbl_stems = {f.stem for f in label_files}
        matched = img_stems & lbl_stems
        img_only = img_stems - lbl_stems
        lbl_only = lbl_stems - img_stems
        print(f"     • Matched      : {len(matched):,}")
        if img_only:
            print(f"     • ⚠️  Ảnh không có label: {len(img_only)}")
        if lbl_only:
            print(f"     • ⚠️  Label không có ảnh: {len(lbl_only)}")

        # Parse labels
        class_counts = Counter()
        objs_per_image = []
        bbox_ws, bbox_hs = [], []
        empty_labels = 0

        for lf in label_files:
            try:
                with open(lf, "r") as f:
                    lines = [l.strip() for l in f if l.strip()]
                if not lines:
                    empty_labels += 1
                    objs_per_image.append(0)
                    continue
                objs_per_image.append(len(lines))
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cls_name = classes[cls_id] if cls_id < len(classes) else f"class_{cls_id}"
                        class_counts[cls_name] += 1
                        bbox_ws.append(float(parts[3]))
                        bbox_hs.append(float(parts[4]))
            except:
                pass

        total_anns = sum(class_counts.values())
        print(f"     • Tổng objects : {total_anns:,}")
        if empty_labels:
            print(f"     • Label trống  : {empty_labels:,} (background/negative)")

        if class_counts:
            print(f"\n  📈 Phân bố class:")
            for cls_name, count in class_counts.most_common():
                print_bar(cls_name, count, total_anns)

        if objs_per_image:
            print(f"\n  🔢 Objects per image:")
            print(f"     • Avg: {sum(objs_per_image)/len(objs_per_image):.2f}")
            print(f"     • Min: {min(objs_per_image)} | Max: {max(objs_per_image)}")

        if bbox_ws:
            print(f"\n  📏 BBox size (normalized 0-1):")
            print(f"     • Width  : {min(bbox_ws):.4f} → {max(bbox_ws):.4f} (avg: {sum(bbox_ws)/len(bbox_ws):.4f})")
            print(f"     • Height : {min(bbox_hs):.4f} → {max(bbox_hs):.4f} (avg: {sum(bbox_hs)/len(bbox_hs):.4f})")

        # Image sizes (sample)
        sample_imgs = images[:50]
        sizes = []
        for img in sample_imgs:
            w, h = get_image_size_pil(img)
            if w and h:
                sizes.append((w, h))
        if sizes:
            unique = set(sizes)
            print(f"\n  📐 Kích thước ảnh (sampled {len(sizes)}):")
            if len(unique) <= 5:
                for w, h in sorted(unique):
                    print(f"     • {w} × {h}")
            else:
                ws = [s[0] for s in sizes]
                hs = [s[1] for s in sizes]
                print(f"     • Width  : {min(ws)} → {max(ws)}")
                print(f"     • Height : {min(hs)} → {max(hs)}")

        print(f"\n  💾 Dung lượng ảnh: {fmt_size(dir_size(img_dir))}")

        all_results[name] = {"n_images": len(images), "n_labels": len(label_files),
                             "n_objects": total_anns}

    if len(all_results) > 1:
        print_subheader("📋 TỔNG KẾT YOLO")
        for name, r in all_results.items():
            print(f"  • {name:10s}: {r['n_images']:6,} ảnh, {r['n_objects']:6,} objects")


# ══════════════════════════════════════════════════════════════════════════════
#  PASCAL VOC FORMAT EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

def explore_voc(root):
    """Khám phá dataset dạng Pascal VOC."""
    print_header("📦 PASCAL VOC FORMAT DATASET")
    
    root = Path(root)
    xml_files = list(root.rglob("*.xml"))
    
    if not xml_files:
        print("  ❌ Không tìm thấy file XML annotation!")
        return

    # Group by parent directory (potential splits)
    by_dir = defaultdict(list)
    for xf in xml_files:
        by_dir[xf.parent].append(xf)

    for dir_path, files in by_dir.items():
        rel_path = dir_path.relative_to(root) if dir_path != root else Path(".")
        print_subheader(f"📂 {rel_path} ({len(files):,} annotations)")

        class_counts = Counter()
        objs_per_image = []
        bbox_ws, bbox_hs = [], []
        img_sizes = set()
        difficulties = Counter()
        truncated_count = 0
        total_objs = 0

        for xf in files:
            try:
                tree = ET.parse(xf)
                r = tree.getroot()
                
                # Image size
                size = r.find("size")
                if size is not None:
                    w = int(size.findtext("width", "0"))
                    h = int(size.findtext("height", "0"))
                    if w and h:
                        img_sizes.add((w, h))

                objects = r.findall("object")
                objs_per_image.append(len(objects))

                for obj in objects:
                    cls_name = obj.findtext("name", "unknown")
                    class_counts[cls_name] += 1
                    total_objs += 1

                    diff = obj.findtext("difficult", "0")
                    difficulties[diff] += 1
                    if obj.findtext("truncated", "0") == "1":
                        truncated_count += 1

                    bndbox = obj.find("bndbox")
                    if bndbox is not None:
                        x1 = float(bndbox.findtext("xmin", "0"))
                        y1 = float(bndbox.findtext("ymin", "0"))
                        x2 = float(bndbox.findtext("xmax", "0"))
                        y2 = float(bndbox.findtext("ymax", "0"))
                        bbox_ws.append(x2 - x1)
                        bbox_hs.append(y2 - y1)
            except:
                pass

        print(f"\n  📊 Thống kê:")
        print(f"     • Số annotation XML : {len(files):,}")
        print(f"     • Tổng objects      : {total_objs:,}")
        print(f"     • Số classes        : {len(class_counts)}")

        if class_counts:
            print(f"\n  📈 Phân bố class:")
            for cls_name, count in class_counts.most_common():
                print_bar(cls_name, count, total_objs)

        if objs_per_image:
            print(f"\n  🔢 Objects per image:")
            print(f"     • Avg: {sum(objs_per_image)/len(objs_per_image):.2f}")
            print(f"     • Min: {min(objs_per_image)} | Max: {max(objs_per_image)}")

        if img_sizes:
            print(f"\n  📐 Kích thước ảnh ({len(img_sizes)} loại):")
            for w, h in sorted(img_sizes)[:10]:
                print(f"     • {w} × {h}")

        if bbox_ws:
            print(f"\n  📏 BBox (VOC: [xmin,ymin,xmax,ymax]):")
            print(f"     • Width  : {min(bbox_ws):.0f} → {max(bbox_ws):.0f} (avg: {sum(bbox_ws)/len(bbox_ws):.0f})")
            print(f"     • Height : {min(bbox_hs):.0f} → {max(bbox_hs):.0f} (avg: {sum(bbox_hs)/len(bbox_hs):.0f})")

        print(f"\n  🧩 Thuộc tính:")
        print(f"     • Difficult : {difficulties}")
        print(f"     • Truncated : {truncated_count:,}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLASSIFICATION FORMAT EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

def explore_classification(root):
    """Khám phá dataset dạng Image Classification (folder-based)."""
    print_header("📦 IMAGE CLASSIFICATION DATASET")
    
    root = Path(root)
    split_names = {"train", "valid", "val", "test", "training", "validation", "testing"}
    
    # Detect splits
    subdirs = [d for d in root.iterdir() if d.is_dir() and d.name not in {"__pycache__", ".git"}]
    potential_splits = [d for d in subdirs if d.name.lower() in split_names]
    
    if potential_splits:
        # Has train/val/test splits
        for split_dir in sorted(potential_splits, key=lambda d: d.name):
            _explore_classification_folder(split_dir, split_dir.name)
    else:
        # No splits, treat entire root as one set
        _explore_classification_folder(root, "all")


def _explore_classification_folder(folder, name):
    """Khám phá 1 folder classification."""
    print_subheader(f"📂 Split: {name.upper()}")
    
    class_dirs = [d for d in sorted(folder.iterdir())
                  if d.is_dir() and d.name not in {"__pycache__", ".git"}]
    
    if not class_dirs:
        print("  ❌ Không tìm thấy class folders!")
        return

    class_counts = {}
    total_images = 0

    for cd in class_dirs:
        imgs = [f for f in cd.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS]
        class_counts[cd.name] = len(imgs)
        total_images += len(imgs)

    print(f"\n  📊 Thống kê:")
    print(f"     • Số classes     : {len(class_counts)}")
    print(f"     • Tổng số ảnh    : {total_images:,}")
    print(f"     • Avg ảnh/class  : {total_images/len(class_counts):.0f}")

    print(f"\n  📈 Phân bố class:")
    for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print_bar(cls_name, count, total_images)

    # Check imbalance
    counts = list(class_counts.values())
    if counts:
        ratio = max(counts) / max(min(counts), 1)
        print(f"\n  ⚖️  Class imbalance ratio: {ratio:.1f}x (max/min)")
        if ratio > 5:
            print(f"     ⚠️  Dataset bị mất cân bằng nghiêm trọng!")
        elif ratio > 2:
            print(f"     ⚠️  Dataset hơi mất cân bằng")
        else:
            print(f"     ✅ Dataset khá cân bằng")

    # Sample image sizes
    all_imgs = list_images(folder)[:100]
    sizes = []
    for img in all_imgs:
        w, h = get_image_size_pil(img)
        if w and h:
            sizes.append((w, h))
    if sizes:
        unique = set(sizes)
        ws = [s[0] for s in sizes]
        hs = [s[1] for s in sizes]
        print(f"\n  📐 Kích thước ảnh (sampled {len(sizes)}):")
        print(f"     • Số loại size  : {len(unique)}")
        print(f"     • Width  : {min(ws)} → {max(ws)}")
        print(f"     • Height : {min(hs)} → {max(hs)}")

    print(f"\n  💾 Dung lượng: {fmt_size(dir_size(folder))}")


# ══════════════════════════════════════════════════════════════════════════════
#  SEGMENTATION FORMAT EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

def explore_segmentation(root):
    """Khám phá dataset dạng Segmentation (images + masks)."""
    print_header("📦 SEGMENTATION DATASET")
    
    root = Path(root)
    
    # Tìm cặp thư mục images/masks
    all_dirs = {d.name.lower(): d for d in root.rglob("*") if d.is_dir()}
    
    img_keywords = {"images", "imgs", "image", "img", "jpegimages", "photos"}
    mask_keywords = {"masks", "mask", "labels", "label", "annotations",
                     "segmentation", "seg", "gt", "groundtruth",
                     "labelimages", "labelmask", "semantic"}

    img_dirs = {name: path for name, path in all_dirs.items() if name in img_keywords}
    mask_dirs = {name: path for name, path in all_dirs.items() if name in mask_keywords}

    if not img_dirs:
        # Fallback: tìm thư mục có nhiều ảnh nhất
        print("  ℹ️  Không tìm thấy thư mục 'images'. Quét tất cả thư mục...")
        img_dirs = {"root": root}
    
    if not mask_dirs:
        print("  ⚠️  Không tìm thấy thư mục masks riêng biệt.")
        return

    for mname, mdir in mask_dirs.items():
        print_subheader(f"📂 Mask folder: {mname}")
        
        mask_files = [f for f in mdir.iterdir()
                      if f.is_file() and f.suffix.lower() in IMG_EXTS | {".npy"}]
        
        print(f"\n  📊 Thống kê:")
        print(f"     • Số mask files : {len(mask_files):,}")
        
        # Check matching with images
        for iname, idir in img_dirs.items():
            img_files = list_images(idir, recursive=False)
            img_stems = {f.stem for f in img_files}
            mask_stems = {f.stem for f in mask_files}
            matched = img_stems & mask_stems
            print(f"     • Matched với '{iname}': {len(matched):,} / {len(img_files):,}")

        # Sample mask info
        sample_masks = mask_files[:20]
        mask_sizes = []
        for mf in sample_masks:
            w, h = get_image_size_pil(mf)
            if w and h:
                mask_sizes.append((w, h))
        
        if mask_sizes:
            unique = set(mask_sizes)
            print(f"\n  📐 Kích thước mask (sampled):")
            for w, h in sorted(unique):
                print(f"     • {w} × {h}")

        print(f"\n  💾 Dung lượng masks: {fmt_size(dir_size(mdir))}")


# ══════════════════════════════════════════════════════════════════════════════
#  GENERAL STATS (for unknown or any format)
# ══════════════════════════════════════════════════════════════════════════════

def explore_general(root):
    """Thống kê tổng quát cho bất kỳ thư mục nào."""
    print_header("📦 GENERAL DIRECTORY STATS")
    
    root = Path(root)
    
    # Count file types
    ext_counts = Counter()
    total_size = 0
    for f in root.rglob("*"):
        if f.is_file():
            ext_counts[f.suffix.lower()] += 1
            total_size += f.stat().st_size

    print(f"\n  📊 File types:")
    for ext, count in ext_counts.most_common(20):
        print(f"     • {ext or '(no ext)':10s} : {count:,}")

    print(f"\n  📁 Cấu trúc thư mục (depth 1):")
    for item in sorted(root.iterdir()):
        if item.is_dir():
            n_files = sum(1 for _ in item.rglob("*") if _.is_file())
            print(f"     📂 {item.name}/ ({n_files:,} files)")
        else:
            print(f"     📄 {item.name} ({fmt_size(item.stat().st_size)})")

    total_imgs = count_images(root)
    print(f"\n  🖼️  Tổng số ảnh (đệ quy): {total_imgs:,}")
    print(f"  💾 Tổng dung lượng     : {fmt_size(total_size)}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█" * 70)
    print("  🔬 UNIVERSAL COMPUTER VISION DATASET EXPLORER")
    print("█" * 70)
    print(f"\n  📁 Đường dẫn: {DATASET_ROOT}")

    if not os.path.isdir(DATASET_ROOT):
        print(f"\n  ❌ Không tìm thấy thư mục: {DATASET_ROOT}")
        print("  ➡️  Hãy điền đúng đường dẫn vào biến DATASET_ROOT ở đầu file!\n")
        return

    # Auto-detect format
    print(f"\n  🔎 Đang phát hiện loại dataset...")
    formats = detect_format(DATASET_ROOT)
    fmt_labels = {
        "coco": "COCO (JSON annotations)",
        "yolo": "YOLO (txt labels, normalized)",
        "voc": "Pascal VOC (XML annotations)",
        "classification": "Image Classification (folder-based)",
        "segmentation": "Segmentation (images + masks)",
        "unknown": "Unknown / Custom",
    }
    
    print(f"  ✅ Phát hiện: {', '.join(fmt_labels.get(f, f) for f in formats)}")

    # Explore per detected format
    for fmt in formats:
        if fmt == "coco":
            explore_coco(DATASET_ROOT)
        elif fmt == "yolo":
            explore_yolo(DATASET_ROOT)
        elif fmt == "voc":
            explore_voc(DATASET_ROOT)
        elif fmt == "classification":
            explore_classification(DATASET_ROOT)
        elif fmt == "segmentation":
            explore_segmentation(DATASET_ROOT)

    # Always show general stats
    explore_general(DATASET_ROOT)

    print(f"\n{'='*70}")
    print(f"  ✅ Hoàn tất khám phá dataset!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
