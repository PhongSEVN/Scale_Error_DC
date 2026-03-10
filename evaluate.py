"""
DETR Evaluation Script — mAP Computation
===========================================
Computes standard object detection metrics:
    - mAP@0.5 (Pascal VOC-style)
    - mAP@0.5:0.95 (COCO-style, averaged over IoU thresholds)
    - Per-class AP

Usage:
    python evaluate.py
    python evaluate.py --checkpoint checkpoints/best_model.pth
    python evaluate.py --split val
"""

import os
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.train_config import *
from model.detr import build_detr
from model.criterion import box_cxcywh_to_xyxy
from dataset.dataset import DETRDataset, detr_collate_fn


# =============================================================================
# IoU Computation (numpy, for evaluation)
# =============================================================================

def compute_iou_np(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two boxes in (x1, y1, x2, y2) format.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / (union + 1e-6)


# =============================================================================
# AP Computation
# =============================================================================

def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Compute Average Precision (AP) using the 11-point interpolation method
    or the all-point method (COCO-style).

    Uses the all-point interpolation (area under PR curve).
    """
    # Prepend/append sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Make precision monotonically decreasing (from right to left)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Find points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Sum up the rectangular areas under the PR curve
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def evaluate_detections(all_predictions: list, all_targets: list,
                        num_classes: int, iou_threshold: float = 0.5) -> dict:
    """
    Compute per-class AP and mAP at a given IoU threshold.

    Args:
        all_predictions: List of dicts per image, each with:
            'boxes': (K, 4) numpy array in (x1, y1, x2, y2) normalized
            'scores': (K,) numpy array of confidence scores
            'labels': (K,) numpy array of predicted class indices
        all_targets: List of dicts per image, each with:
            'boxes': (M, 4) numpy array in (x1, y1, x2, y2) normalized
            'labels': (M,) numpy array of GT class indices
        num_classes: Number of object classes.
        iou_threshold: IoU threshold for a detection to be considered correct.

    Returns:
        Dict with 'mAP', 'AP' (per-class), 'precision', 'recall'.
    """
    # Collect all detections and ground truths per class
    class_detections = defaultdict(list)   # class_id -> [(score, is_tp, image_idx)]
    class_num_gt = defaultdict(int)        # class_id -> total GT count

    for img_idx, (preds, gts) in enumerate(zip(all_predictions, all_targets)):
        gt_boxes = gts['boxes']
        gt_labels = gts['labels']
        pred_boxes = preds['boxes']
        pred_scores = preds['scores']
        pred_labels = preds['labels']

        # Count GT objects per class
        for label in gt_labels:
            class_num_gt[label] += 1

        # Track which GT boxes have been matched (per image)
        gt_matched = np.zeros(len(gt_labels), dtype=bool)

        # Sort predictions by confidence (descending)
        if len(pred_scores) > 0:
            sort_idx = np.argsort(-pred_scores)
            pred_boxes = pred_boxes[sort_idx]
            pred_scores = pred_scores[sort_idx]
            pred_labels = pred_labels[sort_idx]

        # Match each prediction to a GT
        for i in range(len(pred_labels)):
            pred_cls = pred_labels[i]
            pred_box = pred_boxes[i]
            score = pred_scores[i]

            best_iou = 0.0
            best_gt_idx = -1

            # Find best matching GT of the same class
            for j in range(len(gt_labels)):
                if gt_labels[j] != pred_cls:
                    continue
                if gt_matched[j]:
                    continue

                iou = compute_iou_np(pred_box, gt_boxes[j])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                # True Positive
                gt_matched[best_gt_idx] = True
                class_detections[pred_cls].append((score, True))
            else:
                # False Positive
                class_detections[pred_cls].append((score, False))

    # Compute AP per class
    ap_per_class = {}
    precision_per_class = {}
    recall_per_class = {}

    for cls_id in range(num_classes):
        detections = class_detections.get(cls_id, [])
        num_gt = class_num_gt.get(cls_id, 0)

        if num_gt == 0:
            # No GT objects for this class
            ap_per_class[cls_id] = 0.0
            precision_per_class[cls_id] = 0.0
            recall_per_class[cls_id] = 0.0
            continue

        if len(detections) == 0:
            ap_per_class[cls_id] = 0.0
            precision_per_class[cls_id] = 0.0
            recall_per_class[cls_id] = 0.0
            continue

        # Sort by confidence (descending)
        detections.sort(key=lambda x: -x[0])
        scores = np.array([d[0] for d in detections])
        is_tp = np.array([d[1] for d in detections])

        # Cumulative TP and FP
        tp_cumsum = np.cumsum(is_tp)
        fp_cumsum = np.cumsum(~is_tp)

        recall = tp_cumsum / num_gt
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)

        ap = compute_ap(recall, precision)
        ap_per_class[cls_id] = ap
        precision_per_class[cls_id] = precision[-1] if len(precision) > 0 else 0.0
        recall_per_class[cls_id] = recall[-1] if len(recall) > 0 else 0.0

    # mAP = mean of per-class APs (only over classes with GT objects)
    valid_aps = [ap_per_class[c] for c in range(num_classes) if class_num_gt.get(c, 0) > 0]
    mAP = np.mean(valid_aps) if len(valid_aps) > 0 else 0.0

    return {
        'mAP': mAP,
        'AP': ap_per_class,
        'precision': precision_per_class,
        'recall': recall_per_class,
        'num_gt_per_class': dict(class_num_gt),
    }


# =============================================================================
# Main Evaluation
# =============================================================================

@torch.no_grad()
def run_evaluation(model, dataloader, device, num_classes,
                   confidence_threshold=0.01):
    """
    Run model on the dataset and collect all predictions + ground truths.

    Args:
        model: DETR model in eval mode.
        dataloader: DataLoader for the evaluation dataset.
        device: Computation device.
        num_classes: Number of object classes.
        confidence_threshold: Minimum score to keep (low for mAP computation).

    Returns:
        all_predictions, all_targets: Lists of dicts per image.
    """
    model.eval()
    all_predictions = []
    all_targets = []

    pbar = tqdm(dataloader, desc="Evaluating", unit="batch", ncols=100)

    for images, masks, targets in pbar:
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = model(images, mask=masks)
        pred_logits = outputs['pred_logits']  # (B, N, C+1)
        pred_boxes = outputs['pred_boxes']    # (B, N, 4)

        # Process each image in the batch
        B = pred_logits.shape[0]
        probs = pred_logits.softmax(-1)  # (B, N, C+1)

        for b in range(B):
            # Get best class score (excluding no-object = last class)
            scores, labels = probs[b, :, :-1].max(-1)  # (N,), (N,)

            # Filter by confidence
            keep = scores > confidence_threshold
            pred_scores = scores[keep].cpu().numpy()
            pred_labels = labels[keep].cpu().numpy()
            pred_bboxes = pred_boxes[b, keep].cpu()  # (K, 4) cxcywh

            # Convert to xyxy
            if len(pred_bboxes) > 0:
                pred_bboxes = box_cxcywh_to_xyxy(pred_bboxes).numpy()
            else:
                pred_bboxes = np.zeros((0, 4))

            all_predictions.append({
                'boxes': pred_bboxes,
                'scores': pred_scores,
                'labels': pred_labels,
            })

            # Ground truth
            gt_boxes = targets[b]['boxes']  # (M, 4) cxcywh
            gt_labels = targets[b]['labels'].numpy()

            if len(gt_boxes) > 0:
                gt_boxes = box_cxcywh_to_xyxy(gt_boxes).numpy()
            else:
                gt_boxes = np.zeros((0, 4))

            all_targets.append({
                'boxes': gt_boxes,
                'labels': gt_labels,
            })

    return all_predictions, all_targets


def main():
    parser = argparse.ArgumentParser(description='DETR Evaluation — mAP')
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(CHECKPOINT_DIR, 'best_model.pth'),
                        help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='val',
                        choices=['val', 'test', 'train'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='Confidence threshold (low for mAP computation)')
    parser.add_argument('--img_size', type=int, default=IMAGE_SIZE,
                        help='Input image size')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- Load model ----
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Train the model first: python train.py")
        sys.exit(1)

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    num_classes = checkpoint.get('num_classes', NUM_CLASSES)
    class_names = checkpoint.get('class_names', CLASS_NAMES)

    model = build_detr(
        num_classes=num_classes,
        num_queries=checkpoint.get('num_queries', NUM_QUERIES),
        hidden_dim=checkpoint.get('hidden_dim', HIDDEN_DIM),
        pretrained_backbone=False,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"  Loaded (epoch {checkpoint.get('epoch', '?')}, "
          f"loss {checkpoint.get('loss', 0):.4f})")

    # ---- Load dataset ----
    split_dirs = {'val': VAL_DIR, 'test': TEST_DIR, 'train': TRAIN_DIR}
    data_dir = split_dirs[args.split]

    if not os.path.exists(os.path.join(data_dir, 'images')):
        print(f"Error: No images found at {data_dir}/images")
        sys.exit(1)

    dataset = DETRDataset(
        root=data_dir,
        img_size=args.img_size,
        class_names=class_names,
        augment=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=detr_collate_fn,
        pin_memory=True,
    )

    # ---- Run evaluation ----
    print(f"\nEvaluating on {args.split} set ({len(dataset)} images)...")
    all_preds, all_gts = run_evaluation(
        model, dataloader, device, num_classes,
        confidence_threshold=args.threshold
    )

    # ---- Compute mAP at different IoU thresholds ----
    print(f"\n{'='*70}")
    print(f"  DETR Evaluation Results — {args.split} set")
    print(f"{'='*70}")

    # mAP@0.5 (Pascal VOC-style)
    results_50 = evaluate_detections(all_preds, all_gts, num_classes, iou_threshold=0.5)

    print(f"\n  mAP@0.5 = {results_50['mAP']*100:.2f}%")
    print(f"\n  Per-class AP@0.5:")
    print(f"  {'Class':<20} {'AP':>8} {'Precision':>10} {'Recall':>8} {'#GT':>6}")
    print(f"  {'-'*54}")
    for cls_id in range(num_classes):
        name = class_names[cls_id] if cls_id < len(class_names) else f'class_{cls_id}'
        ap = results_50['AP'].get(cls_id, 0.0)
        prec = results_50['precision'].get(cls_id, 0.0)
        rec = results_50['recall'].get(cls_id, 0.0)
        ngt = results_50['num_gt_per_class'].get(cls_id, 0)
        print(f"  {name:<20} {ap*100:>7.2f}% {prec*100:>9.2f}% {rec*100:>7.2f}% {ngt:>6}")

    # mAP@0.5:0.95 (COCO-style — averaged over 10 IoU thresholds)
    iou_thresholds = np.arange(0.5, 1.0, 0.05)  # [0.50, 0.55, ..., 0.95]
    maps = []
    for iou_thr in iou_thresholds:
        res = evaluate_detections(all_preds, all_gts, num_classes, iou_threshold=iou_thr)
        maps.append(res['mAP'])

    mAP_coco = np.mean(maps)

    print(f"\n  mAP@0.5:0.95 (COCO) = {mAP_coco*100:.2f}%")
    print(f"\n  AP at each IoU threshold:")
    for iou_thr, m in zip(iou_thresholds, maps):
        bar = '█' * int(m * 40)
        print(f"    IoU={iou_thr:.2f}: {m*100:>6.2f}% {bar}")

    # ---- Summary statistics ----
    total_preds = sum(len(p['scores']) for p in all_preds)
    total_gt = sum(len(g['labels']) for g in all_gts)
    avg_preds_per_img = total_preds / max(len(all_preds), 1)

    print(f"\n  {'─'*54}")
    print(f"  Total GT objects:        {total_gt}")
    print(f"  Total predictions:       {total_preds}")
    print(f"  Avg predictions/image:   {avg_preds_per_img:.1f}")
    print(f"{'='*70}\n")

    # ---- Verdict ----
    print("📊 Model Quality Guide:")
    if mAP_coco >= 0.40:
        print("  ✅ EXCELLENT — mAP@0.5:0.95 ≥ 40% (DETR paper level on COCO)")
    elif mAP_coco >= 0.25:
        print("  ✅ GOOD — mAP@0.5:0.95 ≥ 25% (reasonable for custom dataset)")
    elif mAP_coco >= 0.10:
        print("  ⚠️  FAIR — mAP@0.5:0.95 ≥ 10% (model is learning, needs more training)")
    else:
        print("  ❌ POOR — mAP@0.5:0.95 < 10% (model needs more epochs or tuning)")

    if results_50['mAP'] >= 0.50:
        print("  ✅ mAP@0.5 ≥ 50% — Good detection at IoU=0.5")
    elif results_50['mAP'] >= 0.25:
        print("  ⚠️  mAP@0.5 ≥ 25% — Acceptable, but can improve")
    else:
        print("  ❌ mAP@0.5 < 25% — Poor detection, keep training")


if __name__ == '__main__':
    main()
