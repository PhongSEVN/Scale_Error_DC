import os
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import config and model components
import configs.train_config as cfg
from model.detr import build_detr
from model.criterion import box_cxcywh_to_xyxy
from dataset.coco_dataset import COCODETRDataset, coco_detr_collate_fn

def compute_iou_np(box1: np.ndarray, box2: np.ndarray) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-6)

def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def evaluate_detections(all_predictions: list, all_targets: list, num_classes: int, iou_threshold: float = 0.5) -> dict:
    class_detections = defaultdict(list)
    class_num_gt = defaultdict(int)

    for img_idx, (preds, gts) in enumerate(zip(all_predictions, all_targets)):
        gt_boxes = gts['boxes']
        gt_labels = gts['labels']
        pred_boxes = preds['boxes']
        pred_scores = preds['scores']
        pred_labels = preds['labels']

        for label in gt_labels:
            class_num_gt[label] += 1

        gt_matched = np.zeros(len(gt_labels), dtype=bool)

        if len(pred_scores) > 0:
            sort_idx = np.argsort(-pred_scores)
            pred_boxes = pred_boxes[sort_idx]
            pred_scores = pred_scores[sort_idx]
            pred_labels = pred_labels[sort_idx]

        for i in range(len(pred_labels)):
            pred_cls = pred_labels[i]
            pred_box = pred_boxes[i]
            score = pred_scores[i]

            best_iou = 0.0
            best_gt_idx = -1

            for j in range(len(gt_labels)):
                if gt_labels[j] != pred_cls or gt_matched[j]:
                    continue
                iou = compute_iou_np(pred_box, gt_boxes[j])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                gt_matched[best_gt_idx] = True
                class_detections[pred_cls].append((score, True))
            else:
                class_detections[pred_cls].append((score, False))

    ap_per_class = {}
    precision_per_class = {}
    recall_per_class = {}

    for cls_id in range(num_classes):
        detections = class_detections.get(cls_id, [])
        num_gt = class_num_gt.get(cls_id, 0)

        if num_gt == 0 or len(detections) == 0:
            ap_per_class[cls_id] = 0.0
            precision_per_class[cls_id] = 0.0
            recall_per_class[cls_id] = 0.0
            continue

        detections.sort(key=lambda x: -x[0])
        scores = np.array([d[0] for d in detections])
        is_tp = np.array([d[1] for d in detections])

        tp_cumsum = np.cumsum(is_tp)
        fp_cumsum = np.cumsum(~is_tp)

        recall = tp_cumsum / num_gt
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)

        ap_per_class[cls_id] = compute_ap(recall, precision)
        precision_per_class[cls_id] = precision[-1]
        recall_per_class[cls_id] = recall[-1]

    valid_aps = [ap_per_class[c] for c in range(num_classes) if class_num_gt.get(c, 0) > 0]
    mAP = np.mean(valid_aps) if len(valid_aps) > 0 else 0.0

    return {
        'mAP': mAP,
        'AP': ap_per_class,
        'precision': precision_per_class,
        'recall': recall_per_class,
        'num_gt_per_class': dict(class_num_gt),
    }

@torch.no_grad()
def run_evaluation(model, dataloader, device, confidence_threshold=0.01):
    model.eval()
    all_predictions = []
    all_targets = []

    pbar = tqdm(dataloader, desc="Evaluating", unit="batch", ncols=100)

    for images, masks, targets in pbar:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images, mask=masks)
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']

        B = pred_logits.shape[0]
        probs = pred_logits.softmax(-1)

        for b in range(B):
            scores, labels = probs[b, :, :-1].max(-1)
            keep = scores > confidence_threshold
            
            pred_scores = scores[keep].cpu().numpy()
            pred_labels = labels[keep].cpu().numpy()
            pred_bboxes = pred_boxes[b, keep].cpu()

            if len(pred_bboxes) > 0:
                pred_bboxes = box_cxcywh_to_xyxy(pred_bboxes).numpy()
            else:
                pred_bboxes = np.zeros((0, 4))

            all_predictions.append({
                'boxes': pred_bboxes,
                'scores': pred_scores,
                'labels': pred_labels,
            })

            gt_boxes = targets[b]['boxes']
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
    parser = argparse.ArgumentParser(description='DETR Evaluation')
    parser.add_argument('--checkpoint', type=str, default=os.path.join(cfg.CHECKPOINT_DIR, 'best_model.pth'))
    parser.add_argument('--split', type=str, default='val', choices=['val', 'train'])
    parser.add_argument('--threshold', type=float, default=0.01)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model = build_detr(
        num_classes=cfg.NUM_CLASSES,
        num_queries=cfg.NUM_QUERIES,
        hidden_dim=cfg.HIDDEN_DIM,
        nhead=cfg.NHEAD,
        num_encoder_layers=cfg.NUM_ENCODER_LAYERS,
        num_decoder_layers=cfg.NUM_DECODER_LAYERS,
        dim_feedforward=cfg.DIM_FEEDFORWARD,
        dropout=cfg.DROPOUT,
        backbone_name=cfg.BACKBONE_NAME
    )
    
    # Load state dict (check if it's the full checkpoint or just model weights)
    state_dict = checkpoint if 'model_state_dict' not in checkpoint else checkpoint['model_state_dict']
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    root = cfg.VAL_ROOT if args.split == 'val' else cfg.TRAIN_ROOT
    ann = cfg.VAL_ANN if args.split == 'val' else cfg.TRAIN_ANN

    dataset = COCODETRDataset(root=root, ann_file=ann, class_names=cfg.CLASS_NAMES, augment=False, img_size=cfg.IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg.NUM_WORKERS, collate_fn=coco_detr_collate_fn, pin_memory=True)

    print(f"Evaluating on {args.split} split...")
    all_preds, all_gts = run_evaluation(model, dataloader, device, confidence_threshold=args.threshold)

    results_50 = evaluate_detections(all_preds, all_gts, cfg.NUM_CLASSES, iou_threshold=0.5)

    print(f"\n{'='*60}")
    print(f" mAP@0.5 = {results_50['mAP']*100:.2f}%")
    print(f"{'='*60}")
    
    for cls_id in range(cfg.NUM_CLASSES):
        name = cfg.CLASS_NAMES[cls_id]
        ap = results_50['AP'].get(cls_id, 0.0)
        ngt = results_50['num_gt_per_class'].get(cls_id, 0)
        print(f"  {name:<20}: {ap*100:>7.2f}% (GT: {ngt})")

    # COCO mAP
    iou_thrs = np.arange(0.5, 1.0, 0.05)
    maps = [evaluate_detections(all_preds, all_gts, cfg.NUM_CLASSES, iou_threshold=thr)['mAP'] for thr in iou_thrs]
    print(f"\n mAP@0.5:0.95 (COCO) = {np.mean(maps)*100:.2f}%")

if __name__ == '__main__':
    main()
