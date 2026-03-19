import torch
import torch.nn as nn
import torch.nn.functional as F


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5*w, cy - 0.5*h, cx + 0.5*w, cy + 0.5*h], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=2)
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-6)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=2)
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / (union + 1e-6)
    enclose_lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    enclose_rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enclose_area = (enclose_rb - enclose_lt).clamp(min=0).prod(dim=2)
    return iou - (enclose_area - union) / (enclose_area + 1e-6)


class SetCriterion(nn.Module):

    def __init__(self, num_classes: int, matcher,
                 weight_ce: float = 1.0, weight_bbox: float = 5.0,
                 weight_giou: float = 2.0, eos_coef: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_ce = weight_ce
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou
        self.eos_coef = eos_coef
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs: dict, targets: list, indices: list) -> torch.Tensor:
        pred_logits = outputs['pred_logits']
        B, N = pred_logits.shape[:2]
        target_classes = torch.full((B, N), self.num_classes, dtype=torch.int64, device=pred_logits.device)
        for batch_idx, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                target_classes[batch_idx, pred_idx] = targets[batch_idx]['labels'][gt_idx]
        return F.cross_entropy(pred_logits.transpose(1, 2), target_classes, weight=self.empty_weight)

    def loss_boxes(self, outputs: dict, targets: list, indices: list) -> dict:
        pred_boxes_list, gt_boxes_list = [], []
        for batch_idx, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                pred_boxes_list.append(outputs['pred_boxes'][batch_idx, pred_idx])
                gt_boxes_list.append(targets[batch_idx]['boxes'][gt_idx])

        if not pred_boxes_list:
            device = outputs['pred_boxes'].device
            return {'loss_bbox': torch.tensor(0.0, device=device),
                    'loss_giou': torch.tensor(0.0, device=device)}

        pred_boxes = torch.cat(pred_boxes_list)
        gt_boxes = torch.cat(gt_boxes_list)
        loss_bbox = F.l1_loss(pred_boxes, gt_boxes, reduction='mean')
        giou = generalized_box_iou(box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(gt_boxes))
        loss_giou = (1 - torch.diag(giou)).mean()
        return {'loss_bbox': loss_bbox, 'loss_giou': loss_giou}

    def forward(self, outputs: dict, targets: list) -> dict:
        indices = self.matcher(outputs, targets)
        loss_ce = self.loss_labels(outputs, targets, indices)
        box_losses = self.loss_boxes(outputs, targets, indices)
        loss_total = (self.weight_ce * loss_ce +
                      self.weight_bbox * box_losses['loss_bbox'] +
                      self.weight_giou * box_losses['loss_giou'])
        return {
            'loss_ce': loss_ce,
            'loss_bbox': box_losses['loss_bbox'],
            'loss_giou': box_losses['loss_giou'],
            'loss_total': loss_total,
        }
