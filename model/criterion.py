"""
DETR Set-Based Loss (SetCriterion)
====================================
Ref: Section 2.2 of "End-to-End Object Detection with Transformers"

After the Hungarian matching assigns each GT to a prediction, the loss is
computed over matched pairs plus the unmatched predictions (assigned to ∅):

    L = Σ_i [λ_cls * L_CE(class) + 1_{c_i≠∅} * (λ_L1 * L1(box) + λ_giou * L_GIoU(box))]

Components:
    1. Cross-Entropy Loss for classification
       - The "no-object" (∅) class uses a reduced weight (default=0.1) to
         account for the large imbalance (N=100 queries vs few GT objects)
       - Ref: "We down-weight the log-probability term when c_i = ∅ by a
               factor of 10." (Section 4)

    2. L1 Loss for bounding box regression
       - Applied on normalized (cx, cy, w, h) coordinates

    3. Generalized IoU (GIoU) Loss
       - Scale-invariant box similarity metric
       - Ref: "We use a linear combination of L1 loss and GIoU loss for
               bounding box regression." (Section 2.2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Box Utility Functions
# =============================================================================

def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format.

    Args:
        boxes: Tensor of shape (..., 4) in (cx, cy, w, h) format.

    Returns:
        Tensor of shape (..., 4) in (x1, y1, x2, y2) format.
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from (x1, y1, x2, y2) to (cx, cy, w, h) format.

    Args:
        boxes: Tensor of shape (..., 4) in (x1, y1, x2, y2) format.

    Returns:
        Tensor of shape (..., 4) in (cx, cy, w, h) format.
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes in (x1, y1, x2, y2) format.

    Args:
        boxes1: (N, 4)
        boxes2: (M, 4)

    Returns:
        iou: (N, M) pairwise IoU matrix.
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (N, M, 2)
    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    union = area1[:, None] + area2[None, :] - inter
    iou = inter / (union + 1e-6)

    return iou


def generalized_box_iou(boxes1: torch.Tensor,
                        boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute Generalized IoU (GIoU) between two sets of boxes.

    GIoU = IoU - (|C \ (A ∪ B)| / |C|)
    where C is the smallest enclosing box of A and B.

    Ref: Rezatofighi et al., "Generalized Intersection over Union", CVPR 2019

    Args:
        boxes1: (N, 4) in (x1, y1, x2, y2) format
        boxes2: (M, 4) in (x1, y1, x2, y2) format

    Returns:
        giou: (N, M) pairwise GIoU matrix in [-1, 1].
    """
    # Standard IoU
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2[None, :] - inter
    iou = inter / (union + 1e-6)

    # Enclosing box
    enclose_lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    enclose_rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)
    enclose_area = enclose_wh[:, :, 0] * enclose_wh[:, :, 1]

    giou = iou - (enclose_area - union) / (enclose_area + 1e-6)

    return giou


# =============================================================================
# SetCriterion
# =============================================================================

class SetCriterion(nn.Module):
    """
    DETR loss function combining classification, L1, and GIoU losses.

    The loss is computed based on the Hungarian matching output:
        - Matched predictions: CE loss + L1 loss + GIoU loss
        - Unmatched predictions: CE loss with target = ∅ (no-object)

    Ref: "The matching cost and the loss are computed efficiently with
          the Hungarian algorithm." (Section 2)
    """

    def __init__(self, num_classes: int, matcher,
                 weight_ce: float = 1.0, weight_bbox: float = 5.0,
                 weight_giou: float = 2.0, eos_coef: float = 0.1):
        """
        Args:
            num_classes: Number of object classes (excluding no-object).
            matcher: HungarianMatcher instance for bipartite matching.
            weight_ce: Weight for the CE classification loss.
            weight_bbox: Weight for the L1 bounding box loss.
                         Ref: "λ_L1 = 5" (Section 4)
            weight_giou: Weight for the GIoU loss.
                         Ref: "λ_iou = 2" (Section 4)
            eos_coef: Weight for the no-object (∅) class in CE loss.
                      Ref: "Down-weighted by a factor of 10 → 0.1" (Section 4)
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_ce = weight_ce
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou
        self.eos_coef = eos_coef

        # Create class weight tensor: all classes = 1.0, no-object = eos_coef
        # The no-object class is at index `num_classes`
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef  # Down-weight the ∅ class
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs: dict, targets: list,
                    indices: list) -> torch.Tensor:
        """
        Classification loss (Cross-Entropy).

        For matched predictions, the target is the GT class label.
        For unmatched predictions, the target is the no-object class.

        Args:
            outputs: Model outputs with 'pred_logits' (B, N, C+1).
            targets: List of B target dicts with 'labels'.
            indices: Hungarian matching indices.

        Returns:
            Scalar CE loss.
        """
        pred_logits = outputs['pred_logits']  # (B, N, C+1)
        B, N = pred_logits.shape[:2]
        device = pred_logits.device

        # Initialize all targets as no-object (index = num_classes)
        target_classes = torch.full(
            (B, N), self.num_classes,
            dtype=torch.int64, device=device
        )

        # Fill in matched targets
        for batch_idx, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                target_classes[batch_idx, pred_idx] = \
                    targets[batch_idx]['labels'][gt_idx]

        # Cross-Entropy loss with per-class weighting
        loss_ce = F.cross_entropy(
            pred_logits.transpose(1, 2),  # (B, C+1, N) for CE
            target_classes,                # (B, N)
            weight=self.empty_weight
        )

        return loss_ce

    def loss_boxes(self, outputs: dict, targets: list,
                   indices: list) -> dict:
        """
        Bounding box losses (L1 + GIoU) on matched predictions only.

        Ref: "We use a linear combination of the L1 loss and the generalized
              IoU loss for bounding box regression." (Section 2.2)

        Args:
            outputs: Model outputs with 'pred_boxes' (B, N, 4).
            targets: List of B target dicts with 'boxes'.
            indices: Hungarian matching indices.

        Returns:
            Dict with 'loss_bbox' (L1) and 'loss_giou'.
        """
        # Gather matched predictions and GT boxes
        pred_boxes_list = []
        gt_boxes_list = []

        for batch_idx, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                pred_boxes_list.append(outputs['pred_boxes'][batch_idx, pred_idx])
                gt_boxes_list.append(targets[batch_idx]['boxes'][gt_idx])

        if len(pred_boxes_list) == 0:
            # No matched pairs in the batch
            device = outputs['pred_boxes'].device
            return {
                'loss_bbox': torch.tensor(0.0, device=device),
                'loss_giou': torch.tensor(0.0, device=device),
            }

        pred_boxes = torch.cat(pred_boxes_list, dim=0)  # (total_matched, 4)
        gt_boxes = torch.cat(gt_boxes_list, dim=0)      # (total_matched, 4)

        # L1 loss on (cx, cy, w, h)
        loss_bbox = F.l1_loss(pred_boxes, gt_boxes, reduction='mean')

        # GIoU loss
        # Convert to (x1, y1, x2, y2) for GIoU computation
        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        gt_xyxy = box_cxcywh_to_xyxy(gt_boxes)

        # Diagonal GIoU: compute GIoU for each matched pair, not full pairwise
        giou = generalized_box_iou(pred_xyxy, gt_xyxy)
        # Extract diagonal (matched pairs)
        loss_giou = 1 - torch.diag(giou)
        loss_giou = loss_giou.mean()

        return {
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou,
        }

    def forward(self, outputs: dict, targets: list) -> dict:
        """
        Compute the total DETR loss.

        Steps:
            1. Run Hungarian matching to find optimal assignment
            2. Compute classification loss (CE) for all predictions
            3. Compute box losses (L1 + GIoU) for matched predictions only
            4. Combine with loss weights

        Args:
            outputs: Dict with 'pred_logits' (B, N, C+1) and 'pred_boxes' (B, N, 4).
            targets: List of B dicts, each with 'labels' (M_i,) and 'boxes' (M_i, 4).

        Returns:
            Dict with individual and total losses:
                'loss_ce', 'loss_bbox', 'loss_giou', 'loss_total'
        """
        # Step 1: Hungarian matching
        indices = self.matcher(outputs, targets)

        # Step 2: Classification loss
        loss_ce = self.loss_labels(outputs, targets, indices)

        # Step 3: Box losses
        box_losses = self.loss_boxes(outputs, targets, indices)

        # Step 4: Total loss
        loss_total = (
            self.weight_ce * loss_ce +
            self.weight_bbox * box_losses['loss_bbox'] +
            self.weight_giou * box_losses['loss_giou']
        )

        return {
            'loss_ce': loss_ce,
            'loss_bbox': box_losses['loss_bbox'],
            'loss_giou': box_losses['loss_giou'],
            'loss_total': loss_total,
        }
