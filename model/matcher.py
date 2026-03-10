"""
Hungarian Matcher for DETR
============================
Ref: Section 2.1 of "End-to-End Object Detection with Transformers"

DETR uses a set-based loss that performs bipartite matching between predicted
and ground-truth objects. The matching uses the Hungarian algorithm to find
an optimal one-to-one assignment that minimizes the total matching cost.

The cost matrix C ∈ R^{N x M} (N predictions, M ground truths) combines:
    1. Classification cost: negative probability of the correct class
    2. L1 bounding box cost: L1 distance between predicted and GT boxes
    3. GIoU cost: 1 - GIoU between predicted and GT boxes

Ref: "We search for a permutation σ ∈ S_N with the lowest cost:
      σ_hat = argmin_σ Σ_i L_match(y_i, y_hat_σ(i))" (Eq. 1)

We use scipy.optimize.linear_sum_assignment for the Hungarian algorithm.
"""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from model.criterion import generalized_box_iou, box_cxcywh_to_xyxy


class HungarianMatcher(nn.Module):
    """
    Bipartite matching between predictions and ground truth using the
    Hungarian algorithm.

    Computes an assignment between the N predicted detections and M
    ground-truth objects. Each GT object is matched to exactly one
    prediction. The remaining (N - M) predictions are unmatched and
    will be assigned the "no-object" class during loss computation.

    Ref: "We design a loss based on finding a bipartite matching between
          ground truth and prediction." (Section 2.1)
    """

    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0,
                 cost_giou: float = 2.0):
        """
        Args:
            cost_class: Weight for the classification cost in the matching.
            cost_bbox: Weight for the L1 bounding box cost.
            cost_giou: Weight for the GIoU cost.

        Ref: "We found that a cost weighting of λ_L1 = 5 and λ_iou = 2
              works well." (Section 4)
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, \
            "At least one cost must be non-zero"

    @torch.no_grad()
    def forward(self, outputs: dict,
                targets: list) -> list:
        """
        Perform Hungarian matching for a batch.

        Args:
            outputs: Dict with:
                'pred_logits': (B, N, num_classes + 1) — raw class logits
                'pred_boxes':  (B, N, 4) — predicted boxes (cx, cy, w, h) in [0,1]

            targets: List of B dicts, each with:
                'labels': (M_i,) — class labels for ground truth objects
                'boxes':  (M_i, 4) — GT boxes (cx, cy, w, h) in [0,1]

        Returns:
            List of B tuples (pred_indices, gt_indices), where:
                pred_indices: (M_i,) — indices of matched predictions
                gt_indices:   (M_i,) — indices of matched GT objects
        """
        B, N = outputs['pred_logits'].shape[:2]

        # Flatten predictions for efficient cost computation
        # (B, N, num_classes+1) -> softmax -> (B*N, num_classes+1)
        out_prob = outputs['pred_logits'].flatten(0, 1).softmax(-1)  # (B*N, C+1)
        out_bbox = outputs['pred_boxes'].flatten(0, 1)               # (B*N, 4)

        # Concatenate all GT labels and boxes
        tgt_ids = torch.cat([t['labels'] for t in targets])   # (sum(M_i),)
        tgt_bbox = torch.cat([t['boxes'] for t in targets])    # (sum(M_i), 4)

        if tgt_ids.shape[0] == 0:
            # No ground truth objects in the entire batch
            return [(torch.as_tensor([], dtype=torch.int64),
                     torch.as_tensor([], dtype=torch.int64)) for _ in range(B)]

        # ==================== CLASSIFICATION COST ====================
        # Cost = -P(correct class) for each (prediction, GT) pair
        # Lower cost = higher probability of the correct class
        cost_class = -out_prob[:, tgt_ids]  # (B*N, sum(M_i))

        # ==================== L1 BOUNDING BOX COST ====================
        # L1 distance between predicted and GT boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  # (B*N, sum(M_i))

        # ==================== GIOU COST ====================
        # GIoU between predicted and GT boxes
        # Convert from (cx, cy, w, h) to (x1, y1, x2, y2) for GIoU computation
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox)
        )  # (B*N, sum(M_i))

        # ==================== COMBINED COST MATRIX ====================
        # Ref Eq. 1: L_match = -1_{c_i ≠ ∅} * p_hat(c_i) + L_box
        C = (self.cost_class * cost_class +
             self.cost_bbox * cost_bbox +
             self.cost_giou * cost_giou)

        # Reshape to (B, N, sum(M_i))
        C = C.view(B, N, -1).cpu()

        # ==================== HUNGARIAN MATCHING ====================
        # Solve the assignment problem independently for each image
        sizes = [len(t['labels']) for t in targets]
        indices = []

        for i, c in enumerate(C.split(sizes, dim=-1)):
            # c[i]: (N, M_i) cost matrix for image i
            cost_matrix = c[i]  # (N, M_i)

            # scipy Hungarian algorithm: finds minimum cost assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix.numpy())

            indices.append((
                torch.as_tensor(row_ind, dtype=torch.int64),
                torch.as_tensor(col_ind, dtype=torch.int64)
            ))

        return indices
