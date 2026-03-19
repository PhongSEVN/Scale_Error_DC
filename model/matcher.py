import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from model.criterion import generalized_box_iou, box_cxcywh_to_xyxy


class HungarianMatcher(nn.Module):

    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0

    @torch.no_grad()
    def forward(self, outputs: dict, targets: list) -> list:
        B, N = outputs['pred_logits'].shape[:2]

        out_prob = outputs['pred_logits'].flatten(0, 1).softmax(-1)
        out_bbox = outputs['pred_boxes'].flatten(0, 1)
        tgt_ids = torch.cat([t['labels'] for t in targets])
        tgt_bbox = torch.cat([t['boxes'] for t in targets])

        if tgt_ids.shape[0] == 0:
            return [(torch.as_tensor([], dtype=torch.int64),
                     torch.as_tensor([], dtype=torch.int64)) for _ in range(B)]

        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        C = (self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou)
        C = C.view(B, N, -1).cpu()

        sizes = [len(t['labels']) for t in targets]
        indices = []
        for i, c in enumerate(C.split(sizes, dim=-1)):
            row_ind, col_ind = linear_sum_assignment(c[i].numpy())
            indices.append((
                torch.as_tensor(row_ind, dtype=torch.int64),
                torch.as_tensor(col_ind, dtype=torch.int64)
            ))

        return indices
