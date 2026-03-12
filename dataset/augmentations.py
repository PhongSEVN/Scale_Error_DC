import numpy as np
import torch
import random
from PIL import Image
import torchvision.transforms.functional as TF

def mixup_detection(img1, target1, img2, target2, alpha=0.4):
    """
    MixUp for Object Detection.
    - img = lam * img1 + (1-lam) * img2
    - targets = concat(target1, target2)
    """
    lam = np.random.beta(alpha, alpha)
    
    # Resize img2 to matches img1
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.BILINEAR)
    
    arr1 = np.array(img1).astype(np.float32)
    arr2 = np.array(img2).astype(np.float32)
    
    mixed_img_arr = lam * arr1 + (1 - lam) * arr2
    mixed_img = Image.fromarray(mixed_img_arr.astype(np.uint8))
    
    # Concatenate targets
    mixed_target = {
        'labels': torch.cat([target1['labels'], target2['labels']]),
        'boxes': torch.cat([target1['boxes'], target2['boxes']]),
    }
    
    return mixed_img, mixed_target

def cutmix_detection(img1, target1, img2, target2, beta=1.0):
    """
    CutMix for Object Detection.
    - img = img1 with a patch from img2
    - targets = boxes from img1 (outside patch) + boxes from img2 (inside patch)
    """
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.BILINEAR)
        
    arr1 = np.array(img1).copy()
    arr2 = np.array(img2)
    
    w, h = img1.size
    lam = np.random.beta(beta, beta)
    cut_ratio = np.sqrt(1.0 - lam)
    
    cut_w = int(w * cut_ratio)
    cut_h = int(h * cut_ratio)
    
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    
    x1 = np.clip(cx - cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    
    # Paste patch
    arr1[y1:y2, x1:x2] = arr2[y1:y2, x1:x2]
    mixed_img = Image.fromarray(arr1)
    
    # Filter/Clip boxes
    # Normalize coordinates to 0-1
    nx1, ny1, nx2, ny2 = x1/w, y1/h, x2/w, y2/h
    
    # boxes are [cx, cy, w, h] normalized
    # target1 boxes: keep if center is outside cutmix box (simplified)
    # target2 boxes: keep if center is inside cutmix box (simplified)
    
    def is_inside(box, x1, y1, x2, y2):
        bcx, bcy = box[0], box[1]
        return (bcx >= x1) and (bcx <= x2) and (bcy >= y1) and (bcy <= y2)

    boxes1 = target1['boxes']
    labels1 = target1['labels']
    mask1 = torch.tensor([not is_inside(b, nx1, ny1, nx2, ny2) for b in boxes1], dtype=torch.bool)
    
    boxes2 = target2['boxes']
    labels2 = target2['labels']
    mask2 = torch.tensor([is_inside(b, nx1, ny1, nx2, ny2) for b in boxes2], dtype=torch.bool)
    
    mixed_target = {
        'labels': torch.cat([labels1[mask1], labels2[mask2]]),
        'boxes': torch.cat([boxes1[mask1], boxes2[mask2]]),
    }
    
    return mixed_img, mixed_target

def fmix_detection(img1, target1, img2, target2, alpha=1.0, decay_power=3.0):
    """
    FMix for Object Detection.
    - img = img1 * mask + img2 * (1 - mask)
    - targets = boxes from img1 (where mask=1) + boxes from img2 (where mask=0)
    """
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.BILINEAR)
        
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    w, h = img1.size
    
    # FMix Mask generation
    lam = np.random.beta(alpha, alpha)
    freqs = np.fft.fftfreq(h)[:, None] ** 2 + np.fft.fftfreq(w)[None, :] ** 2
    freqs = np.sqrt(freqs)
    spectrum = np.random.randn(h, w) * (1.0 / (freqs + 1e-6) ** decay_power)
    mask = np.fft.ifft2(spectrum).real
    mask = np.abs(mask)
    mask = (mask > np.percentile(mask, (1 - lam) * 100)).astype(np.float32)
    
    mixed_img_arr = arr1 * mask[..., None] + arr2 * (1 - mask[..., None])
    mixed_img = Image.fromarray(mixed_img_arr.astype(np.uint8))
    
    # Filter boxes based on centers
    def is_in_mask(box, mask_arr, value=1.0):
        # bcx, bcy are normalized
        bcx, bcy = box[0], box[1]
        px = int(bcx * (mask_arr.shape[1] - 1))
        py = int(bcy * (mask_arr.shape[0] - 1))
        return mask_arr[py, px] == value

    boxes1 = target1['boxes']
    labels1 = target1['labels']
    mask1 = torch.tensor([is_in_mask(b, mask, 1.0) for b in boxes1], dtype=torch.bool)
    
    boxes2 = target2['boxes']
    labels2 = target2['labels']
    mask2 = torch.tensor([is_in_mask(b, mask, 0.0) for b in boxes2], dtype=torch.bool)
    
    mixed_target = {
        'labels': torch.cat([labels1[mask1], labels2[mask2]]),
        'boxes': torch.cat([boxes1[mask1], boxes2[mask2]]),
    }
    
    return mixed_img, mixed_target


def copy_paste_minority(img1, target1, img2, target2):
    """
    Copy objects from img2 (minority) and paste onto img1.
    """
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.BILINEAR)
        
    arr1 = np.array(img1).copy()
    arr2 = np.array(img2)
    w, h = img1.size
    
    new_labels = []
    new_boxes = []
    
    for i, label in enumerate(target2['labels']):
        if label.item() in [4, 5]: # Rare classes
            box = target2['boxes'][i]
            # Convert normalized cx, cy, w, h to pixel x1, y1, x2, y2
            cx, cy, bw, bh = box[0]*w, box[1]*h, box[2]*w, box[3]*h
            x1, y1 = int(cx - bw/2), int(cy - bh/2)
            x2, y2 = int(cx + bw/2), int(cy + bh/2)
            
            # Clip
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            
            if (x2-x1) > 5 and (y2-y1) > 5: # At least 5x5 pixels
                # Attempt to paste at a random location
                # We try a few times to find a position that fits
                p_w, p_h = x2-x1, y2-y1
                target_x = random.randint(0, w - p_w)
                target_y = random.randint(0, h - p_h)
                
                arr1[target_y:target_y+p_h, target_x:target_x+p_w] = arr2[y1:y2, x1:x2]
                
                new_labels.append(label.item())
                new_boxes.append([(target_x + p_w/2)/w, (target_y + p_h/2)/h, p_w/w, p_h/h])

    if len(new_labels) == 0:
        return img1, target1

    mixed_target = {
        'labels': torch.cat([target1['labels'], torch.tensor(new_labels, dtype=torch.long)]),
        'boxes': torch.cat([target1['boxes'], torch.tensor(new_boxes, dtype=torch.float32)]),
    }
    
    return Image.fromarray(arr1), mixed_target
