"""
DETR Inference / Testing Script
=================================
Visualizes DETR predictions on images with bounding boxes and class labels.

Usage:
    python test.py                          # Run on test set
    python test.py --image path/to/img.jpg  # Run on single image
    python test.py --checkpoint path/to/model.pth
"""

import os
import sys
import argparse
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.train_config import *
from model.detr import build_detr
from model.criterion import box_cxcywh_to_xyxy


# Color palette for detection visualization (vibrant, distinguishable colors)
COLORS = [
    (230, 25, 75),    # Red
    (60, 180, 75),    # Green
    (255, 225, 25),   # Yellow
    (0, 130, 200),    # Blue
    (245, 130, 48),   # Orange
    (145, 30, 180),   # Purple
    (70, 240, 240),   # Cyan
    (240, 50, 230),   # Magenta
    (210, 245, 60),   # Lime
    (250, 190, 212),  # Pink
]


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load a trained DETR model from checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint file.
        device: Device to load model onto.

    Returns:
        Loaded DETR model in eval mode.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract model config from checkpoint if available
    num_classes = checkpoint.get('num_classes', NUM_CLASSES)
    num_queries = checkpoint.get('num_queries', NUM_QUERIES)
    hidden_dim = checkpoint.get('hidden_dim', HIDDEN_DIM)

    model = build_detr(
        num_classes=num_classes,
        num_queries=num_queries,
        hidden_dim=hidden_dim,
        pretrained_backbone=False,  # We load weights from checkpoint
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"  Model loaded (epoch {checkpoint.get('epoch', '?')}, "
          f"loss {checkpoint.get('loss', '?'):.4f})")

    return model


def preprocess_image(img_path: str, img_size: int = 800):
    """
    Preprocess a single image for DETR inference.

    Args:
        img_path: Path to the image file.
        img_size: Target shortest side.

    Returns:
        img_tensor: Preprocessed image tensor, shape (1, 3, H, W).
        orig_img: Original PIL image (for visualization).
        orig_size: (orig_h, orig_w) tuple.
    """
    orig_img = Image.open(img_path).convert('RGB')
    orig_w, orig_h = orig_img.size

    # Resize keeping aspect ratio
    scale = img_size / min(orig_w, orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    img = TF.resize(orig_img, [new_h, new_w])

    # To tensor and normalize
    img_tensor = TF.to_tensor(img)
    img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])

    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)  # (1, 3, H, W)

    return img_tensor, orig_img, (orig_h, orig_w)


@torch.no_grad()
def predict(model, img_tensor: torch.Tensor, device: torch.device,
            confidence_threshold: float = 0.7):
    """
    Run DETR inference on a single image.

    Args:
        model: DETR model in eval mode.
        img_tensor: Preprocessed image, shape (1, 3, H, W).
        device: Computation device.
        confidence_threshold: Minimum confidence to keep a detection.

    Returns:
        List of dicts with 'label', 'confidence', 'box' (x1, y1, x2, y2 absolute).
    """
    img_tensor = img_tensor.to(device)

    # Forward pass
    outputs = model(img_tensor)

    # Get predictions
    pred_logits = outputs['pred_logits']  # (1, N, C+1)
    pred_boxes = outputs['pred_boxes']    # (1, N, 4)

    # Convert logits to probabilities
    probs = pred_logits.softmax(-1)[0]  # (N, C+1)

    # Get best class (excluding no-object which is last class)
    # max over all real classes (not no-object)
    scores, labels = probs[:, :-1].max(-1)  # (N,), (N,)

    # Filter by confidence
    keep = scores > confidence_threshold
    scores = scores[keep]
    labels = labels[keep]
    boxes = pred_boxes[0, keep]  # (K, 4) in (cx, cy, w, h) normalized

    # Convert to (x1, y1, x2, y2) normalized
    boxes_xyxy = box_cxcywh_to_xyxy(boxes)  # (K, 4)

    detections = []
    for i in range(len(scores)):
        detections.append({
            'label': labels[i].item(),
            'label_name': CLASS_NAMES[labels[i].item()] if labels[i].item() < len(CLASS_NAMES) else f'class_{labels[i].item()}',
            'confidence': scores[i].item(),
            'box': boxes_xyxy[i].cpu().tolist(),  # normalized (x1, y1, x2, y2)
        })

    return detections


def visualize_detections(img: Image.Image, detections: list,
                         save_path: str = None) -> Image.Image:
    """
    Draw bounding boxes and labels on the image.

    Args:
        img: Original PIL image.
        detections: List of detection dicts from predict().
        save_path: If provided, save the annotated image.

    Returns:
        Annotated PIL image.
    """
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Try to use a nice font, fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for det in detections:
        # Scale normalized box to image size
        x1 = det['box'][0] * w
        y1 = det['box'][1] * h
        x2 = det['box'][2] * w
        y2 = det['box'][3] * h

        label_idx = det['label']
        color = COLORS[label_idx % len(COLORS)]

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label background
        text = f"{det['label_name']} {det['confidence']:.2f}"
        bbox = draw.textbbox((x1, y1), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1],
                       fill=color)
        draw.text((x1 + 2, y1 - text_h - 2), text, fill='white', font=font)

    if save_path:
        img.save(save_path)
        print(f"  Saved: {save_path}")

    return img


def main():
    parser = argparse.ArgumentParser(description='DETR Inference')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to a single image for inference')
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(CHECKPOINT_DIR, 'best_model.pth'),
                        help='Path to model checkpoint')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Confidence threshold for detections')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save output images')
    parser.add_argument('--img_size', type=int, default=IMAGE_SIZE,
                        help='Input image size (shortest side)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Please train the model first with: python train.py")
        sys.exit(1)

    model = load_model(args.checkpoint, device)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if args.image:
        # ---- Single image inference ----
        print(f"\nRunning inference on: {args.image}")
        img_tensor, orig_img, orig_size = preprocess_image(
            args.image, args.img_size
        )

        detections = predict(model, img_tensor, device, args.threshold)

        print(f"  Found {len(detections)} detection(s):")
        for det in detections:
            print(f"    {det['label_name']}: {det['confidence']:.3f} "
                  f"box={[f'{x:.3f}' for x in det['box']]}")

        # Visualize
        out_name = os.path.splitext(os.path.basename(args.image))[0]
        save_path = os.path.join(args.output_dir, f'{out_name}_det.jpg')
        visualize_detections(orig_img, detections, save_path)

    else:
        # ---- Run on test set ----
        test_img_dir = os.path.join(TEST_DIR, 'images')
        if not os.path.exists(test_img_dir):
            print(f"Error: Test images not found at {test_img_dir}")
            sys.exit(1)

        import glob
        img_paths = sorted(glob.glob(os.path.join(test_img_dir, '*.jpg')))
        img_paths += sorted(glob.glob(os.path.join(test_img_dir, '*.png')))

        print(f"\nRunning inference on {len(img_paths)} test images...")

        total_detections = 0
        for i, img_path in enumerate(img_paths):
            img_tensor, orig_img, orig_size = preprocess_image(
                img_path, args.img_size
            )

            detections = predict(model, img_tensor, device, args.threshold)
            total_detections += len(detections)

            # Save visualization
            out_name = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(args.output_dir, f'{out_name}_det.jpg')
            visualize_detections(orig_img, detections, save_path)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(img_paths)} images...")

        print(f"\nDone! Total detections: {total_detections}")
        print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
