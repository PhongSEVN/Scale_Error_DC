import os
import torch
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision.transforms import functional as TF

import configs.train_config as cfg
from model.detr import build_detr
from model.criterion import box_cxcywh_to_xyxy

@torch.no_grad()
def visualize_predictions(model, img_path, device, threshold=0.3):
    model.eval()
    
    # Load and preprocess image
    orig_img = Image.open(img_path).convert("RGB")
    w, h = orig_img.size
    
    # Resize to match training size
    shortest_side = cfg.IMAGE_SIZE
    scale = shortest_side / min(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img = TF.resize(orig_img, [new_h, new_w])
    
    # To tensor and normalize
    img_tensor = TF.to_tensor(img)
    img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Forward pass
    outputs = model(img_tensor)
    
    # Process outputs
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold
    
    # Convert boxes to pixel coordinates on the original image
    bboxes_scaled = outputs['pred_boxes'][0, keep].cpu()
    bboxes_scaled = box_cxcywh_to_xyxy(bboxes_scaled)
    bboxes_scaled = bboxes_scaled * torch.tensor([w, h, w, h], dtype=torch.float32)
    
    scores = probas[keep].max(-1).values.cpu().numpy()
    labels = probas[keep].argmax(-1).cpu().numpy()
    
    # Plotting
    plt.figure(figsize=(12, 8))
    plt.imshow(orig_img)
    ax = plt.gca()
    
    colors = plt.cm.hsv(torch.linspace(0, 1, cfg.NUM_CLASSES)).tolist()
    
    for score, label, (x1, y1, x2, y2) in zip(scores, labels, bboxes_scaled):
        color = colors[label]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        label_text = f"{cfg.CLASS_NAMES[label]}: {score:.2f}"
        ax.text(x1, y1, label_text, fontsize=10, bbox=dict(facecolor=color, alpha=0.5))
    
    plt.title(f"DETR Predictions (Threshold: {threshold})")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, "best_model.pth")
    
    if not os.path.exists(checkpoint_path):
        print("Error: No checkpoint found!")
        return
        
    print(f"Loading model: {checkpoint_path}")
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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    
    # Choose a random image from validation set
    val_files = [f for f in os.listdir(cfg.VAL_ROOT) if f.endswith(('.jpg', '.png'))]
    random_img = random.choice(val_files)
    img_path = os.path.join(cfg.VAL_ROOT, random_img)
    
    print(f"Visualizing: {img_path}")
    visualize_predictions(model, img_path, device)

if __name__ == "__main__":
    main()
