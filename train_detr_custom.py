import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import config
import configs.train_config as cfg

# Import components
from model.detr import build_detr
from model.matcher import HungarianMatcher
from model.criterion import SetCriterion
from dataset.coco_dataset import COCODETRDataset, coco_detr_collate_fn

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading datasets...")
    train_dataset = COCODETRDataset(
        root=cfg.TRAIN_ROOT,
        ann_file=cfg.TRAIN_ANN,
        class_names=cfg.CLASS_NAMES,
        augment=True,
        img_size=cfg.IMAGE_SIZE
    )
    
    val_dataset = COCODETRDataset(
        root=cfg.VAL_ROOT,
        ann_file=cfg.VAL_ANN,
        class_names=cfg.CLASS_NAMES,
        augment=False,
        img_size=cfg.IMAGE_SIZE
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=coco_detr_collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=coco_detr_collate_fn,
        pin_memory=True
    )

    print(f"Building DETR model (Backbone: {cfg.BACKBONE_NAME})...")
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
    model.to(device)
    
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad],
            "lr": cfg.LR_TRANSFORMER,
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": cfg.LR_BACKBONE,
        },
    ]
    
    optimizer = torch.optim.AdamW(param_dicts, weight_decay=cfg.WEIGHT_DECAY)
    
    # LR Scheduler (optional but recommended in paper)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.LR_DROP, gamma=cfg.LR_DROP_FACTOR)
    
    matcher = HungarianMatcher(
        cost_class=cfg.COST_CLASS, 
        cost_bbox=cfg.COST_BBOX, 
        cost_giou=cfg.COST_GIOU
    )
    
    criterion = SetCriterion(
        num_classes=cfg.NUM_CLASSES,
        matcher=matcher,
        weight_ce=cfg.WEIGHT_CE,
        weight_bbox=cfg.WEIGHT_BBOX,
        weight_giou=cfg.WEIGHT_GIOU,
        eos_coef=cfg.EOS_COEF
    )
    criterion.to(device)
    
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    best_val_loss = float('inf')
    
    print(f"Starting training on {cfg.NUM_CLASSES} classes...")
    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        criterion.train()
        
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS} [Train]")
        
        for images, masks, targets in pbar:
            images = images.to(device)
            masks = masks.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            outputs = model(images, mask=masks)
            loss_dict = criterion(outputs, targets)
            loss = loss_dict['loss_total']
            
            optimizer.zero_grad()
            loss.backward()
            
            if cfg.CLIP_MAX_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_MAX_NORM)
                
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'ce': f"{loss_dict['loss_ce'].item():.4f}",
                'l1': f"{loss_dict['loss_bbox'].item():.4f}",
                'giou': f"{loss_dict['loss_giou'].item():.4f}"
            })
            
        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()
        
        # Validation
        model.eval()
        criterion.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks, targets in tqdm(val_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS} [Val]"):
                images = images.to(device)
                masks = masks.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                outputs = model(images, mask=masks)
                loss_dict = criterion(outputs, targets)
                val_loss += loss_dict['loss_total'].item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, "best_model.pth"))
            print(f"  --> Saved Best Model (Loss: {best_val_loss:.4f})")
            
        if epoch % cfg.SAVE_EVERY == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, os.path.join(cfg.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth"))

    print("Training Complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Training failed: {e}")
        print("\nPossible issues: Check paths, class IDs, or CUDA memory.")

