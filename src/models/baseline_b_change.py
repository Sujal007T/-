import os
import random
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from src.preprocessing.patch_cropper import XBDPatchDataset
from src.models.baseline_a_cnn import save_checkpoint, load_checkpoint, plot_training_curves, plot_confusion_matrix

class ChangeDetectionClassifier(nn.Module):
    """
    Change-detection based building damage classifier.
    Takes pre and post disaster building patches and classifies
    damage from their feature-level difference.
    
    Architecture:
    - Shared ResNet50 backbone (same weights process both images)
    - Change representation: concat(z_pre, z_post, z_post - z_pre)
    - Classification head on 6144-dim change vector
    
    Input: (pre_patch [B,3,224,224], post_patch [B,3,224,224])
    Output: logits [B, 4]
    """
    
    def __init__(self, num_classes=4, dropout=0.3, pretrained=True,
                 backbone_checkpoint=None):
        super().__init__()
        
        # Load backbone
        if pretrained:
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.backbone = resnet50(weights=None)
            
        # Remove the classification head (fc layer)
        self.backbone.fc = nn.Identity()
        
        # Optionally load Baseline A checkpoint to warm-start the backbone
        if backbone_checkpoint and os.path.exists(backbone_checkpoint):
            print(f"Loading backbone weights from Baseline A: {backbone_checkpoint}")
            # Weights only = False is needed because the dict has metrics inside it
            state = torch.load(backbone_checkpoint, map_location='cpu', weights_only=False)
            model_state_dict = state.get('model_state_dict', state)
            
            # Extract just the backbone weights (strip "backbone." prefix)
            backbone_state_dict = {}
            for k, v in model_state_dict.items():
                if k.startswith("backbone."):
                    backbone_state_dict[k.replace("backbone.", "")] = v
                    
            self.backbone.load_state_dict(backbone_state_dict, strict=False)
            
        # Freeze backbone initially
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Classification head: 2048 * 3 = 6144
        self.head = nn.Sequential(
            nn.Linear(6144, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, pre_patch, post_patch):
        """
        Args:
            pre_patch: [B, 3, 224, 224] pre-disaster patches
            post_patch: [B, 3, 224, 224] post-disaster patches
        Returns:
            logits: [B, 4]
        """
        # Extract features from both patches using SAME backbone
        # (weight sharing is key — same weights, different inputs)
        z_pre  = self.backbone(pre_patch)   # [B, 2048]
        z_post = self.backbone(post_patch)  # [B, 2048]
        
        # Change representation
        z_diff   = z_post - z_pre           # [B, 2048]
        z_change = torch.cat([z_pre, z_post, z_diff], dim=1)  # [B, 6144]
        
        return self.head(z_change)
        
    def unfreeze_backbone(self, layers_to_unfreeze="last2"):
        """Same interface as Baseline A for consistency."""
        if layers_to_unfreeze == "all":
            for param in self.backbone.parameters():
                param.requires_grad = True
        elif layers_to_unfreeze == "last2":
            # ResNet layer4 and layer3
            for name, child in self.backbone.named_children():
                if name in ["layer4", "layer3"]:
                    for param in child.parameters():
                        param.requires_grad = True
                        
    def get_num_trainable_params(self):
        """Returns count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def evaluate_baseline_b(model, dataloader, device, class_names):
    """
    Evaluates Baseline B.
    Returns IDENTICAL dict structure to evaluate_baseline_a() —
    this is critical for the ensemble to combine both outputs uniformly.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            pre_patch, post_patch, labels, meta = batch
            
            pre_patch = pre_patch.to(device)
            post_patch = post_patch.to(device)
            labels = labels.to(device)
            
            logits = model(pre_patch, post_patch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
    if len(all_probs) == 0:
        return {}
        
    all_probs = np.vstack(all_probs)
    
    num_classes = len(class_names)
    labels_range = list(range(num_classes))
    
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    per_class_f1_arr = f1_score(all_labels, all_preds, average=None, labels=labels_range, zero_division=0)
    
    per_class_f1 = {class_names[i]: float(per_class_f1_arr[i]) for i in range(num_classes)}
    variance_score = float(np.std(per_class_f1_arr))
    cm = confusion_matrix(all_labels, all_preds, labels=labels_range)
    
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "variance_score": variance_score,
        "confusion_matrix": cm,
        "all_predictions": all_preds,
        "all_labels": all_labels,
        "all_softmax_probs": all_probs
    }

def train_baseline_b(xbd_root, disaster_types=None,
                     checkpoint_dir=None, results_dir=None,
                     baseline_a_checkpoint=None,
                     num_epochs=10, batch_size=16,
                     learning_rate=1e-3, seed=42):
    """
    Training loop for Change-Detection Baseline B.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_dataset = XBDPatchDataset(xbd_root, disaster_types=disaster_types, split="train", seed=seed)
    val_dataset = XBDPatchDataset(xbd_root, disaster_types=disaster_types, split="val", seed=seed)
    
    if len(train_dataset) == 0:
        print("Training dataset empty.")
        return {}
        
    # Note: batch_size=16 because each sample loads TWO images (pre+post)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    class_weights = train_dataset.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    model = ChangeDetectionClassifier(
        pretrained=True, 
        backbone_checkpoint=baseline_a_checkpoint
    ).to(device)
    
    optimizer = torch.optim.Adam(model.head.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    class_names = ["no-damage", "minor-damage", "major-damage", "destroyed"]
    
    best_macro_f1 = 0.0
    best_epoch = 0
    best_val_accuracy = 0.0
    start_epoch = 1
    
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "baseline_b_best.pt")
    else:
        checkpoint_path = "baseline_b_best.pt"
        
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        
    log_data = []
    
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint {checkpoint_path}...")
        model, optimizer, loaded_epoch, metrics = load_checkpoint(checkpoint_path, model, optimizer)
        start_epoch = loaded_epoch + 1
        best_macro_f1 = metrics.get('macro_f1', 0.0)
        best_val_accuracy = metrics.get('accuracy', 0.0)
        best_epoch = loaded_epoch
        print(f"Resuming at epoch {start_epoch} (Best F1: {best_macro_f1:.4f})")
    
    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\n--- Epoch {epoch}/{num_epochs} ---")
        
        # Phase 2: unfreeze backbone at epoch 4
        if epoch == 4:
            print("Starting Phase 2: Unfreezing 'last2' layers of backbone.")
            model.unfreeze_backbone("last2")
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate / 10.0)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
            
        model.train()
        train_loss = 0.0
        
        for i, batch in enumerate(train_loader):
            pre_patch, post_patch, labels, _ = batch
            
            pre_patch = pre_patch.to(device)
            post_patch = post_patch.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(pre_patch, post_patch)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if i % 50 == 0:
                print(f"Epoch {epoch} Train: {i}/{len(train_loader)} - loss: {loss.item():.4f}")
                
        train_loss /= len(train_loader)
        print(f"Epoch {epoch} Avg Train Loss: {train_loss:.4f}")
        
        # Validation
        val_metrics = evaluate_baseline_b(model, val_loader, device, class_names)
        if not val_metrics:
            continue
            
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["macro_f1"]
        print(f"Epoch {epoch} Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        # Baseline A calculates val_loss as 0.0 in its log, reproducing that here
        scheduler.step(train_loss)
        
        log_data.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": 0.0,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1
        })
        
        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            best_val_accuracy = val_acc
            best_epoch = epoch
            print(f"New best model! F1: {best_macro_f1:.4f}")
            save_checkpoint(model, optimizer, epoch, val_metrics, checkpoint_path)
            
    # Post-training processing
    if results_dir and log_data:
        training_log_csv_path = os.path.join(results_dir, "baseline_b_training_log.csv")
        pd.DataFrame(log_data).to_csv(training_log_csv_path, index=False)
        
        try:
            plot_training_curves(training_log_csv_path, os.path.join(results_dir, "baseline_b_history.png"))
            
            # Re-load best model for final CM evaluation
            model, _, _, _ = load_checkpoint(checkpoint_path, model, optimizer=None)
            final_metrics = evaluate_baseline_b(model, val_loader, device, class_names)
            plot_confusion_matrix(final_metrics["confusion_matrix"], class_names, 
                                  "Baseline B: Confusion Matrix", 
                                  os.path.join(results_dir, "baseline_b_confusion_matrix.png"))
        except Exception as e:
            print(f"Failed to generate plots: {e}")
        
    return {
        "best_val_accuracy": best_val_accuracy,
        "best_val_macro_f1": best_macro_f1,
        "best_epoch": best_epoch,
        "checkpoint_path": checkpoint_path,
        "warm_started_from": baseline_a_checkpoint if baseline_a_checkpoint else None
    }

if __name__ == "__main__":
    import yaml
    
    # Try multiple paths for config depending on where script is executed
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    config_path = os.path.join(project_root, 'configs', 'config.yaml')
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    baseline_a_ckpt = os.path.join(
        config["paths"]["checkpoint_dir"], "baseline_a_best.pt"
    )
    
    ckpt = baseline_a_ckpt if os.path.exists(baseline_a_ckpt) else None
    if ckpt:
        print(f"Warm-starting from Baseline A checkpoint: {ckpt}")
    else:
        print("No Baseline A checkpoint found. Using ImageNet weights.")
        
    results = train_baseline_b(
        xbd_root=config["paths"]["xbd_root"],
        checkpoint_dir=config["paths"]["checkpoint_dir"],
        results_dir=config["paths"]["results_dir"],
        baseline_a_checkpoint=ckpt,
        num_epochs=config.get("training", {}).get("num_train_epochs", 5),
        batch_size=16,
        seed=config.get("project", {}).get("seed", 42)
    )
    
    if results:
        print("\n=== BASELINE B TRAINING COMPLETE ===")
        print(f"Best Val Accuracy:  {results['best_val_accuracy']:.4f}")
        print(f"Best Val Macro-F1:  {results['best_val_macro_f1']:.4f}")
        print(f"Best Epoch:         {results['best_epoch']}")
        print(f"Warm-started from:  {results['warm_started_from']}")
        print(f"Checkpoint saved:   {results['checkpoint_path']}")
