import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import random
import os
import sys
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, Any, List

# Add project root to sys.path to allow standalone testing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
try:
    from src.preprocessing.patch_cropper import XBDPatchDataset
except ImportError:
    print("Warning: Could not import XBDPatchDataset. Ensure you are running from the project root.")

class ResNet50DamageClassifier(nn.Module):
    """
    ResNet50-based building damage classifier.
    Uses ImageNet-pretrained backbone with fine-tuned classification head.
    
    Architecture:
    - Backbone: ResNet50 pretrained on ImageNet (frozen during warm-up, 
      then unfrozen for fine-tuning)
    - Classifier head: Linear(2048, 512) → ReLU → Dropout(0.3) → Linear(512, 4)
    
    Input: Tensor [B, 3, 224, 224]
    Output: Logits tensor [B, 4]
    """
    
    def __init__(self, num_classes=4, dropout=0.3, pretrained=True):
        super().__init__()
        # Load pretrained ResNet50
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = resnet50(weights=weights)
        
        # Remove the final FC layer by replacing it with Identity
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Add custom classification head
        self.head = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        # Initially freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Ensure head is trainable
        for param in self.head.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        # Pass through backbone
        features = self.backbone(x)
        # Pass through head
        logits = self.head(features)
        # Return logits (NOT softmax — use CrossEntropyLoss which applies internally)
        return logits
    
    def unfreeze_backbone(self, layers_to_unfreeze="last2"):
        """
        Progressively unfreeze backbone layers.
        layers_to_unfreeze options:
          "none" — keep all frozen
          "last1" — unfreeze layer4 only
          "last2" — unfreeze layer3 + layer4 (default after warm-up)
          "all" — unfreeze entire backbone
        Print which layers were unfrozen.
        """
        if layers_to_unfreeze == "none":
            print("Unfrozen backbone layers: none")
            return
            
        unfrozen_layers = []
        if layers_to_unfreeze == "all":
            for param in self.backbone.parameters():
                param.requires_grad = True
            unfrozen_layers = ["all"]
        else:
            if layers_to_unfreeze in ["last1", "last2"]:
                for param in self.backbone.layer4.parameters():
                    param.requires_grad = True
                unfrozen_layers.append("layer4")
            if layers_to_unfreeze == "last2":
                for param in self.backbone.layer3.parameters():
                    param.requires_grad = True
                unfrozen_layers.append("layer3")
                
        print(f"Unfrozen backbone layers: {', '.join(unfrozen_layers)}")
    
    def get_num_trainable_params(self):
        """Returns count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    """Saves model state, optimizer state, epoch, and metrics dict."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(state, filepath)

def load_checkpoint(filepath, model, optimizer=None):
    """Loads checkpoint. Returns (model, optimizer, epoch, metrics)."""
    device = next(model.parameters()).device
    state = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in state:
        optimizer.load_state_dict(state['optimizer_state_dict'])
    return model, optimizer, state.get('epoch', 0), state.get('metrics', {})

def evaluate_baseline_a(model, dataloader, device, class_names):
    """
    Evaluates model on a given dataloader.
    
    Returns dict:
    {
      "accuracy": float,
      "macro_f1": float,
      "per_class_f1": {class_name: float},
      "variance_score": float,  <- std of per_class_f1 values
      "confusion_matrix": np.ndarray,
      "all_predictions": list,
      "all_labels": list,
      "all_softmax_probs": np.ndarray  <- shape [N, 4], needed for ensemble later
    }
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            pre_patch, post_patch, labels, meta = batch
            
            post_patch = post_patch.to(device)
            labels = labels.to(device)
            
            logits = model(post_patch)
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
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    per_class_f1_arr = f1_score(all_labels, all_preds, average=None, labels=labels_range)
    
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

def train_baseline_a(xbd_root, disaster_types=None, 
                     checkpoint_dir=None, results_dir=None,
                     num_epochs=10, batch_size=32, 
                     learning_rate=1e-3, seed=42):
    """
    Complete training loop for ResNet50 Baseline A.
    """
    # 1. Set all random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        
    # 2. Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
    # 3. Create train and val datasets
    train_dataset = XBDPatchDataset(xbd_root, disaster_types=disaster_types, split="train", seed=seed)
    val_dataset = XBDPatchDataset(xbd_root, disaster_types=disaster_types, split="val", seed=seed)
    
    if len(train_dataset) == 0:
        print("Training dataset is empty. Aborting training.")
        return {}
        
    # 4. Print class distribution
    print(f"Train split summary: {train_dataset.get_split_summary()}")
    print(f"Val split summary: {val_dataset.get_split_summary()}")
    
    # 5. Create DataLoaders
    # Using num_workers=0 to prevent local environment hanging/pickling issues, but perfectly adaptable to Colab
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 6. Get class weights and criterion
    class_weights = train_dataset.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    model = ResNet50DamageClassifier().to(device)
    
    # 7. Phase 1 Optimizer
    optimizer = torch.optim.Adam(model.head.parameters(), lr=learning_rate)
    
    # 9. LR scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    class_names = ["no-damage", "minor-damage", "major-damage", "destroyed"]
    
    best_macro_f1 = 0.0
    best_epoch = 0
    best_val_accuracy = 0.0
    
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "baseline_a_best.pt")
    else:
        checkpoint_path = "baseline_a_best.pt"
        
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        
    log_data = []
    
    start_epoch = 1
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint {checkpoint_path}...")
        model, optimizer, loaded_epoch, metrics = load_checkpoint(checkpoint_path, model, optimizer)
        start_epoch = loaded_epoch + 1
        best_macro_f1 = metrics.get('macro_f1', 0.0)
        best_val_accuracy = metrics.get('accuracy', 0.0)
        best_epoch = loaded_epoch
        print(f"Resuming at epoch {start_epoch} (Best F1: {best_macro_f1:.4f})")
    
    # 10. Training loop
    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\\n--- Epoch {epoch}/{num_epochs} ---")
        
        # 8. Phase 2 (start at epoch 4)
        if epoch == 4:
            print("Starting Phase 2: Unfreezing 'last2' layers of backbone.")
            model.unfreeze_backbone("last2")
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate / 10.0)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
            
        model.train()
        running_loss = 0.0
        
        # 13. Use tqdm progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=False)
        for pre_patch, post_patch, labels, meta in progress_bar:
            post_patch = post_patch.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(post_patch)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * post_patch.size(0)
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        epoch_train_loss = running_loss / len(train_dataset)
        
        # Validation
        print("Running validation...")
        val_metrics = evaluate_baseline_a(model, val_loader, device, class_names)
        
        val_acc = val_metrics["accuracy"]
        val_macro_f1 = val_metrics["macro_f1"]
        
        # Compute val loss for scheduler
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for pre_patch, post_patch, labels, meta in val_loader:
                post_patch = post_patch.to(device)
                labels = labels.to(device)
                logits = model(post_patch)
                loss = criterion(logits, labels)
                val_loss += loss.item() * post_patch.size(0)
        epoch_val_loss = val_loss / len(val_dataset)
        
        scheduler.step(epoch_val_loss)
        
        print(f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_macro_f1:.4f}")
        
        log_data.append({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_macro_f1
        })
        
        # 11. Save best model based on macro_f1
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_val_accuracy = val_acc
            best_epoch = epoch
            if checkpoint_dir:
                save_checkpoint(model, optimizer, epoch, val_metrics, checkpoint_path)
                print(f"Best model updated and saved to {checkpoint_path}")
                
    # 12. Save training log
    if results_dir:
        df_log = pd.DataFrame(log_data)
        log_csv_path = os.path.join(results_dir, "baseline_a_training_log.csv")
        df_log.to_csv(log_csv_path, index=False)
        print(f"Training log saved to {log_csv_path}")
        
    return {
        "best_val_accuracy": best_val_accuracy,
        "best_val_macro_f1": best_macro_f1,
        "best_epoch": best_epoch,
        "train_losses": [row["train_loss"] for row in log_data],
        "val_accuracies": [row["val_accuracy"] for row in log_data],
        "val_macro_f1s": [row["val_macro_f1"] for row in log_data],
        "checkpoint_path": checkpoint_path
    }

def plot_training_curves(training_log_csv_path, save_path):
    """
    Plots 3 subplots:
    1. Training loss over epochs
    2. Validation accuracy over epochs
    3. Validation macro-F1 over epochs
    """
    if not os.path.exists(training_log_csv_path):
        print(f"File not found: {training_log_csv_path}")
        return
        
    df = pd.read_csv(training_log_csv_path)
    best_epoch = df.loc[df['val_macro_f1'].idxmax()]['epoch']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(df['epoch'], df['train_loss'], marker='o', label='Train Loss')
    axes[0].plot(df['epoch'], df['val_loss'], marker='s', label='Val Loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].axvline(best_epoch, color='red', linestyle='--', alpha=0.5)
    
    axes[1].plot(df['epoch'], df['val_accuracy'], marker='o', color='green')
    axes[1].set_title('Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].axvline(best_epoch, color='red', linestyle='--', alpha=0.5)
    
    axes[2].plot(df['epoch'], df['val_macro_f1'], marker='o', color='purple')
    axes[2].set_title('Validation Macro-F1')
    axes[2].set_xlabel('Epoch')
    axes[2].axvline(best_epoch, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_confusion_matrix(cm, class_names, title, save_path):
    """
    Plots normalised confusion matrix using seaborn heatmap.
    Values shown as percentages.
    """
    # Normalize by row (true label)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm) # handle division by zero
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm * 100, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

if __name__ == "__main__":
    import yaml
    import os
    
    # Dynamically find the project root to load the config, no matter where the script is run from
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    config_path = os.path.join(project_root, 'configs', 'config.yaml')
    
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        results = train_baseline_a(
            xbd_root=config["paths"]["xbd_root"],
            disaster_types=None,
            checkpoint_dir=config["paths"]["checkpoint_dir"],
            results_dir=config["paths"]["results_dir"],
            num_epochs=config.get("training", {}).get("num_train_epochs", 5),
            batch_size=32,
            seed=42
        )
        
        if results:
            print("\\n=== BASELINE A TRAINING COMPLETE ===")
            print(f"Best Val Accuracy: {results['best_val_accuracy']:.4f}")
            print(f"Best Val Macro-F1: {results['best_val_macro_f1']:.4f}")
            print(f"Best Epoch: {results['best_epoch']}")
            print(f"Checkpoint saved: {results['checkpoint_path']}")
            
    except FileNotFoundError:
        print("configs/config.yaml not found. Run from the project root directory.")
