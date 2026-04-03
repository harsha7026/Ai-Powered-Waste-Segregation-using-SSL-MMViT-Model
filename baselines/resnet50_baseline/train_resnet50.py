"""
ResNet-50 CNN Baseline for Waste Segmentation
Fine-tunes pretrained ResNet-50 on waste classification task.
"""

import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import numpy as np

from torchvision import models
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Add parent directory to path to import dataset_utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset_utils import (
    CLASS_NAMES, NUM_CLASSES, get_dataloaders, resolve_default_data_dir
)


class ResNet50Classifier(nn.Module):
    """ResNet-50 with custom classifier head for 5-class waste classification."""
    
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Replace final FC layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc="Training")
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return epoch_loss, epoch_acc, epoch_f1


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return epoch_loss, epoch_acc, epoch_f1, all_labels, all_preds


def train_resnet50(
    data_dir,
    split_dir,
    num_epochs=30,
    batch_size=32,
    learning_rate=1e-4,
    patience=10,
    freeze_backbone=False,
    save_path=None
):
    """
    Train ResNet-50 classifier.
    
    Args:
        data_dir: Path to data/train folder
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        save_path: Directory to save model and metrics
    """
    if save_path is None:
        save_path = Path(__file__).parent
    else:
        save_path = Path(save_path)
    
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("\n" + "=" * 70)
    print("RESNET-50 CNN BASELINE TRAINING")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader, dataset_sizes = get_dataloaders(
        data_dir,
        split_dir=split_dir,
        batch_size=batch_size,
        num_workers=0  # Windows compatibility
    )
    
    # Create model
    print("\nCreating ResNet-50 model...")
    model = ResNet50Classifier(num_classes=NUM_CLASSES, pretrained=True)
    if freeze_backbone:
        for name, param in model.resnet.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training variables
    best_val_f1 = 0.0
    best_model_state = None
    patience_counter = 0
    
    print("\n" + "=" * 70)
    print("TRAINING LOOP")
    print("=" * 70)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
        print("-" * 70)
        
        # Train
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_f1, _, _ = validate(
            model, val_loader, criterion, device
        )
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1:   {val_f1:.4f}")
        
        scheduler.step()
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"✓ New best model (F1: {val_f1:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered (patience={patience})")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Save best checkpoint
    checkpoint_path = save_path / "resnet50_best.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'best_val_f1': best_val_f1,
    }, checkpoint_path)
    print(f"\nBest model saved to: {checkpoint_path}")
    
    # Final evaluation on test set
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 70)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Final evaluation"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    
    # Per-class metrics
    print("\n" + "=" * 70)
    print("PER-CLASS PERFORMANCE")
    print("=" * 70)
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    
    # Prepare metrics dictionary
    metrics = {
        "model": "ResNet-50",
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "evaluation_split": "test",
        "num_samples": len(all_labels),
        "num_epochs_trained": epoch + 1,
        "best_val_f1": float(best_val_f1),
        "per_class_metrics": {}
    }
    
    # Add per-class metrics
    for i, class_name in enumerate(CLASS_NAMES):
        class_mask = all_labels == i
        
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics["per_class_metrics"][class_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support)
        }
    
    # Save metrics
    metrics_path = save_path / "metrics_resnet50.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to: {metrics_path}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    # Path to data/train folder
    script_dir = Path(__file__).parent
    
    try:
        data_dir = str(resolve_default_data_dir())
    except FileNotFoundError:
        print("ERROR: Could not find data/train directory")
        sys.exit(1)

    split_dir = str(script_dir.parent / "splits")
    parser = argparse.ArgumentParser(description="Train ResNet-50 baseline.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--freeze-backbone", action="store_true")
    args = parser.parse_args()

    print(f"Using data directory: {data_dir}")
    print(f"Using split directory: {split_dir}")
    train_resnet50(
        data_dir=data_dir,
        split_dir=split_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        freeze_backbone=args.freeze_backbone,
        save_path=script_dir,
    )
