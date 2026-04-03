"""
Evaluate SSL + MMViT model on shared test split (same as baselines).
Produces metrics_mvm_vit.json for comparison.
"""

import sys
import json
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Add baselines to path for dataset_utils
sys.path.insert(0, str(Path(__file__).parent / "baselines"))

# Add backend to path for imports
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

from dataset_utils import CLASS_NAMES, NUM_CLASSES, get_dataloaders

from app.config import BASE_DIR
from app.models.vit_classifier import create_model

def evaluate_on_test_set(model_checkpoint_path, data_dir, split_dir, batch_size=32, save_path=None):
    """
    Evaluate SSL + MMViT model on the shared test split.
    
    Args:
        model_checkpoint_path: Path to checkpoint_best.pt 
        data_dir: Path to data/train folder
        split_dir: Path to splits directory
        batch_size: Batch size for evaluation
        save_path: Directory to save results JSON
    """
    if save_path is None:
        save_path = Path(__file__).parent
    else:
        save_path = Path(save_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("\n" + "=" * 70)
    print("SSL + MMViT MODEL EVALUATION ON TEST SET")
    print("=" * 70)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {model_checkpoint_path}")
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    
    # Create model
    print("Creating ViT model...")
    model = create_model(
        model_name="vit_base_patch16_224",
        num_classes=NUM_CLASSES,
        pretrained=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load data
    print("\nLoading dataloaders...")
    train_loader, val_loader, test_loader, dataset_sizes = get_dataloaders(
        data_dir=data_dir,
        split_dir=split_dir,
        batch_size=batch_size,
        num_workers=0
    )
    
    print(f"Test set size: {dataset_sizes['test']}")
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("EVALUATING ON TEST SET")
    print("=" * 70)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluation"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    print(f"\n{'=' * 70}")
    print("PERFORMANCE METRICS")
    print(f"{'=' * 70}")
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Macro F1:     {macro_f1:.4f}")
    print(f"Weighted F1:  {weighted_f1:.4f}")
    
    # Per-class metrics
    print(f"\n{'=' * 70}")
    print("PER-CLASS PERFORMANCE")
    print(f"{'=' * 70}")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Prepare metrics dictionary (same format as baselines)
    metrics = {
        "model": "SSL + MMViT (ViT-Base)",
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "evaluation_split": "test",
        "num_samples": len(all_labels),
        "checkpoint_epoch": int(checkpoint.get('epoch', 0)) + 1,
        "best_val_f1": float(checkpoint.get('best_acc', 0)),
        "per_class_metrics": {}
    }
    
    # Add per-class metrics
    for i, class_name in enumerate(CLASS_NAMES):
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
    metrics_path = save_path / "metrics_mvm_vit.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to: {metrics_path}")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    
    return metrics


if __name__ == "__main__":
    # Resolve paths
    script_dir = Path(__file__).parent
    data_dir = str(script_dir / "backend" / "data" / "train")
    split_dir = str(script_dir / "baselines" / "splits")
    checkpoint_path = str(script_dir / "backend" / "models" / "checkpoint_best.pt")
    
    print(f"Data dir: {data_dir}")
    print(f"Split dir: {split_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    
    evaluate_on_test_set(
        model_checkpoint_path=checkpoint_path,
        data_dir=data_dir,
        split_dir=split_dir,
        save_path=script_dir / "baselines",
    )
