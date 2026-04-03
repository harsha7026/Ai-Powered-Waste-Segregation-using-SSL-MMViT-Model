"""
Model Evaluation Script
Generates comprehensive performance metrics including per-class analysis
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json

from app.config import BASE_DIR, CLASS_NAMES, NUM_CLASSES, VIT_MODEL_NAME
from app.models.vit_classifier import create_model
from app.services.preprocessing import get_transforms


def compute_confusion_matrix(labels, predictions, num_classes):
    """Compute confusion matrix manually."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(labels, predictions):
        cm[true][pred] += 1
    return cm


def compute_metrics(cm, class_idx):
    """Compute precision, recall, f1 for a class from confusion matrix."""
    tp = cm[class_idx, class_idx]
    fp = cm[:, class_idx].sum() - tp
    fn = cm[class_idx, :].sum() - tp
    support = cm[class_idx, :].sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1, support


def evaluate_model(checkpoint_path, data_dir, batch_size=32):
    """
    Comprehensive model evaluation with per-class metrics.
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation on device: {device}\n")
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Display checkpoint info
    print(f"\n{'='*60}")
    print(f"CHECKPOINT INFORMATION")
    print(f"{'='*60}")
    print(f"Epoch: {checkpoint['epoch'] + 1}")
    print(f"Best Validation Accuracy: {checkpoint['best_acc']:.4f}")
    print(f"Training Loss: {checkpoint['train_loss']:.4f}")
    print(f"Training Accuracy: {checkpoint['train_acc']:.4f}")
    print(f"Validation Loss: {checkpoint['val_loss']:.4f}")
    print(f"Validation Accuracy: {checkpoint['val_acc']:.4f}")
    
    # Create model
    print(f"\n{'='*60}")
    print(f"MODEL ARCHITECTURE")
    print(f"{'='*60}")
    model = create_model(
        model_name=VIT_MODEL_NAME,
        num_classes=NUM_CLASSES,
        pretrained=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model: {VIT_MODEL_NAME}")
    print(f"Number of Classes: {NUM_CLASSES}")
    print(f"Classes: {CLASS_NAMES}")
    
    # Load validation dataset
    print(f"\n{'='*60}")
    print(f"DATASET INFORMATION")
    print(f"{'='*60}")
    full_dataset = datasets.ImageFolder(
        root=data_dir,
        transform=get_transforms(train=False)
    )
    
    # Use same split as training (20% validation)
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * 0.2)
    train_size = dataset_size - val_size
    
    # Get validation split
    _, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    print(f"Total Dataset Size: {dataset_size}")
    print(f"Training Set Size: {train_size}")
    print(f"Validation Set Size: {val_size}")
    
    # Count samples per class
    class_counts = {name: 0 for name in CLASS_NAMES}
    for _, label in full_dataset:
        class_counts[CLASS_NAMES[label]] += 1
    
    print(f"\nClass Distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} images ({count/dataset_size*100:.1f}%)")
    
    # Perform evaluation
    print(f"\n{'='*60}")
    print(f"EVALUATION IN PROGRESS")
    print(f"{'='*60}")
    
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    print(f"\n{'='*60}")
    print(f"PERFORMANCE METRICS")
    print(f"{'='*60}")
    
    # Overall accuracy
    accuracy = np.mean(all_labels == all_predictions)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Confusion matrix
    print(f"\n{'='*60}")
    print(f"CONFUSION MATRIX")
    print(f"{'='*60}\n")
    cm = compute_confusion_matrix(all_labels, all_predictions, NUM_CLASSES)
    
    # Classification report
    print(f"\n{'='*60}")
    print(f"PER-CLASS PERFORMANCE")
    print(f"{'='*60}\n")
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 60)
    
    total_support = 0
    weighted_precision = 0
    weighted_recall = 0
    weighted_f1 = 0
    
    for i, class_name in enumerate(CLASS_NAMES):
        precision, recall, f1, support = compute_metrics(cm, i)
        print(f"{class_name:<12} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10}")
        total_support += support
        weighted_precision += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1 * support
    
    print("-" * 60)
    print(f"{'Weighted Avg':<12} {weighted_precision/total_support:>10.4f} {weighted_recall/total_support:>10.4f} {weighted_f1/total_support:>10.4f} {total_support:>10}")
    
    # Print confusion matrix with headers
    print(f"{'':12}", end='')
    for name in CLASS_NAMES:
        print(f"{name:>12}", end='')
    print()
    
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:12}", end='')
        for j in range(len(CLASS_NAMES)):
            print(f"{cm[i][j]:>12}", end='')
        print()
    
    # Average confidence scores
    print(f"\n{'='*60}")
    print(f"AVERAGE CONFIDENCE SCORES")
    print(f"{'='*60}\n")
    
    for i, class_name in enumerate(CLASS_NAMES):
        class_mask = all_labels == i
        if class_mask.sum() > 0:
            avg_confidence = all_probabilities[class_mask, i].mean()
            print(f"{class_name}: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")
    
    # Save results to JSON
    results = {
        "checkpoint_info": {
            "epoch": int(checkpoint['epoch']) + 1,
            "best_acc": float(checkpoint['best_acc']),
            "train_loss": float(checkpoint['train_loss']),
            "train_acc": float(checkpoint['train_acc']),
            "val_loss": float(checkpoint['val_loss']),
            "val_acc": float(checkpoint['val_acc'])
        },
        "model_info": {
            "architecture": VIT_MODEL_NAME,
            "num_classes": NUM_CLASSES,
            "classes": CLASS_NAMES
        },
        "dataset_info": {
            "total_size": dataset_size,
            "train_size": train_size,
            "val_size": val_size,
            "class_distribution": class_counts
        },
        "evaluation_metrics": {
            "overall_accuracy": float(accuracy),
            "per_class_metrics": {},
            "confusion_matrix": cm.tolist()
        }
    }
    
    # Add per-class metrics
    for i, class_name in enumerate(CLASS_NAMES):
        precision, recall, f1, support = compute_metrics(cm, i)
        avg_conf = float(all_probabilities[all_labels == i, i].mean()) if (all_labels == i).sum() > 0 else 0.0
        
        results["evaluation_metrics"]["per_class_metrics"][class_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "support": int(support),
            "avg_confidence": avg_conf
        }
    
    # Save to file
    results_file = Path("models/evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    checkpoint_path = BASE_DIR / "models" / "checkpoint_best.pt"
    data_dir = BASE_DIR / "data" / "train"
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        exit(1)
    
    if not data_dir.exists():
        print(f"Error: Data directory not found at {data_dir}")
        exit(1)
    
    evaluate_model(str(checkpoint_path), str(data_dir))
