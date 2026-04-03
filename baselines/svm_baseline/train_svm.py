"""
SVM Baseline Classifier for Waste Segmentation
Trains a classical ML SVM on handcrafted features (color histogram + HOG).
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid


# Add parent directory to path to import dataset_utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset_utils import CLASS_NAMES, get_split_records, resolve_default_data_dir


def extract_handcrafted_features(image_path, image_size=224):
    """
    Extract handcrafted features from an image:
    - Color histogram (32 bins per channel = 96 features)
    - HOG (histogram of oriented gradients)
    
    Args:
        image_path: Path to image file
        image_size: Resize image to this size before feature extraction
        
    Returns:
        1D numpy array of concatenated features
    """
    # Load image
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize
    img = img.resize((image_size, image_size))
    img_array = np.array(img)
    
    # Color histogram features (32 bins per channel)
    hist_features = []
    for channel in range(3):
        hist = np.histogram(img_array[:, :, channel], bins=32, range=(0, 256))[0]
        hist = hist / (hist.sum() + 1e-6)  # Normalize
        hist_features.append(hist)
    
    hist_features = np.concatenate(hist_features)  # 96 features
    
    # HOG features
    # Convert to grayscale for HOG
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    hog_features = hog(
        img_gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True
    )
    
    # Concatenate all features
    all_features = np.concatenate([hist_features, hog_features])
    
    return all_features


def load_features_and_labels(data_dir, records, desc="Loading"):
    """
    Load images from data directory and extract handcrafted features.
    
    Args:
        data_dir: Path to data/train folder
        split_indices: If provided, only load specific indices (for train/val split)
        desc: Description for progress bar
        
    Returns:
        (features_array, labels_array)
    """
    data_dir = Path(data_dir)
    image_paths = [str(data_dir / r["relative_path"]) for r in records]
    labels_list = [int(r["label"]) for r in records]
    
    features_list = []
    for img_path in tqdm(image_paths, desc=desc):
        try:
            features = extract_handcrafted_features(img_path)
            features_list.append(features)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # Convert to numpy arrays
    X = np.array(features_list)
    y = np.array(labels_list)
    
    return X, y


def load_or_extract_features(data_dir, train_records, val_records, test_records, cache_path):
    """Load cached features if present; otherwise extract and cache them."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        cache = np.load(cache_path)
        print(f"Loaded feature cache from: {cache_path}")
        return (
            cache["X_train"],
            cache["y_train"],
            cache["X_val"],
            cache["y_val"],
            cache["X_test"],
            cache["y_test"],
        )

    print("\n" + "=" * 70)
    print("EXTRACTING HANDCRAFTED FEATURES")
    print("=" * 70)

    print("\nExtracting features for training set...")
    X_train, y_train = load_features_and_labels(
        data_dir,
        records=train_records,
        desc="Train features"
    )
    print(f"Train features shape: {X_train.shape}")

    print("\nExtracting features for validation set...")
    X_val, y_val = load_features_and_labels(
        data_dir,
        records=val_records,
        desc="Val features"
    )
    print(f"Val features shape: {X_val.shape}")

    print("\nExtracting features for test set...")
    X_test, y_test = load_features_and_labels(
        data_dir,
        records=test_records,
        desc="Test features"
    )
    print(f"Test features shape: {X_test.shape}")

    np.savez_compressed(
        cache_path,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )
    print(f"Saved feature cache to: {cache_path}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_svm(data_dir, split_dir, save_path=None, mode="fast", pca_components=256):
    """
    Train SVM classifier on handcrafted features.
    
    Args:
        data_dir: Path to data/train folder
        save_path: Directory to save model and metrics (default: current directory)
    """
    if save_path is None:
        save_path = Path(__file__).parent
    else:
        save_path = Path(save_path)
    
    save_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("SVM BASELINE TRAINING")
    print("=" * 70)
    
    train_records, val_records, test_records = get_split_records(data_dir=data_dir, split_dir=split_dir)
    print(f"Train samples: {len(train_records)}")
    print(f"Val samples: {len(val_records)}")
    print(f"Test samples: {len(test_records)}")

    feature_cache_path = save_path / "svm_features_cache.npz"
    X_train, y_train, X_val, y_val, X_test, y_test = load_or_extract_features(
        data_dir=data_dir,
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        cache_path=feature_cache_path,
    )
    
    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    if pca_components and pca_components > 0:
        print(f"Applying PCA: {pca_components} components")
        pca = PCA(n_components=pca_components, svd_solver="randomized", random_state=42)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_val_scaled = pca.transform(X_val_scaled)
        X_test_scaled = pca.transform(X_test_scaled)
    else:
        pca = None
    
    # Train SVM
    print("\n" + "=" * 70)
    print("TRAINING SVM CLASSIFIER")
    print("=" * 70)
    print("\nSelecting SVM hyperparameters on validation set...")

    if mode == "fast":
        param_grid = {"C": [10], "gamma": [0.001]}
    else:
        param_grid = {"C": [10, 100], "gamma": [0.001, 0.0001]}

    best_macro_f1 = -1.0
    best_params = None
    best_model = None
    for params in ParameterGrid(param_grid):
        candidate = SVC(
            kernel="rbf",
            C=params["C"],
            gamma=params["gamma"],
            verbose=False,
        )
        candidate.fit(X_train_scaled, y_train)
        val_pred = candidate.predict(X_val_scaled)
        val_macro_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
        print(f"Params {params} -> val macro F1: {val_macro_f1:.4f}")
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_params = params
            best_model = candidate

    svm_model = best_model
    print(f"Best params: {best_params}")
    print(f"Best validation macro F1: {best_macro_f1:.4f}")

    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 70)
    y_pred = svm_model.predict(X_test_scaled)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    
    # Per-class metrics
    print("\n" + "=" * 70)
    print("PER-CLASS PERFORMANCE")
    print("=" * 70)
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Save model and scaler
    model_path = save_path / "svm_model.joblib"
    scaler_path = save_path / "svm_scaler.joblib"
    pca_path = save_path / "svm_pca.joblib"
    
    joblib.dump(svm_model, model_path)
    joblib.dump(scaler, scaler_path)
    if pca is not None:
        joblib.dump(pca, pca_path)
    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    if pca is not None:
        print(f"PCA saved to: {pca_path}")
    
    # Prepare metrics dictionary
    metrics = {
        "model": "SVM (RBF kernel)",
        "mode": mode,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "evaluation_split": "test",
        "best_params": best_params,
        "best_val_macro_f1": float(best_macro_f1),
        "pca_components": int(pca_components) if pca is not None else 0,
        "num_samples": len(y_test),
        "per_class_metrics": {}
    }
    
    # Add per-class metrics
    for i, class_name in enumerate(CLASS_NAMES):
        class_mask = y_test == i
        
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
    metrics_path = save_path / "metrics_svm.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to: {metrics_path}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    # Path to data/train folder (adjust if running from different location)
    # Expected structure: waste-segmentation-project/backend/data/train/
    script_dir = Path(__file__).parent
    
    try:
        data_dir = str(resolve_default_data_dir())
    except FileNotFoundError:
        print("ERROR: Could not find data/train directory")
        sys.exit(1)

    split_dir = str(script_dir.parent / "splits")
    parser = argparse.ArgumentParser(description="Train SVM baseline.")
    parser.add_argument("--mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--pca-components", type=int, default=256)
    args = parser.parse_args()

    print(f"Using data directory: {data_dir}")
    print(f"Using split directory: {split_dir}")
    train_svm(
        data_dir=data_dir,
        split_dir=split_dir,
        save_path=script_dir,
        mode=args.mode,
        pca_components=args.pca_components,
    )
