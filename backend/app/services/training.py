import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from app.config import BASE_DIR, CLASS_NAMES, NUM_CLASSES, VIT_MODEL_NAME
from app.models.vit_classifier import create_model
from app.services.preprocessing import get_transforms


class WasteDataset(Dataset):
    """
    Custom dataset for waste classification.
    Expects data in format: data/train/<class_name>/<images>
    """
    
    def __init__(self, root_dir: str, transform=None):
        self.dataset = datasets.ImageFolder(root=root_dir, transform=transform)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 4
):
    """
    Create training and validation dataloaders.
    
    Args:
        data_dir: Root directory containing train/<class_name> folders
        batch_size: Batch size for dataloaders
        val_split: Fraction of data to use for validation
        num_workers: Number of worker processes
        
    Returns:
        train_loader, val_loader, dataset_sizes
    """
    # Create dataset
    full_dataset = WasteDataset(
        root_dir=data_dir,
        transform=get_transforms(train=True)
    )
    
    # Split into train and val
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply different transforms for validation
    val_dataset.dataset.dataset.transform = get_transforms(train=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, {'train': train_size, 'val': val_size}


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
        pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    
    return epoch_loss, epoch_acc


def train_model(
    data_dir: str = None,
    model_name: str = VIT_MODEL_NAME,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    freeze_backbone: bool = True,
    save_path: Optional[str] = None,
    resume_from: Optional[str] = None
):
    """
    Main training function.
    
    Args:
        data_dir: Path to training data directory
        model_name: Name of the ViT model to use
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        freeze_backbone: Whether to freeze backbone initially
        save_path: Path to save the trained model
        resume_from: Path to checkpoint to resume training from
    """
    if data_dir is None:
        data_dir = str(BASE_DIR / "data" / "train")
    
    if save_path is None:
        save_path = str(BASE_DIR / "models" / "waste_classifier.pt")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader, dataset_sizes = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size
    )
    print(f"Dataset sizes: {dataset_sizes}")
    
    # Create model
    print("Creating model...")
    model = create_model(
        model_name=model_name,
        num_classes=NUM_CLASSES,
        pretrained=True
    )
    
    if freeze_backbone:
        print("Freezing backbone...")
        for param in model.backbone.parameters():
            param.requires_grad = False
    
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Variables for resuming
    start_epoch = 0
    best_acc = 0.0
    
    # Resume from checkpoint if provided
    if resume_from and Path(resume_from).exists():
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print(f"Resuming from epoch {start_epoch}, Best Acc: {best_acc:.4f}")
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        scheduler.step()
        
        # Save checkpoint after each epoch
        checkpoint_path = str(Path(save_path).parent / "checkpoint_latest.pt")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            # Also save best checkpoint
            best_checkpoint_path = str(Path(save_path).parent / "checkpoint_best.pt")
            torch.save(checkpoint, best_checkpoint_path)
            print(f"Best model saved to {save_path}")
    
    print(f"\nTraining complete! Best Val Acc: {best_acc:.4f}")
    return model


if __name__ == "__main__":
    # Example usage
    train_model(
        num_epochs=10,
        batch_size=32,
        learning_rate=1e-4,
        freeze_backbone=True
    )
