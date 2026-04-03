"""
Resume training from the latest checkpoint.
"""
from app.services.training import train_model
from pathlib import Path

if __name__ == "__main__":
    # Path to the latest checkpoint
    checkpoint_path = Path("models/checkpoint_latest.pt")
    
    if checkpoint_path.exists():
        print(f"Found checkpoint: {checkpoint_path}")
        print("Resuming training...")
        
        train_model(
            num_epochs=10,           # Total epochs (will continue from saved epoch)
            batch_size=32,
            learning_rate=1e-4,
            freeze_backbone=True,
            resume_from=str(checkpoint_path)
        )
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        print("Starting training from scratch...")
        train_model(
            num_epochs=10,
            batch_size=32,
            learning_rate=1e-4,
            freeze_backbone=True
        )
