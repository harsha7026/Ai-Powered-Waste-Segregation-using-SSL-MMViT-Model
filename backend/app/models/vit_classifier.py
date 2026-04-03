import torch
import torch.nn as nn
import timm
from typing import Optional


class WasteViTClassifier(nn.Module):
    """
    Vision Transformer classifier for waste segregation.
    
    Uses a pretrained ViT backbone from timm library and adds
    a classification head for 4 waste categories.
    """
    
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        num_classes: int = 4,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        # Load pretrained ViT backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove original classifier
        )
        
        # Get feature dimension
        if hasattr(self.backbone, 'num_features'):
            self.feature_dim = self.backbone.num_features
        elif hasattr(self.backbone, 'embed_dim'):
            self.feature_dim = self.backbone.embed_dim
        else:
            # Default for vit_base
            self.feature_dim = 768
        
        # Optionally freeze backbone weights
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def unfreeze_backbone(self, num_layers: Optional[int] = None):
        """Unfreeze backbone layers for fine-tuning."""
        if num_layers is None:
            # Unfreeze all
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Unfreeze last N transformer blocks
            if hasattr(self.backbone, 'blocks'):
                for block in self.backbone.blocks[-num_layers:]:
                    for param in block.parameters():
                        param.requires_grad = True


def create_model(
    model_name: str = "vit_base_patch16_224",
    num_classes: int = 4,
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None
) -> WasteViTClassifier:
    """
    Factory function to create and optionally load a trained model.
    
    Args:
        model_name: Name of the timm model to use
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights for backbone
        checkpoint_path: Path to saved model weights
        
    Returns:
        Initialized model
    """
    model = WasteViTClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained
    )
    
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
    
    return model
