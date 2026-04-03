import torch
from torchvision import transforms
from PIL import Image
from app.config import IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD


def get_transforms(train: bool = False):
    """
    Get image preprocessing transforms for ViT.
    
    Args:
        train: If True, returns training transforms with augmentation
        
    Returns:
        torchvision.transforms.Compose object
    """
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
        ])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess a PIL Image for inference.
    
    Args:
        image: PIL Image in RGB format
        
    Returns:
        Preprocessed tensor of shape (1, 3, H, W)
    """
    # Convert to RGB if needed (handles RGBA, grayscale, etc.)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    transform = get_transforms(train=False)
    tensor = transform(image)
    
    # Add batch dimension
    tensor = tensor.unsqueeze(0)
    
    return tensor


def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize image tensor back to [0, 1] range.
    
    Args:
        tensor: Normalized tensor of shape (C, H, W) or (B, C, H, W)
        
    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(IMAGE_MEAN).view(-1, 1, 1)
    std = torch.tensor(IMAGE_STD).view(-1, 1, 1)
    
    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    tensor = tensor * std + mean
    return tensor.clamp(0, 1)
