import torch
from pathlib import Path
from PIL import Image
from typing import Dict, Tuple
import torch.nn.functional as F
import base64
import io
import numpy as np

from app.config import MODEL_PATH, CLASS_NAMES, DEVICE, VIT_MODEL_NAME, NUM_CLASSES
from app.models.vit_classifier import create_model
from app.services.preprocessing import preprocess_image


class ModelInferenceService:
    """Service for loading model and running inference."""
    
    def __init__(self):
        self.model = None
        self.device = None
        
    def load_model(self, model_path: str = MODEL_PATH) -> None:
        """
        Load the trained model from checkpoint.
        
        Args:
            model_path: Path to saved model weights
        """
        # Determine device
        if DEVICE == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(DEVICE)
        
        print(f"Loading model on device: {self.device}")
        
        # Check if model file exists
        model_file = Path(model_path)
        if not model_file.exists():
            print(f"Warning: Model file not found at {model_path}")
            print("Creating model with random weights for testing...")
            self.model = create_model(
                model_name=VIT_MODEL_NAME,
                num_classes=NUM_CLASSES,
                pretrained=True,
                checkpoint_path=None
            )
        else:
            # Load trained model
            self.model = create_model(
                model_name=VIT_MODEL_NAME,
                num_classes=NUM_CLASSES,
                pretrained=False,
                checkpoint_path=model_path
            )
        
        # Move to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully")
    
    def predict(self, image: Image.Image) -> Tuple[str, Dict[str, float]]:
        """
        Run inference on an image.
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple of (predicted_class_name, probabilities_dict)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Preprocess image
        input_tensor = preprocess_image(image).to(self.device)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
        
        # Get predictions
        probs = probabilities[0].cpu().numpy()
        predicted_idx = probs.argmax()
        predicted_class = CLASS_NAMES[predicted_idx]
        
        # Create probability dict
        prob_dict = {
            class_name: float(probs[i])
            for i, class_name in enumerate(CLASS_NAMES)
        }
        
        return predicted_class, prob_dict

    @staticmethod
    def _patch_token_reshape(tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert ViT patch tokens into a 2D feature map.

        Expected token shape is (B, N, C), where token 0 is class token.
        Remaining tokens are reshaped to (B, C, H, W) using sqrt(N - 1).
        """
        if tokens.dim() != 3:
            raise RuntimeError("Expected transformer token activations with shape (B, N, C).")

        patch_tokens = tokens[:, 1:, :]
        patch_count = patch_tokens.shape[1]
        spatial_dim = int(np.sqrt(patch_count))
        if spatial_dim * spatial_dim != patch_count:
            raise RuntimeError("Patch token count is not a perfect square, cannot reshape for Grad-CAM.")

        feature_map = patch_tokens.reshape(tokens.shape[0], spatial_dim, spatial_dim, tokens.shape[2])
        return feature_map.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _heatmap_to_base64(cam_map: np.ndarray) -> str:
        """Colorize a normalized CAM map and return PNG base64 payload."""
        cam_uint8 = np.clip(cam_map * 255.0, 0, 255).astype(np.uint8)

        # Simple blue->green->red ramp without extra plotting deps.
        red = np.clip((cam_uint8 - 96) * 2, 0, 255)
        green = np.clip(255 - np.abs(cam_uint8.astype(np.int16) - 128) * 2, 0, 255).astype(np.uint8)
        blue = np.clip((160 - cam_uint8) * 2, 0, 255)
        rgb = np.stack([red, green, blue], axis=-1).astype(np.uint8)

        image = Image.fromarray(rgb, mode="RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def generate_grad_cam(self, image: Image.Image, target_class_idx: int | None = None) -> Dict[str, object]:
        """
        Generate a ViT Grad-CAM-like heatmap for the requested or top predicted class.

        Steps:
        1) Forward pass while capturing last-block token activations.
        2) Backward pass from the chosen class score to obtain gradients.
        3) Compute channel weights (global average over spatial gradients).
        4) Weighted sum over activations, ReLU, normalize, and upsample.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        input_tensor = preprocess_image(image).to(self.device)
        input_height = input_tensor.shape[2]
        input_width = input_tensor.shape[3]

        activations: Dict[str, torch.Tensor] = {}
        gradients: Dict[str, torch.Tensor] = {}

        target_layer = None
        backbone = getattr(self.model, "backbone", None)
        if backbone is not None and hasattr(backbone, "blocks") and len(backbone.blocks) > 0:
            target_layer = backbone.blocks[-1].norm1

        if target_layer is None:
            raise RuntimeError("Could not identify a ViT transformer block for Grad-CAM.")

        def forward_hook(_module, _inputs, output):
            activations["value"] = output

        def backward_hook(_module, _grad_input, grad_output):
            gradients["value"] = grad_output[0]

        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)

        try:
            self.model.zero_grad(set_to_none=True)
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1)

            predicted_idx = int(torch.argmax(probabilities, dim=1).item())
            selected_class_idx = predicted_idx if target_class_idx is None else int(target_class_idx)

            if selected_class_idx < 0 or selected_class_idx >= len(CLASS_NAMES):
                raise ValueError("target_class_idx is out of range for configured classes.")

            score = logits[0, selected_class_idx]
            score.backward(retain_graph=False)

            token_activations = activations.get("value")
            token_gradients = gradients.get("value")
            if token_activations is None or token_gradients is None:
                raise RuntimeError("Failed to capture transformer activations/gradients for Grad-CAM.")

            activation_map = self._patch_token_reshape(token_activations)
            gradient_map = self._patch_token_reshape(token_gradients)

            # Standard Grad-CAM weighting: average gradients per channel.
            channel_weights = gradient_map.mean(dim=(2, 3), keepdim=True)
            cam = (channel_weights * activation_map).sum(dim=1, keepdim=True)
            cam = torch.relu(cam)
            cam = F.interpolate(
                cam,
                size=(input_height, input_width),
                mode="bilinear",
                align_corners=False,
            )

            cam_np = cam[0, 0].detach().cpu().numpy()
            cam_min = float(cam_np.min())
            cam_max = float(cam_np.max())
            if cam_max - cam_min < 1e-8:
                normalized = np.zeros_like(cam_np)
            else:
                normalized = (cam_np - cam_min) / (cam_max - cam_min)

            confidence = float(probabilities[0, selected_class_idx].item())
            heatmap_b64 = self._heatmap_to_base64(normalized)

            return {
                "predicted_class": CLASS_NAMES[selected_class_idx],
                "confidence": confidence,
                "heatmap": heatmap_b64,
            }
        finally:
            forward_handle.remove()
            backward_handle.remove()


# Global inference service instance
inference_service = ModelInferenceService()


def load_model():
    """Load the model (called at startup)."""
    inference_service.load_model()


def predict(image: Image.Image) -> Dict[str, any]:
    """
    Run prediction on an image.
    
    Args:
        image: PIL Image
        
    Returns:
        Dictionary with predicted_class and probabilities
    """
    predicted_class, probabilities = inference_service.predict(image)
    
    return {
        "predicted_class": predicted_class,
        "probabilities": probabilities
    }


def generate_grad_cam(image: Image.Image, target_class_idx: int | None = None) -> Dict[str, object]:
    """Public wrapper used by API endpoint."""
    return inference_service.generate_grad_cam(image=image, target_class_idx=target_class_idx)
