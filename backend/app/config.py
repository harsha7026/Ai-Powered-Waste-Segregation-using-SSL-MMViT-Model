import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model configuration
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "waste_classifier.pt"))
CLASS_NAMES = ["glass", "metal", "organic", "paper", "plastic"]
NUM_CLASSES = len(CLASS_NAMES)

# Image processing
IMAGE_SIZE = 224
IMAGE_MEAN = [0.485, 0.456, 0.406]  # ImageNet stats
IMAGE_STD = [0.229, 0.224, 0.225]

# Model architecture
VIT_MODEL_NAME = os.getenv("VIT_MODEL_NAME", "vit_base_patch16_224")
PRETRAINED = True

# API configuration
API_PREFIX = "/api"
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB

# Device configuration
DEVICE = "cuda" if os.getenv("USE_CUDA", "auto") == "true" else "auto"

# Database configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "waste_segregation")

# Build database URL
DATABASE_URL = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
