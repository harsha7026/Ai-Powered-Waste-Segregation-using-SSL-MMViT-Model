# Backend - AI Waste Segregation

FastAPI backend for waste classification using Vision Transformers.

## Setup

### 1. Create Virtual Environment

```bash
cd backend
python -m venv venv
```

### 2. Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration settings
│   ├── models/
│   │   ├── __init__.py
│   │   └── vit_classifier.py  # ViT model definition
│   ├── services/
│   │   ├── __init__.py
│   │   ├── preprocessing.py   # Image preprocessing
│   │   ├── inference.py       # Model inference
│   │   └── training.py        # Training scaffolding
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── prediction.py      # Pydantic schemas
│   └── utils/
│       ├── __init__.py
│       └── logging.py         # Logger setup
├── models/                  # Saved model weights (.pt files)
├── data/                    # Training data (gitignored)
│   └── train/
│       ├── organic/
│       ├── plastic/
│       ├── paper/
│       └── metal/
└── requirements.txt
```

## Running the Server

### Development Mode

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### Health Check
```
GET /api/health
```

Response:
```json
{
  "status": "ok"
}
```

### Predict Waste Category
```
POST /api/predict
Content-Type: multipart/form-data
```

Request:
- `file`: Image file (JPEG/PNG, max 10MB)

Response:
```json
{
  "predicted_class": "plastic",
  "probabilities": {
    "organic": 0.05,
    "plastic": 0.87,
    "paper": 0.03,
    "metal": 0.05
  }
}
```

## Training Your Model

### 1. Prepare Dataset

Organize your data in the following structure:

```
backend/data/train/
├── organic/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── plastic/
│   ├── img1.jpg
│   └── ...
├── paper/
│   └── ...
└── metal/
    └── ...
```

### 2. Run Training

```python
from app.services.training import train_model

train_model(
    data_dir="data/train",
    num_epochs=10,
    batch_size=32,
    learning_rate=1e-4,
    freeze_backbone=True
)
```

Or run directly:
```bash
python -m app.services.training
```

The trained model will be saved to `backend/models/waste_classifier.pt`

## Configuration

Edit `app/config.py` to customize:

- `MODEL_PATH`: Path to saved model weights
- `CLASS_NAMES`: List of waste categories
- `IMAGE_SIZE`: Input image size for the model
- `VIT_MODEL_NAME`: Vision Transformer architecture (from timm)

### Environment Variables

You can also use environment variables:

```bash
export MODEL_PATH=/path/to/model.pt
export VIT_MODEL_NAME=vit_base_patch16_224
export USE_CUDA=true
```

## Model Architecture

The model uses:
- **Backbone**: Vision Transformer (ViT) from timm library
- **Pretraining**: ImageNet or self-supervised (DINOv2/MAE)
- **Head**: Custom classification layer for 4 waste categories

You can easily swap the backbone by changing `VIT_MODEL_NAME` in config.

### Supported ViT Models

- `vit_base_patch16_224` (default)
- `vit_large_patch16_224`
- `dino_vitbase16` (DINOv2)
- `vit_base_patch16_224.mae` (MAE pretrained)

See [timm documentation](https://github.com/huggingface/pytorch-image-models) for more options.

## Testing

Test the API with curl:

```bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.jpg"
```

## Troubleshooting

### Model Not Found
If you see "Model file not found", the API will create a model with random weights for testing. Train a model or download pretrained weights to `backend/models/waste_classifier.pt`.

### CUDA Out of Memory
Reduce batch size in training or use CPU by setting `USE_CUDA=false`.

### Import Errors
Make sure you're in the backend directory and the virtual environment is activated.
