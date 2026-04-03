# AI-Powered Waste Segregation System

A full-stack web application for automated waste classification using Vision Transformers and self-supervised learning.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![React](https://img.shields.io/badge/React-18.2-61dafb)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red)

## 🎯 Project Overview

This project demonstrates a modern approach to waste classification using:
- **Self-supervised Vision Transformers** (DINOv2/MAE) for robust feature learning
- **Multi-modal inputs** via image upload or webcam capture
- **Real-time inference** with FastAPI backend
- **Interactive UI** built with React + Vite

## 🔬 Research Workflow

The repository now includes a research workflow that supports hypothesis-driven evaluation and reproducible benchmarking.

### Research Artifacts

- `research/RESEARCH_PLAN.md` - Research questions, hypotheses, and evaluation protocol
- `research/experiment_matrix.csv` - Experiment tracker template for planned and completed runs
- `research/run_research_pipeline.py` - End-to-end runner for split creation, training, evaluation, and summary generation

### Run Research Pipeline

From the project root:

```bash
# Quick pipeline for sanity checks
python research/run_research_pipeline.py --mode quick

# Full baseline training pipeline
python research/run_research_pipeline.py --mode full

# Recompute evaluation and comparison only
python research/run_research_pipeline.py --skip-train

# Multi-run research statistics (3 repeated trials)
python research/run_research_pipeline.py --mode full --trials 3
```

The pipeline generates `research/last_run_summary.md` with aggregate metrics from available result files.
It also generates trial snapshots and paper-ready statistics tables:

- `research/runs/<run_tag>/trial_XX_metrics.json`
- `research/runs/<run_tag>/aggregate_stats.json`
- `research/runs/<run_tag>/paper_results_table.md`
- `research/runs/<run_tag>/paper_results_table.csv`

Latest copies are also written to:

- `research/aggregate_stats.json`
- `research/paper_results_table.md`
- `research/paper_results_table.csv`

### Waste Categories

The model classifies waste into 4 categories:
- 🍎 **Organic** - Food waste, biodegradable materials
- ♻️ **Plastic** - Plastic bottles, containers, packaging
- 📄 **Paper** - Paper, cardboard, newspapers
- 🔩 **Metal** - Cans, metal containers

## 🏗️ Architecture

```
┌─────────────┐      HTTP/REST      ┌──────────────┐
│   React     │ ◄─────────────────► │   FastAPI    │
│  Frontend   │   JSON/FormData     │   Backend    │
└─────────────┘                     └──────────────┘
                                           │
                                           ▼
                                    ┌──────────────┐
                                    │   PyTorch    │
                                    │  ViT Model   │
                                    └──────────────┘
```

### Tech Stack

**Backend:**
- FastAPI - Modern Python web framework
- PyTorch - Deep learning framework
- timm - Vision Transformer models
- Pillow - Image processing

**Frontend:**
- React 18 - UI framework
- Vite - Build tool and dev server
- Axios - HTTP client
- react-webcam - Camera integration

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Node.js 16 or higher
- npm or yarn
- (Optional) CUDA-capable GPU for training

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd waste-segmentation-project
```

### 2. Setup Backend

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at `http://localhost:8000`

### 3. Setup Frontend

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Frontend will be available at `http://localhost:3000`

### 4. Access Application

Open your browser and navigate to `http://localhost:3000`

## 📂 Project Structure

```
waste-segmentation-project/
│
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app & endpoints
│   │   ├── config.py            # Configuration
│   │   ├── models/
│   │   │   └── vit_classifier.py  # ViT model architecture
│   │   ├── services/
│   │   │   ├── preprocessing.py   # Image preprocessing
│   │   │   ├── inference.py       # Model inference
│   │   │   └── training.py        # Training pipeline
│   │   ├── schemas/
│   │   │   └── prediction.py      # API schemas
│   │   └── utils/
│   │       └── logging.py
│   ├── models/                  # Saved model weights
│   ├── data/                    # Training data
│   ├── requirements.txt
│   └── README.md
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Main component
│   │   ├── main.jsx             # Entry point
│   │   ├── api/
│   │   │   └── client.js        # API client
│   │   ├── components/
│   │   │   ├── Navbar.jsx
│   │   │   ├── WasteClassifier.jsx
│   │   │   ├── PredictionResult.jsx
│   │   │   └── HistoryList.jsx
│   │   └── styles/
│   │       └── App.css
│   ├── package.json
│   ├── vite.config.js
│   └── README.md
│
└── README.md                    # This file
```

## 🎓 Training Your Model

### 1. Prepare Dataset

Organize your images in this structure:

```
backend/data/train/
├── organic/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── plastic/
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
    freeze_backbone=True  # Fine-tune only the classification head
)
```

Or directly:
```bash
cd backend
python -m app.services.training
```

### 3. Model Selection

The model uses Vision Transformers from the `timm` library. You can easily change the backbone in `config.py`:

```python
# Options:
VIT_MODEL_NAME = "vit_base_patch16_224"      # Standard ViT
VIT_MODEL_NAME = "vit_large_patch16_224"     # Larger ViT
VIT_MODEL_NAME = "dino_vitbase16"            # DINOv2 (self-supervised)
VIT_MODEL_NAME = "vit_base_patch16_224.mae"  # MAE pretrained
```

## 🔌 API Documentation

### Endpoints

#### Health Check
```http
GET /api/health
```

Response:
```json
{
  "status": "ok"
}
```

#### Predict Waste Category
```http
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

Interactive API docs available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🎨 Features

### Frontend

✅ **Dual Input Modes**
- Upload images from your device
- Capture photos with webcam

✅ **Real-time Results**
- Predicted class with confidence scores
- Visual probability bars for all categories
- Color-coded waste categories

✅ **History Tracking**
- Last 10 predictions stored
- Thumbnail previews with timestamps

✅ **Responsive Design**
- Works on desktop, tablet, and mobile
- Modern, clean UI with smooth animations

### Backend

✅ **Fast Inference**
- Optimized PyTorch model
- GPU acceleration support
- Sub-second predictions

✅ **Flexible Model Architecture**
- Easy backbone swapping
- Support for various ViT architectures
- Self-supervised pretraining compatible

✅ **Production Ready**
- CORS configured
- Error handling
- Request validation
- Logging

## 🛠️ Configuration

### Backend Configuration

Edit `backend/app/config.py`:

```python
MODEL_PATH = "models/waste_classifier.pt"
CLASS_NAMES = ["organic", "plastic", "paper", "metal"]
IMAGE_SIZE = 224
VIT_MODEL_NAME = "vit_base_patch16_224"
```

Environment variables:
```bash
export MODEL_PATH=/path/to/model.pt
export VIT_MODEL_NAME=vit_base_patch16_224
export USE_CUDA=true
```

### Frontend Configuration

Edit `frontend/vite.config.js` for proxy settings:

```javascript
server: {
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true
    }
  }
}
```

For production, set:
```bash
VITE_API_URL=https://your-api-url.com
```

## 📦 Deployment

### Backend Deployment

1. **Using Docker** (recommended):
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

2. **Using gunicorn**:
```bash
pip install gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Frontend Deployment

```bash
cd frontend
npm run build
# Deploy the 'dist/' folder to Netlify, Vercel, or any static host
```

## 🧪 Testing

### Test Backend API

```bash
# Health check
curl http://localhost:8000/api/health

# Predict
curl -X POST "http://localhost:8000/api/predict" \
  -F "file=@/path/to/image.jpg"
```

### Test Frontend

Open `http://localhost:3000` and:
1. Upload or capture an image
2. Click "Classify"
3. View results and history

## 🐛 Troubleshooting

### Backend Issues

**Model not loading:**
- Check if `backend/models/waste_classifier.pt` exists
- The app will use a random model for testing if weights are missing

**CUDA out of memory:**
- Reduce batch size during training
- Set `USE_CUDA=false` to use CPU

### Frontend Issues

**API connection failed:**
- Ensure backend is running on port 8000
- Check proxy configuration in `vite.config.js`
- Verify CORS settings in backend

**Webcam not working:**
- Grant camera permissions in browser
- Use HTTPS (or localhost for dev)
- Check if camera is available

## 📝 Next Steps

Now that you have the scaffolding, here's how to proceed:

### 1. Collect & Prepare Data
- [ ] Gather images for each waste category
- [ ] Organize in `backend/data/train/` structure
- [ ] Split into train/validation sets
- [ ] Consider data augmentation

### 2. Train Your Model
- [ ] Run the training script
- [ ] Monitor loss and accuracy
- [ ] Experiment with hyperparameters
- [ ] Save best model weights

### 3. Fine-tune & Optimize
- [ ] Try different ViT backbones
- [ ] Experiment with self-supervised pretraining (DINOv2, MAE)
- [ ] Implement learning rate scheduling
- [ ] Add more augmentations

### 4. Enhance Application
- [ ] Add batch prediction support
- [ ] Implement user authentication
- [ ] Add prediction confidence thresholds
- [ ] Store predictions in database
- [ ] Add analytics dashboard

### 5. Deploy
- [ ] Dockerize both frontend and backend
- [ ] Set up CI/CD pipeline
- [ ] Deploy to cloud (AWS, GCP, Azure)
- [ ] Configure domain and SSL

## 📚 Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [timm Library](https://github.com/huggingface/pytorch-image-models)
- [React Documentation](https://react.dev/)
- [Vite Documentation](https://vitejs.dev/)

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 👥 Authors

Your Name - Your Contact

## 🙏 Acknowledgments

- timm library for pretrained Vision Transformers
- FastAPI for the excellent web framework
- React and Vite teams for modern frontend tools

---

**Happy Coding! 🚀**
