from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from contextlib import asynccontextmanager
from typing import Dict

from app.config import API_PREFIX, MAX_UPLOAD_SIZE
from app.schemas.prediction import (
    PredictionResponse,
    HealthResponse,
    GradCamResponse,
    StatsSummaryResponse,
    DisposalRulesResponse,
    DisposalRuleItem,
)
from app.services.inference import load_model, predict, generate_grad_cam
from app.services.analytics import analytics_service
from app.services.disposal_rules import disposal_rules_service
from app.utils.logging import setup_logger
from app.database import init_db, check_db_connection

# Setup logger
logger = setup_logger("waste-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup
    logger.info("Starting up...")
    try:
        load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    
    # Initialize database
    try:
        if check_db_connection():
            init_db()
            logger.info("Database initialized successfully")
        else:
            logger.warning("Database connection not available. Running without persistence.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Waste Segregation API",
    description="AI-powered waste classification using Vision Transformers",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(f"{API_PREFIX}/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Status of the API
    """
    return HealthResponse(status="ok")


@app.post(f"{API_PREFIX}/predict", response_model=PredictionResponse)
async def predict_waste(file: UploadFile = File(...)):
    """
    Predict waste category from uploaded image.
    
    Args:
        file: Uploaded image file (JPEG/PNG)
        
    Returns:
        Predicted class and probabilities for all classes
        
    Raises:
        HTTPException: If file is invalid or processing fails
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG/PNG)"
        )
    
    try:
        # Read image file
        contents = await file.read()
        
        # Check file size
        if len(contents) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum limit of {MAX_UPLOAD_SIZE / (1024*1024):.1f}MB"
            )
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(contents))
        
        # Run prediction
        result = predict(image)

        top_confidence = result["probabilities"].get(result["predicted_class"], 0.0)
        analytics_service.record_prediction(
            predicted_class=result["predicted_class"],
            confidence=top_confidence,
        )
        
        logger.info(f"Prediction: {result['predicted_class']}")
        
        return PredictionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
    finally:
        await file.close()


@app.post(f"{API_PREFIX}/grad-cam", response_model=GradCamResponse)
@app.post("/grad-cam", response_model=GradCamResponse)
async def grad_cam_visualization(
    file: UploadFile = File(...),
    target_class_idx: int | None = Form(default=None),
):
    """
    Generate ViT Grad-CAM heatmap for a selected class or top prediction.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG/PNG)")

    try:
        contents = await file.read()

        if len(contents) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum limit of {MAX_UPLOAD_SIZE / (1024*1024):.1f}MB",
            )

        image = Image.open(io.BytesIO(contents))
        result = generate_grad_cam(image=image, target_class_idx=target_class_idx)
        return GradCamResponse(**result)
    except HTTPException:
        raise
    except ValueError as exc:
        logger.error(f"Invalid grad-cam request: {exc}")
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"Grad-CAM generation failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Grad-CAM generation failed: {str(exc)}")
    finally:
        await file.close()


@app.get(f"{API_PREFIX}/stats/summary", response_model=StatsSummaryResponse)
@app.get("/stats/summary", response_model=StatsSummaryResponse)
async def get_stats_summary():
    """Return usage summary metrics for analytics dashboard."""
    summary = analytics_service.get_summary()
    return StatsSummaryResponse(**summary)


@app.get(f"{API_PREFIX}/stats/class-distribution")
@app.get("/stats/class-distribution")
async def get_class_distribution():
    """Return count distribution of predicted classes."""
    return analytics_service.get_class_distribution()


@app.get(f"{API_PREFIX}/admin/disposal-rules")
@app.get("/admin/disposal-rules")
async def get_disposal_rules():
    """
    Get current disposal rules configuration.
    
    Returns JSON object with disposal rules for each waste class.
    """
    try:
        rules = disposal_rules_service.get_rules()
        return rules
    except Exception as exc:
        logger.error(f"Failed to load disposal rules: {exc}")
        raise HTTPException(status_code=500, detail="Failed to load disposal rules")


@app.post(f"{API_PREFIX}/admin/disposal-rules")
@app.post("/admin/disposal-rules")
async def save_disposal_rules(rules: Dict[str, DisposalRuleItem]):
    """
    Save updated disposal rules configuration.
    
    Args:
        rules: Dictionary mapping class names to disposal rule objects
        
    Returns:
        Success message with saved rules
    """
    try:
        # Convert Pydantic models to dict
        rules_dict = {key: value.dict() for key, value in rules.items()}
        disposal_rules_service.save_rules(rules_dict)
        logger.info("Disposal rules updated successfully")
        return {"message": "Disposal rules saved successfully", "rules": rules_dict}
    except Exception as exc:
        logger.error(f"Failed to save disposal rules: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to save disposal rules: {str(exc)}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Waste Segregation API",
        "version": "1.0.0",
        "endpoints": {
            "health": f"{API_PREFIX}/health",
            "predict": f"{API_PREFIX}/predict"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
