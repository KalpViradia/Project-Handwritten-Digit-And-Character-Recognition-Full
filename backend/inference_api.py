"""
FastAPI Inference API for Handwritten Recognition

REST API endpoints for predicting digits and characters from uploaded images.
Supports both file upload and base64 encoded image data.
"""

import os
import base64
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
import tensorflow as tf

# Local imports
from utils.preprocess import preprocess_image
from utils.canvas_preprocess import (
    preprocess_canvas_for_digits,
    preprocess_canvas_for_characters
)


# Character labels for A-Z
CHAR_LABELS = [chr(ord('A') + i) for i in range(26)]


# Initialize FastAPI app
app = FastAPI(
    title="Handwriting Recognition API",
    description="REST API for predicting handwritten digits and characters using CNN",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models
digit_model = None
char_model = None

DIGIT_MODEL_PATH = os.path.join("checkpoints", "cnn_best.keras")
CHAR_MODEL_PATH = os.path.join("checkpoints", "char_cnn_best.keras")


class Base64ImageRequest(BaseModel):
    """Request model for base64 encoded image."""
    image: str
    debug: bool = False  # Optional debug flag


class DigitPredictionResponse(BaseModel):
    """Response model for digit predictions."""
    model_config = ConfigDict(populate_by_name=True)

    predicted_digit: int = Field(..., alias="predictedDigit")
    confidence: float
    probabilities: list[float]
    image_grid: list[list[float]] | None = Field(None, alias="imageGrid")


class CharacterPredictionResponse(BaseModel):
    """Response model for character predictions."""
    model_config = ConfigDict(populate_by_name=True)

    predicted_character: str = Field(..., alias="predictedCharacter")
    predicted_index: int = Field(..., alias="predictedIndex")
    confidence: float
    probabilities: list[float]
    image_grid: list[list[float]] | None = Field(None, alias="imageGrid")


def load_models():
    """Load trained models on startup."""
    global digit_model, char_model
    
    # Load digit model
    digit_paths = [
        DIGIT_MODEL_PATH,
        os.path.join("checkpoints", "cnn_final.keras"),
    ]
    for path in digit_paths:
        if os.path.exists(path):
            digit_model = tf.keras.models.load_model(path)
            print(f"Digit model loaded from {path}")
            break
    
    if digit_model is None:
        print("Warning: Digit model not found. Run 'python train.py' first.")
    
    # Load character model
    char_paths = [
        CHAR_MODEL_PATH,
        os.path.join("checkpoints", "char_cnn_final.keras"),
    ]
    for path in char_paths:
        if os.path.exists(path):
            char_model = tf.keras.models.load_model(path)
            print(f"Character model loaded from {path}")
            break
    
    if char_model is None:
        print("Warning: Character model not found. Run 'python training/train_characters.py' first.")


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    load_models()


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "message": "Handwriting Recognition API",
        "version": "2.1.0",
        "status": "running",
        "models": {
            "digit_model_loaded": digit_model is not None,
            "char_model_loaded": char_model is not None,
        },
        "endpoints": {
            "digits": {
                "predict": "POST /predict - Upload image file",
                "canvas": "POST /predict/canvas - Canvas drawing",
            },
            "characters": {
                "predict": "POST /predict/character - Upload image file",
                "canvas": "POST /predict/character/canvas - Canvas drawing",
            },
            "health": "GET /health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "digit_model_loaded": digit_model is not None,
        "char_model_loaded": char_model is not None,
        "tensorflow_version": tf.__version__
    }


# ============================================================
# DIGIT PREDICTION ENDPOINTS
# ============================================================

@app.post("/predict", response_model=DigitPredictionResponse)
async def predict_digit(file: UploadFile = File(...)):
    """Predict digit from uploaded image file."""
    if digit_model is None:
        raise HTTPException(status_code=503, detail="Digit model not loaded.")
    
    try:
        image_bytes = await file.read()
        # Use standard preprocessing for uploaded images
        processed_image = preprocess_image(image_bytes)
        predictions = digit_model.predict(processed_image, verbose=0)
        
        return DigitPredictionResponse(
            predictedDigit=int(np.argmax(predictions[0])),
            confidence=float(np.max(predictions[0])),
            probabilities=predictions[0].tolist(),
            imageGrid=processed_image[0].squeeze().tolist()
        ).model_dump(by_alias=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/canvas")
async def predict_digit_canvas(request: Base64ImageRequest):
    """Predict digit from canvas drawing with robust preprocessing."""
    if digit_model is None:
        raise HTTPException(status_code=503, detail="Digit model not loaded.")
    
    try:
        # Use robust canvas preprocessing for digits
        processed_image = preprocess_canvas_for_digits(
            request.image,
            debug=request.debug
        )
        
        predictions = digit_model.predict(processed_image, verbose=0)
        
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        # Add low confidence warning
        warning = None
        if confidence < 0.6:
            warning = "Low confidence - digit may be unclear. Try redrawing."
        
        return {
            "predictedDigit": predicted_digit,
            "confidence": confidence,
            "probabilities": predictions[0].tolist(),
            "warning": warning,
            "allPredictions": [
                {"digit": i, "probability": float(p)} 
                for i, p in enumerate(predictions[0])
            ],
            "imageGrid": processed_image[0].squeeze().tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ============================================================
# CHARACTER PREDICTION ENDPOINTS
# ============================================================

@app.post("/predict/character", response_model=CharacterPredictionResponse)
async def predict_character(file: UploadFile = File(...)):
    """Predict character (A-Z) from uploaded image file."""
    if char_model is None:
        raise HTTPException(status_code=503, detail="Character model not loaded. Train with: python training/train_characters.py")
    
    try:
        image_bytes = await file.read()
        # Use standard preprocessing for uploaded images
        processed_image = preprocess_image(image_bytes)
        predictions = char_model.predict(processed_image, verbose=0)
        
        predicted_idx = int(np.argmax(predictions[0]))

        return CharacterPredictionResponse(
            predictedCharacter=CHAR_LABELS[predicted_idx],
            predictedIndex=predicted_idx,
            confidence=float(np.max(predictions[0])),
            probabilities=predictions[0].tolist(),
            imageGrid=processed_image[0].squeeze().tolist()
        ).model_dump(by_alias=True)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/character/canvas")
async def predict_character_canvas(request: Base64ImageRequest):
    """Predict character (A-Z) from canvas drawing with robust preprocessing."""
    if char_model is None:
        raise HTTPException(status_code=503, detail="Character model not loaded. Train with: python training/train_characters.py")
    
    try:
        # Use robust canvas preprocessing for EMNIST characters
        processed_image = preprocess_canvas_for_characters(
            request.image,
            debug=request.debug
        )
        
        predictions = char_model.predict(processed_image, verbose=0)
        
        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        # Add low confidence warning
        warning = None
        if confidence < 0.6:
            warning = "Low confidence - character may be unclear. Try redrawing."
        
        return {
            "predictedCharacter": CHAR_LABELS[predicted_idx],
            "predictedIndex": predicted_idx,
            "confidence": confidence,
            "probabilities": predictions[0].tolist(),
            "warning": warning,
            "allPredictions": [
                {"character": CHAR_LABELS[i], "probability": float(p)} 
                for i, p in enumerate(predictions[0])
            ],
            "imageGrid": processed_image[0].squeeze().tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ============================================================
# LEGACY ENDPOINTS (for backwards compatibility)
# ============================================================

@app.post("/predict/base64", response_model=DigitPredictionResponse)
async def predict_digit_base64(request: Base64ImageRequest):
    """Predict digit from base64 encoded image (legacy endpoint)."""
    if digit_model is None:
        raise HTTPException(status_code=503, detail="Digit model not loaded.")
    
    try:
        image_data = request.image
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]
        
        image_bytes = base64.b64decode(image_data)
        processed_image = preprocess_image(image_bytes)
        predictions = digit_model.predict(processed_image, verbose=0)
        
        return DigitPredictionResponse(
            predictedDigit=int(np.argmax(predictions[0])),
            confidence=float(np.max(predictions[0])),
            probabilities=predictions[0].tolist()
        ).model_dump(by_alias=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
