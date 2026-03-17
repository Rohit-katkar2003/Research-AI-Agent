"""
api/routers/models.py — /models endpoints
"""
import torch
from fastapi import APIRouter, HTTPException
from api.schemas import ModelInfo, HealthResponse
from models.model_manager import model_manager, ModelSize, MODEL_REGISTRY
from core.config import get_settings

router   = APIRouter(prefix="/models", tags=["Models"])
settings = get_settings()


@router.get("/", response_model=list[ModelInfo])
def list_models():
    """List all available models with their status."""
    loaded = model_manager.list_loaded()
    return [
        ModelInfo(
            id          = size,
            description = info["description"],
            vram        = info["vram"],
            speed       = info["speed"],
            loaded      = size in loaded,
        )
        for size, info in model_manager.available_models.items()
    ]


@router.post("/{model_size}/load")
def load_model(model_size: str):
    """Pre-warm a model into memory."""
    try:
        size = ModelSize(model_size)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_size}")
    model_manager.load(size)
    return {"status": "loaded", "model": model_size}


@router.delete("/{model_size}/unload")
def unload_model(model_size: str):
    """Free VRAM by unloading a model."""
    try:
        size = ModelSize(model_size)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_size}")
    model_manager.unload(size)
    return {"status": "unloaded", "model": model_size}


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status        = "ok",
        version       = settings.APP_VERSION,
        models_loaded = model_manager.list_loaded(),
        gpu_available = torch.cuda.is_available(),
    )