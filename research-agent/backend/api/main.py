"""
api/main.py — FastAPI application entry point
"""
import logging
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from api.routers import research, models
from core.config import get_settings

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log      = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    log.info(f"GPU: {'available — ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available'}")
    yield
    log.info("Shutting down")


app = FastAPI(
    title       = settings.APP_NAME,
    version     = settings.APP_VERSION,
    description = "Production Research Agent powered by fine-tuned Qwen SLM + Tavily",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = settings.CORS_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

app.include_router(research.router, prefix="/api/v1")
app.include_router(models.router,   prefix="/api/v1")


@app.get("/", tags=["Root"])
def root():
    return {
        "name":    settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs":    "/docs",
        "status":  "running",
    }