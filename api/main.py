"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.routes import router
from api.dependencies import get_forecaster


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Pollen Forecast API...")
    # Pre-load models at startup so first request is not slow
    get_forecaster()
    logger.info("Models loaded. API ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Pollen Forecast Stockholm API",
    description=(
        "Probabilistic pollen forecasts for Stockholm powered by a "
        "Temporal Fusion Transformer ensemble trained on CAMS + ERA5 data."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten to Vercel domain in production
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "name": "Pollen Forecast Stockholm API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": ["/api/v1/forecast", "/api/v1/current", "/api/v1/health", "/api/v1/metrics", "/api/v1/history"],
    }
