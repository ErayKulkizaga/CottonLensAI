import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import load_runtime, router, runtime
from app.artifact_importer import import_artifact_directory
from app.config import get_settings
from app.database import SessionLocal
from app.demo import seed_demo_data


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    load_runtime()
    with SessionLocal() as db:
        if runtime.ready:
            try:
                import_artifact_directory(db, Path(settings.artifact_dir))
            except Exception:
                logging.exception("Validated runtime model loaded, but database artifact import failed")
                db.rollback()
        if settings.demo_mode and not runtime.ready:
            seed_demo_data(db)
    yield


settings = get_settings()
app = FastAPI(
    title="CottonLens AI API",
    version="0.1.0",
    description="Lightweight inference API. Model training is performed exclusively in Google Colab.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder(
            {"code": "validation_error", "message": "Request validation failed", "details": exc.errors()}
        ),
    )


app.include_router(router)
