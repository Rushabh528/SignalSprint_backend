import os
import tempfile
from contextlib import asynccontextmanager
from typing import List

import gdown
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from predict import load_model, predict


MODEL_PATH = "model.pkl"
FILE_ID = "1qElVvY1xryWN4J3YrLCt3lw7U88o-CM5"
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "ALLOWED_ORIGINS",
        "*",
    ).split(",")
    if origin.strip()
]
MODELS = None


def download_model() -> None:
    if os.path.exists(MODEL_PATH):
        return

    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)


def list_available_models() -> List[str]:
    return [MODEL_PATH] if os.path.exists(MODEL_PATH) else []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODELS
    download_model()
    MODELS = load_model()
    yield


app = FastAPI(
    title="Signal Sprint Predictor",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_filename": MODEL_PATH,
        "available_models": list_available_models(),
    }


@app.get("/models")
async def get_models():
    return {
        "active_model": MODEL_PATH,
        "available_models": list_available_models(),
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if MODELS is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name.")

    suffix = os.path.splitext(file.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = temp_file.name
        temp_file.write(await file.read())

    try:
        prediction = int(predict(MODELS, temp_path))
        return {
            "prediction": prediction,
            "model_filename": MODEL_PATH,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass
