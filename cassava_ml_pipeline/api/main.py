"""
api/main.py
-----------
FastAPI backend for the Cassava Leaf Disease classification pipeline.

Endpoints:
    GET  /                 -> health check
    GET  /uptime            -> model/service uptime info
    POST /predict            -> single-image prediction
    POST /upload              -> bulk image upload for a given class label (for retraining)
    POST /retrain              -> trigger retraining on all currently held training data
    GET  /metrics                -> latest evaluation metrics (for UI dashboards)
    GET  /visualizations/class-distribution -> class balance data for charting
    GET  /visualizations/image-stats         -> basic per-class image statistics
"""

import os
import shutil
import time
import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np

from src.preprocessing import CLASS_NAMES, ingest_uploaded_images
from src.model import get_latest_metrics, retrain as retrain_model
from src.prediction import predict_image, reset_model_cache

APP_START_TIME = time.time()

app = FastAPI(
    title="Cassava Leaf Disease Classification API",
    description="ML pipeline API: predict, bulk upload, and trigger retraining.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_TMP_DIR = "data/uploads_tmp"
os.makedirs(UPLOAD_TMP_DIR, exist_ok=True)

# In-memory retraining job status (simple demo-scale tracker)
retrain_status = {"state": "idle", "started_at": None, "finished_at": None, "metrics": None, "error": None}


# ---------------------------------------------------------------------------
# Health / uptime
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "service": "cassava-classifier-api"}


@app.get("/uptime")
def uptime():
    seconds = time.time() - APP_START_TIME
    return {
        "uptime_seconds": round(seconds, 2),
        "uptime_human": f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m {int(seconds % 60)}s",
        "started_at": APP_START_TIME,
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    tmp_path = os.path.join(UPLOAD_TMP_DIR, f"{uuid.uuid4().hex}_{file.filename}")
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = predict_image(tmp_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    finally:
        os.remove(tmp_path)

    return result


# ---------------------------------------------------------------------------
# Bulk upload (for retraining)
# ---------------------------------------------------------------------------
@app.post("/upload")
async def upload_bulk(label: str = Form(...), files: List[UploadFile] = File(...)):
    if label not in CLASS_NAMES:
        raise HTTPException(status_code=400, detail=f"label must be one of {CLASS_NAMES}")

    batch_dir = os.path.join(UPLOAD_TMP_DIR, f"batch_{uuid.uuid4().hex}")
    os.makedirs(batch_dir, exist_ok=True)

    for f in files:
        dest = os.path.join(batch_dir, f.filename)
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)

    n = ingest_uploaded_images(batch_dir, label)
    shutil.rmtree(batch_dir, ignore_errors=True)

    return {"ingested": n, "label": label, "message": "Images saved to training set. Trigger /retrain when ready."}


# ---------------------------------------------------------------------------
# Retraining trigger
# ---------------------------------------------------------------------------
def _run_retrain(epochs: int):
    retrain_status.update(state="running", started_at=time.time(), error=None)
    try:
        metrics = retrain_model(epochs=epochs)
        reset_model_cache()
        retrain_status.update(state="idle", finished_at=time.time(), metrics=metrics)
    except Exception as e:
        retrain_status.update(state="failed", finished_at=time.time(), error=str(e))


@app.post("/retrain")
def trigger_retrain(background_tasks: BackgroundTasks, epochs: int = 5):
    if retrain_status["state"] == "running":
        raise HTTPException(status_code=409, detail="A retraining job is already running.")
    background_tasks.add_task(_run_retrain, epochs)
    retrain_status.update(state="queued")
    return {"message": "Retraining triggered.", "epochs": epochs}


@app.get("/retrain/status")
def retrain_job_status():
    return retrain_status


# ---------------------------------------------------------------------------
# Metrics / visualizations
# ---------------------------------------------------------------------------
@app.get("/metrics")
def metrics():
    m = get_latest_metrics()
    if not m:
        raise HTTPException(status_code=404, detail="No evaluation metrics found yet.")
    return m


@app.get("/visualizations/class-distribution")
def class_distribution():
    counts = {}
    for cls in CLASS_NAMES:
        d = Path("data/train") / cls
        counts[cls] = len(list(d.glob("*.*"))) if d.exists() else 0
    return counts


@app.get("/visualizations/image-stats")
def image_stats(sample_size: int = 20):
    """
    Samples a few images per class and reports average brightness and
    resolution -- basic feature-level insight for the UI dashboard.
    """
    stats = {}
    for cls in CLASS_NAMES:
        d = Path("data/train") / cls
        if not d.exists():
            continue
        files = list(d.glob("*.*"))[:sample_size]
        brightness_vals, widths, heights = [], [], []
        for f in files:
            try:
                img = Image.open(f).convert("L")
                arr = np.array(img)
                brightness_vals.append(float(arr.mean()))
                widths.append(img.width)
                heights.append(img.height)
            except Exception:
                continue
        if brightness_vals:
            stats[cls] = {
                "avg_brightness": round(float(np.mean(brightness_vals)), 2),
                "avg_width": round(float(np.mean(widths)), 1),
                "avg_height": round(float(np.mean(heights)), 1),
                "sampled": len(files),
            }
    return stats
