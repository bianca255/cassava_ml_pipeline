"""
prediction.py
-------------
Single-image prediction utilities used by both the API's /predict
endpoint and the evaluation notebook.
"""

import numpy as np

from src.preprocessing import preprocess_single_image, CLASS_NAMES
from src.model import load_latest_model

_model_cache = None


def _get_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = load_latest_model()
    return _model_cache


def reset_model_cache():
    """Call this after retraining so the API picks up the new weights."""
    global _model_cache
    _model_cache = None


def predict_image(image_path: str) -> dict:
    """
    Runs inference on a single leaf image and returns the predicted
    class label plus per-class confidence scores.
    """
    model = _get_model()
    x = preprocess_single_image(image_path)
    probs = model.predict(x, verbose=0)[0]

    predicted_idx = int(np.argmax(probs))
    return {
        "predicted_class": CLASS_NAMES[predicted_idx],
        "confidence": float(probs[predicted_idx]),
        "all_class_probabilities": {
            CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
        },
    }
