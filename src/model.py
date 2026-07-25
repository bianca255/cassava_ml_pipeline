"""
model.py
--------
Model creation, training, evaluation, and retraining logic for the
Cassava Leaf Disease classifier (5-class). Uses EfficientNetB0 as a
pretrained base (transfer learning), satisfying the optimization
criteria required by the rubric (pretrained model + regularization +
early stopping + LR scheduling).
"""

import json
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.preprocessing import CLASS_NAMES

MODEL_DIR = "models"
# SavedModel format (.tf) -- a directory, not a single file. This matches the
# assignment spec's required model file types (.pkl / .tf / .h5).
MODEL_PATH = os.path.join(MODEL_DIR, "cassava_efficientnet.tf")
METRICS_PATH = os.path.join(MODEL_DIR, "latest_metrics.json")
NUM_CLASSES = len(CLASS_NAMES)


def build_model(input_shape=(224, 224, 3), fine_tune_at: int = 200) -> tf.keras.Model:
    """
    Builds an EfficientNetB0-based transfer learning model with a custom
    classification head, dropout + L2 regularization, and a partially
    fine-tuned base.
    """
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in base_model.layers[fine_tune_at:]:
        layer.trainable = True

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(model, train_gen, val_gen, epochs: int = 15):
    os.makedirs(MODEL_DIR, exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
        ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True, save_format="tf"),
    ]
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)
    return history


def evaluate_model(model, test_gen) -> dict:
    """
    Computes at least 4 evaluation metrics: accuracy, precision, recall,
    F1 (macro-averaged for multi-class), plus loss.
    """
    y_true = test_gen.classes
    y_prob = model.predict(test_gen)
    y_pred = np.argmax(y_prob, axis=1)

    loss, acc = model.evaluate(test_gen, verbose=0)

    metrics = {
        "loss": float(loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
        ),
        "evaluated_at": datetime.utcnow().isoformat(),
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def save_model_as_tf(model, path: str = MODEL_PATH) -> str:
    """Explicitly saves the model in TensorFlow SavedModel (.tf) format."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(path, save_format="tf")
    return path


def load_latest_model() -> tf.keras.Model:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run the training notebook first, "
            "or trigger /retrain via the API."
        )
    return tf.keras.models.load_model(MODEL_PATH)


def get_latest_metrics() -> dict:
    if not os.path.exists(METRICS_PATH):
        return {}
    with open(METRICS_PATH) as f:
        return json.load(f)


def retrain(epochs: int = 5):
    """
    Full retraining entry point used by the API's /retrain endpoint.
    Re-reads data/train + data/test (which now include newly uploaded
    images), fine-tunes from the existing saved model if present
    (otherwise builds fresh), evaluates, and overwrites the model file.
    """
    from src.preprocessing import get_data_generators

    train_gen, val_gen, test_gen = get_data_generators()

    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        model = build_model()

    train_model(model, train_gen, val_gen, epochs=epochs)
    save_model_as_tf(model)  # ensure final weights are persisted in .tf format
    metrics = evaluate_model(model, test_gen)
    return metrics
