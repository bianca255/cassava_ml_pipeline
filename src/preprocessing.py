"""
preprocessing.py
-----------------
Data acquisition + preprocessing utilities for the Cassava Leaf Disease
Classification pipeline.

Classes:
    0: Cassava Bacterial Blight (CBB)
    1: Cassava Brown Streak Disease (CBSD)
    2: Cassava Green Mottle (CGM)
    3: Cassava Mosaic Disease (CMD)
    4: Healthy

Used by:
- notebook/cassava_classification.ipynb (offline training)
- api/main.py (retraining trigger, on new uploaded images)
"""

import os
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_NAMES = [
    "Cassava Bacterial Blight (CBB)",
    "Cassava Brown Streak Disease (CBSD)",
    "Cassava Green Mottle (CGM)",
    "Cassava Mosaic Disease (CMD)",
    "Healthy",
]


# ---------------------------------------------------------------------------
# 1. Data acquisition
# ---------------------------------------------------------------------------
def download_dataset(dest_dir: str = "data/raw") -> str:
    """
    Downloads the Cassava Leaf Disease Classification dataset via kagglehub.
    Falls back to manual-download instructions if unavailable in the
    current environment (e.g. inside a restricted/offline container).

    Expected final structure after organizing by class folder:
        data/raw/Cassava Bacterial Blight (CBB)/*.jpg
        data/raw/Cassava Brown Streak Disease (CBSD)/*.jpg
        data/raw/Cassava Green Mottle (CGM)/*.jpg
        data/raw/Cassava Mosaic Disease (CMD)/*.jpg
        data/raw/Healthy/*.jpg
    """
    os.makedirs(dest_dir, exist_ok=True)
    try:
        import kagglehub
        path = kagglehub.dataset_download("competitions/cassava-leaf-disease-classification")
        print(f"Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(
            "Automatic download failed "
            f"({e}). Download manually from: "
            "https://www.kaggle.com/competitions/cassava-leaf-disease-classification/data "
            f"and organize images by class label into '{dest_dir}'."
        )
        return dest_dir


# ---------------------------------------------------------------------------
# 2. Train / test split on disk
# ---------------------------------------------------------------------------
def split_dataset(raw_dir: str, out_dir: str = "data", test_size: float = 0.2, seed: int = 42) -> None:
    """
    Splits class-labeled image folders into data/train/<class> and
    data/test/<class> directories.
    """
    rng = np.random.default_rng(seed)
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)

    for cls in CLASS_NAMES:
        src = raw_dir / cls
        if not src.exists():
            continue
        files = [f for f in src.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")]
        rng.shuffle(files)
        n_test = int(len(files) * test_size)
        test_files, train_files = files[:n_test], files[n_test:]

        for split_name, split_files in (("train", train_files), ("test", test_files)):
            dest = out_dir / split_name / cls
            dest.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy(f, dest / f.name)

    print(f"Split complete. Train/test folders written under '{out_dir}'.")


# ---------------------------------------------------------------------------
# 3. Keras data generators (with augmentation)
# ---------------------------------------------------------------------------
def get_data_generators(train_dir: str = "data/train", test_dir: str = "data/test"):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.15,
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        classes=CLASS_NAMES,
    )
    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        classes=CLASS_NAMES,
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
        classes=CLASS_NAMES,
    )
    return train_gen, val_gen, test_gen


# ---------------------------------------------------------------------------
# 4. Single-image preprocessing (for prediction endpoint)
# ---------------------------------------------------------------------------
def preprocess_single_image(image_path: str) -> np.ndarray:
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)


# ---------------------------------------------------------------------------
# 5. Ingest newly uploaded bulk images for retraining
# ---------------------------------------------------------------------------
def ingest_uploaded_images(upload_dir: str, label: str, train_dir: str = "data/train") -> int:
    """
    Moves newly uploaded images (already sorted by label folder name) into
    the training directory so they participate in the next retraining run.
    Returns the number of files ingested.
    """
    assert label in CLASS_NAMES, f"label must be one of {CLASS_NAMES}"
    dest = Path(train_dir) / label
    dest.mkdir(parents=True, exist_ok=True)

    count = 0
    for f in Path(upload_dir).iterdir():
        if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
            shutil.copy(f, dest / f.name)
            count += 1
    return count
