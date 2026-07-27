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
"""

import os
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input


# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


CLASS_NAMES = [
    "Cassava Bacterial Blight (CBB)",
    "Cassava Brown Streak Disease (CBSD)",
    "Cassava Green Mottle (CGM)",
    "Cassava Mosaic Disease (CMD)",
    "Healthy",
]


# ---------------------------------------------------------
# 1. Dataset download
# ---------------------------------------------------------

def download_dataset(dest_dir="data/raw"):

    os.makedirs(dest_dir, exist_ok=True)

    try:
        import kagglehub

        path = kagglehub.dataset_download(
            "competitions/cassava-leaf-disease-classification"
        )

        print("Dataset downloaded:")
        print(path)

        return path

    except Exception as e:

        print(
            "Automatic download failed.\n"
            f"Reason: {e}\n"
            "\nDownload manually from Kaggle and organize images as:"
            "\n data/raw/<class_name>/<image>.jpg"
        )

        return dest_dir



# ---------------------------------------------------------
# 2. Dataset splitting
# ---------------------------------------------------------

def split_dataset(
    raw_dir="data/raw",
    out_dir="data",
    test_size=0.2,
    seed=42
):

    rng = np.random.default_rng(seed)

    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)


    for cls in CLASS_NAMES:

        src = raw_dir / cls


        if not src.exists():
            print(f"Missing class folder: {cls}")
            continue


        images = [
            f for f in src.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]


        rng.shuffle(images)


        test_count = int(len(images) * test_size)


        test_images = images[:test_count]
        train_images = images[test_count:]


        for split, files in [
            ("train", train_images),
            ("test", test_images)
        ]:


            destination = (
                out_dir /
                split /
                cls
            )


            destination.mkdir(
                parents=True,
                exist_ok=True
            )


            for img in files:

                shutil.copy(
                    img,
                    destination / img.name
                )


    print("Dataset split complete")




# ---------------------------------------------------------
# 3. Data generators
# ---------------------------------------------------------

def get_data_generators(
        train_dir="data/train",
        test_dir="data/test"
):


    # EfficientNet preprocessing
    # IMPORTANT:
    # No rescale=1/255 here.
    # EfficientNet already handles scaling internally.

    train_datagen = ImageDataGenerator(

        preprocessing_function=preprocess_input,


        rotation_range=25,

        width_shift_range=0.15,

        height_shift_range=0.15,

        shear_range=0.1,

        zoom_range=0.2,

        horizontal_flip=True,

        vertical_flip=False,


        validation_split=0.15
    )



    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )



    train_gen = train_datagen.flow_from_directory(

        train_dir,

        target_size=IMG_SIZE,

        batch_size=BATCH_SIZE,

        class_mode="categorical",

        classes=CLASS_NAMES,

        subset="training"

    )



    val_gen = train_datagen.flow_from_directory(

        train_dir,

        target_size=IMG_SIZE,

        batch_size=BATCH_SIZE,

        class_mode="categorical",

        classes=CLASS_NAMES,

        subset="validation"

    )



    test_gen = test_datagen.flow_from_directory(

        test_dir,

        target_size=IMG_SIZE,

        batch_size=BATCH_SIZE,

        class_mode="categorical",

        classes=CLASS_NAMES,

        shuffle=False

    )



    return train_gen, val_gen, test_gen




# ---------------------------------------------------------
# 4. Single image preprocessing
# ---------------------------------------------------------

def preprocess_single_image(image_path):


    img = tf.keras.preprocessing.image.load_img(

        image_path,

        target_size=IMG_SIZE

    )


    img_array = (
        tf.keras.preprocessing.image.img_to_array(img)
    )


    # EfficientNet preprocessing
    img_array = preprocess_input(
        img_array
    )


    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    return img_array




# ---------------------------------------------------------
# 5. Upload new images for retraining
# ---------------------------------------------------------

def ingest_uploaded_images(
        upload_dir,
        label,
        train_dir="data/train"
):


    assert label in CLASS_NAMES, (
        f"Invalid label. Choose one of: {CLASS_NAMES}"
    )


    destination = (
        Path(train_dir)
        /
        label
    )


    destination.mkdir(
        parents=True,
        exist_ok=True
    )


    count = 0


    for file in Path(upload_dir).iterdir():

        if file.suffix.lower() in [
            ".jpg",
            ".jpeg",
            ".png"
        ]:


            shutil.copy(

                file,

                destination / file.name

            )


            count += 1



    return count