"""tf.data input pipelines for the real-vs-fake dataset."""
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from keras import layers

import config


def _augmenter():
    # Kept mild on purpose: heavy warping blurs the high-frequency
    # artifacts that give generated faces away.
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.1),
    ], name="augmentation")


def load_split(split, batch_size=32, data_dir=None, training=False):
    """Load "train", "valid" or "test" as batches of (uint8-range float images, labels).

    Training data is shuffled and augmented; other splits keep a fixed order
    so predictions from different models line up for the ensemble.
    """
    data_dir = Path(data_dir or config.DATA_DIR)
    ds = keras.utils.image_dataset_from_directory(
        data_dir / split,
        labels="inferred",
        label_mode="binary",
        class_names=config.CLASS_NAMES,
        image_size=config.IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=training,
        seed=config.SEED,
    )
    if training:
        augment = _augmenter()
        ds = ds.map(lambda x, y: (augment(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def labels_of(ds):
    return np.concatenate([y.numpy().ravel() for _, y in ds]).astype(int)
