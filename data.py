"""tf.data input pipelines for the real-vs-fake dataset."""
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from keras import layers

import config


AUGMENTATIONS = ("full", "flip", "none")


def _augmenter(mode):
    # Flipping moves pixels without resampling them. Rotation, shifts and zoom
    # interpolate, which blurs the pixel-level artifacts that give generated
    # faces away; pretrained backbones cope, a small from-scratch CNN doesn't.
    steps = [layers.RandomFlip("horizontal")]
    if mode == "full":
        steps += [
            layers.RandomRotation(0.05),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomZoom(0.1),
        ]
    return keras.Sequential(steps, name="augmentation")


def load_split(split, batch_size=32, data_dir=None, training=False, augment="full"):
    """Load "train", "valid" or "test" as batches of (uint8-range float images, labels).

    Training data is shuffled and augmented (`augment`: one of AUGMENTATIONS);
    other splits keep a fixed order so predictions from different models line
    up for the ensemble.
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
    if training and augment != "none":
        augmenter = _augmenter(augment)
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def labels_of(ds):
    return np.concatenate([y.numpy().ravel() for _, y in ds]).astype(int)
