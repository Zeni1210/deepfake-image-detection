"""Shared settings for training, ensembling and inference."""
import os
from pathlib import Path

# Folder containing train/ valid/ test/, each with fake/ and real/ subfolders
# (layout of the Kaggle "140k Real and Fake Faces" dataset).
DATA_DIR = Path(os.environ.get(
    "DEEPFAKE_DATA_DIR",
    "/kaggle/input/140k-real-and-fake-faces/real_vs_fake/real-vs-fake",
))
ARTIFACTS_DIR = Path(os.environ.get("DEEPFAKE_ARTIFACTS_DIR", "artifacts"))

# Every model takes raw 0-255 RGB images of this size and does its own
# resizing/normalisation internally, so one input pipeline feeds all of them.
IMAGE_SIZE = (256, 256)

# Alphabetical order, as used by directory loading: label 0 = fake, 1 = real.
# Every model outputs P(real).
CLASS_NAMES = ["fake", "real"]

MODEL_NAMES = ("vgg16", "custom_cnn", "vit")
SEED = 42

ENSEMBLE_WEIGHTS_PATH = ARTIFACTS_DIR / "ensemble_weights.json"


def model_path(name):
    return ARTIFACTS_DIR / f"{name}.keras"
