"""Classify images as real or fake with the trained ensemble.

    python predict.py face1.jpg face2.png ...

Run ensemble.py first to produce the ensemble weights.
"""
import argparse
import json

import keras
import numpy as np

import config
from models import load_trained


def load_image(path):
    img = keras.utils.load_img(path, target_size=config.IMAGE_SIZE)
    return keras.utils.img_to_array(img)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+")
    args = parser.parse_args()

    ensemble = json.loads(config.ENSEMBLE_WEIGHTS_PATH.read_text())
    names, weights = ensemble["models"], ensemble["weights"]
    batch = np.stack([load_image(p) for p in args.images])

    probs = {name: load_trained(name).predict(batch, verbose=0).ravel() for name in names}
    p_real = sum(w * probs[name] for name, w in zip(names, weights))

    for i, path in enumerate(args.images):
        label = config.CLASS_NAMES[int(p_real[i] >= 0.5)]
        confidence = p_real[i] if label == "real" else 1 - p_real[i]
        detail = ", ".join(f"{name} {probs[name][i]:.3f}" for name in names)
        print(f"{path}: {label.upper()} ({100 * confidence:.2f}% confident)  [P(real): {detail}]")


if __name__ == "__main__":
    main()
