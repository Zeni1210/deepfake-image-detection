"""Score every model and the ensemble on images from another source.

The models are trained only on StyleGAN faces vs. FFHQ photos, so a test
set built from a different generator or manipulation method shows whether
they learned "fake" in general or just StyleGAN's fingerprints.

    python evaluate.py --name ciplab \
        --real-dir /path/to/real --fake-dir /path/to/fake

Either folder may be omitted (e.g. a fakes-only dataset); AUC then can't be
computed, but per-class accuracy still can. Run ensemble.py first.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

import config
from ensemble import combine
from models import load_trained

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder, limit):
    paths = sorted(str(p) for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    return paths[:limit] if limit else paths


def load_dataset(paths, batch_size):
    def load(path):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        return tf.image.resize(img, config.IMAGE_SIZE)

    return (tf.data.Dataset.from_tensor_slices(paths)
            .map(load, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE))


def summarise(y, p):
    pred = (p >= 0.5).astype(int)
    row = {"accuracy": float((pred == y).mean())}
    for label, name in enumerate(config.CLASS_NAMES):
        mask = y == label
        if mask.any():
            row[f"{name}_accuracy"] = float((pred[mask] == label).mean())
    if len(set(y)) == 2:
        row["auc"] = float(roc_auc_score(y, p))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="label for this test set, used in the output file name")
    parser.add_argument("--real-dir")
    parser.add_argument("--fake-dir")
    parser.add_argument("--limit", type=int, default=None, help="max images per class")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if not (args.real_dir or args.fake_dir):
        parser.error("give --real-dir and/or --fake-dir")

    paths, labels = [], []
    for folder, label in ((args.fake_dir, 0), (args.real_dir, 1)):
        if folder:
            found = list_images(folder, args.limit)
            if not found:
                parser.error(f"no images found in {folder}")
            paths += found
            labels += [label] * len(found)
    y = np.array(labels)
    print(f"{args.name}: {int((y == 1).sum())} real, {int((y == 0).sum())} fake images")

    ensemble = json.loads(config.ENSEMBLE_WEIGHTS_PATH.read_text())
    names, weights = ensemble["models"], ensemble["weights"]
    ds = load_dataset(paths, args.batch_size)
    probs = [load_trained(name).predict(ds, verbose=0).ravel() for name in names]

    rows = {name: summarise(y, p) for name, p in zip(names, probs)}
    rows["ENSEMBLE"] = summarise(y, combine(probs, weights))

    columns = [c for c in ("accuracy", "real_accuracy", "fake_accuracy", "auc") if c in rows["ENSEMBLE"]]
    print(f"\n{args.name:<12}" + "".join(f"{c:>15}" for c in columns))
    for name, row in rows.items():
        print(f"{name:<12}" + "".join(f"{row[c]:>15.4f}" for c in columns))

    out = config.ARTIFACTS_DIR / f"eval_{args.name}.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
