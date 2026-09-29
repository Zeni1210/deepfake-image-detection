"""Weighted-average ensemble of the trained models.

Weights are chosen on the validation set (grid search over the simplex,
minimising log loss), then everything is scored on the untouched test set.

    python ensemble.py                       # every model in artifacts/
    python ensemble.py --models vgg16 vit    # a subset
"""
import argparse
import itertools
import json

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

import config
from data import labels_of, load_split
from models import load_trained


def combine(probs, weights):
    return sum(w * p for w, p in zip(weights, probs))


def search_weights(probs, y, step):
    """All weight vectors on a grid with spacing `step` that sum to 1; keep the lowest log loss."""
    n_steps = round(1 / step)
    best, best_loss = None, np.inf
    for grid in itertools.product(range(n_steps + 1), repeat=len(probs)):
        if sum(grid) != n_steps:
            continue
        weights = [g / n_steps for g in grid]
        loss = log_loss(y, combine(probs, weights), labels=[0, 1])
        if loss < best_loss:
            best, best_loss = weights, loss
    return best


def scores(y, p):
    return {
        "accuracy": accuracy_score(y, p >= 0.5),
        "auc": roc_auc_score(y, p),
        "log_loss": log_loss(y, p, labels=[0, 1]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=config.MODEL_NAMES,
                        default=[m for m in config.MODEL_NAMES if config.model_path(m).exists()])
    parser.add_argument("--data-dir", default=None, help=f"default: {config.DATA_DIR}")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--step", type=float, default=0.05, help="weight grid spacing")
    args = parser.parse_args()
    if len(args.models) < 2:
        parser.error(f"need at least two trained models in {config.ARTIFACTS_DIR}/, found {args.models}")

    val_ds = load_split("valid", args.batch_size, args.data_dir)
    test_ds = load_split("test", args.batch_size, args.data_dir)
    y_val, y_test = labels_of(val_ds), labels_of(test_ds)

    val_probs, test_probs = [], []
    for name in args.models:
        print(f"Predicting with {name}...")
        model = load_trained(name)
        val_probs.append(model.predict(val_ds, verbose=0).ravel())
        test_probs.append(model.predict(test_ds, verbose=0).ravel())

    weights = search_weights(val_probs, y_val, args.step)

    rows = [(name, scores(y_test, p)) for name, p in zip(args.models, test_probs)]
    rows.append(("EQUAL_AVG", scores(y_test, np.mean(test_probs, axis=0))))
    rows.append(("ENSEMBLE", scores(y_test, combine(test_probs, weights))))

    print("\nWeights (fit on validation set):")
    for name, w in zip(args.models, weights):
        print(f"  {name:<12}{w:.2f}")
    print(f"\n{'Test set':<12}{'Accuracy':>10}{'AUC':>10}{'LogLoss':>10}")
    for name, s in rows:
        print(f"{name:<12}{s['accuracy']:>10.4f}{s['auc']:>10.4f}{s['log_loss']:>10.4f}")

    config.ENSEMBLE_WEIGHTS_PATH.write_text(json.dumps({
        "models": args.models,
        "weights": weights,
        "test_scores": {name: {k: float(v) for k, v in s.items()} for name, s in rows},
    }, indent=2))
    print(f"\nSaved ensemble weights to {config.ENSEMBLE_WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
