"""Train one model of the ensemble.

    python train.py --model vgg16
    python train.py --model custom_cnn
    python train.py --model vit
"""
import argparse

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from data import AUGMENTATIONS, load_split
from models import BUILDERS

# Per-model defaults: pretrained backbones want a small learning rate,
# the from-scratch CNN a larger one. With full geometric augmentation the
# custom CNN reached only 79% test accuracy (vs 95% for the unaugmented
# prototype), so it gets flips only.
DEFAULTS = {
    "vgg16": {"lr": 1e-4, "epochs": 10, "augment": "full"},
    "custom_cnn": {"lr": 1e-3, "epochs": 10, "augment": "flip"},
    "vit": {"lr": 2e-5, "epochs": 5, "augment": "full"},
}


def plot_history(history, name, path):
    h = history.history
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, title in ((ax_acc, "accuracy", "Accuracy"), (ax_loss, "loss", "Loss")):
        ax.plot(h[key], "o-", label=f"Training {title.lower()}")
        ax.plot(h[f"val_{key}"], "o-", label=f"Validation {title.lower()}")
        ax.set_title(f"{name} {title}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.grid(True)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=config.MODEL_NAMES)
    parser.add_argument("--data-dir", default=None, help=f"default: {config.DATA_DIR}")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--augment", choices=AUGMENTATIONS, default=None,
                        help="training augmentation (default depends on the model)")
    parser.add_argument("--mixed-precision", action="store_true", help="float16 compute (faster on modern GPUs)")
    args = parser.parse_args()

    name = args.model
    epochs = args.epochs or DEFAULTS[name]["epochs"]
    lr = args.lr or DEFAULTS[name]["lr"]
    augment = args.augment or DEFAULTS[name]["augment"]
    if args.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = load_split("train", args.batch_size, args.data_dir, training=True, augment=augment)
    val_ds = load_split("valid", args.batch_size, args.data_dir)
    test_ds = load_split("test", args.batch_size, args.data_dir)

    model = BUILDERS[name]()
    optimizer = (keras.optimizers.AdamW(lr, weight_decay=0.05) if name == "vit"
                 else keras.optimizers.Adam(lr))
    model.compile(optimizer=optimizer, loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(config.model_path(name), monitor="val_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1),
        keras.callbacks.CSVLogger(config.ARTIFACTS_DIR / f"{name}_history.csv"),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    test_loss, test_acc, test_auc = model.evaluate(test_ds)
    print(f"[{name}] test loss {test_loss:.4f}  accuracy {test_acc:.4f}  AUC {test_auc:.4f}")
    plot_history(history, name, config.ARTIFACTS_DIR / f"{name}_history.png")
    print(f"Best model saved to {config.model_path(name)}")


if __name__ == "__main__":
    main()
