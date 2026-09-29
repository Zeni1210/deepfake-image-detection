"""Grad-CAM heatmaps: which parts of a face each model relied on.

    python gradcam.py face1.jpg face2.png ...

Writes one figure per image to artifacts/gradcam/: the input, then each
model's heatmap for the class it predicted. For the CNNs the map comes from
the last convolutional layer; for the ViT, from the final patch tokens
(one cell per 16x16 patch).
"""
import argparse
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import config
from models import load_trained
from predict import load_image


def layer_chain(model):
    """The model's layers in call order, expanding the nested VGG16 base.

    All three models are plain chains, so running these one after another
    reproduces model(x) while exposing every intermediate output.
    """
    chain = []
    for layer in model.layers:
        if isinstance(layer, keras.layers.InputLayer):
            continue
        if type(layer).__name__ == "Functional":  # keras.applications base model
            chain += layer_chain(layer)
        else:
            chain.append(layer)
    return chain


def target_index(chain):
    """Last Conv2D for CNNs; the transformer backbone for the ViT."""
    for i in reversed(range(len(chain))):
        if isinstance(chain[i], keras.layers.Conv2D) or "Backbone" in type(chain[i]).__name__:
            return i
    raise ValueError("no convolutional layer or backbone found")


def gradcam(model, image):
    """Heatmap in [0, 1] at feature-map resolution, and the model's P(real)."""
    chain = layer_chain(model)
    split = target_index(chain)
    x = tf.convert_to_tensor(image[None])
    for layer in chain[:split + 1]:
        x = layer(x)
    features = tf.cast(x, tf.float32)

    # Take gradients of the pre-sigmoid logit: on a confident prediction the
    # sigmoid saturates and its gradient vanishes, which blanks the heatmap.
    head = chain[-1]
    with tf.GradientTape() as tape:
        tape.watch(features)
        out = features
        for layer in chain[split + 1:-1]:
            out = layer(out)
        logit = (tf.matmul(tf.cast(out, tf.float32), tf.cast(head.kernel, tf.float32))
                 + tf.cast(head.bias, tf.float32))[0, 0]
        # Explain whichever class the model picked.
        score = logit if logit >= 0 else -logit
    grads = tape.gradient(score, features)[0].numpy()
    p_real = tf.sigmoid(logit)
    features = features[0].numpy()

    if features.ndim == 2:  # ViT tokens: drop the class token, lay patches out on their grid
        side = int(np.sqrt(features.shape[0] - 1))
        features = features[1:].reshape(side, side, -1)
        grads = grads[1:].reshape(side, side, -1)

    channel_weights = grads.mean(axis=(0, 1))
    cam = np.maximum((features * channel_weights).sum(axis=-1), 0)
    return cam / (cam.max() + 1e-8), float(p_real)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+")
    parser.add_argument("--models", nargs="+", choices=config.MODEL_NAMES,
                        default=[m for m in config.MODEL_NAMES if config.model_path(m).exists()])
    args = parser.parse_args()

    out_dir = config.ARTIFACTS_DIR / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)
    models = {name: load_trained(name) for name in args.models}

    for path in args.images:
        image = load_image(path)
        fig, axes = plt.subplots(1, len(models) + 1, figsize=(4 * (len(models) + 1), 4.4))
        axes[0].imshow(image.astype("uint8"))
        axes[0].set_title(Path(path).name)
        for ax, (name, model) in zip(axes[1:], models.items()):
            cam, p_real = gradcam(model, image)
            cam = tf.image.resize(cam[..., None], config.IMAGE_SIZE).numpy()[..., 0]
            label = "real" if p_real >= 0.5 else "fake"
            ax.imshow(image.astype("uint8"))
            ax.imshow(cam, cmap="jet", alpha=0.45)
            ax.set_title(f"{name}: {label} (P(real)={p_real:.2f})")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        out = out_dir / f"{Path(path).stem}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
