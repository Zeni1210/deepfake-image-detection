"""Model definitions: fine-tuned VGG16, the custom CNN, and a Vision Transformer.

Each model takes raw 0-255 RGB images of config.IMAGE_SIZE and outputs a
single sigmoid probability that the image is real.
"""
import keras
from keras import layers

import config

VIT_PRESET = "vit_base_patch16_224_imagenet"


@keras.saving.register_keras_serializable(package="deepfake")
class VGGPreprocess(layers.Layer):
    """Caffe-style preprocessing the ImageNet VGG weights expect: RGB->BGR, subtract channel means."""

    def call(self, x):
        x = keras.ops.flip(x, axis=-1)
        return x - keras.ops.convert_to_tensor([103.939, 116.779, 123.68], dtype=x.dtype)


def _inputs():
    return keras.Input(shape=(*config.IMAGE_SIZE, 3), name="image")


def _head(x):
    return layers.Dense(1, activation="sigmoid", dtype="float32", name="p_real")(x)


def build_vgg16(unfreeze_last=5):
    """VGG16 pretrained on ImageNet with only its last `unfreeze_last` layers trainable."""
    inputs = _inputs()
    base = keras.applications.VGG16(
        weights="imagenet", include_top=False, input_shape=(*config.IMAGE_SIZE, 3))
    for layer in base.layers[:-unfreeze_last]:
        layer.trainable = False

    x = VGGPreprocess()(inputs)
    x = base(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    return keras.Model(inputs, _head(x), name="vgg16")


def build_custom_cnn():
    """Three conv blocks and a large dense layer, trained from scratch."""
    inputs = _inputs()
    x = layers.Rescaling(1.0 / 255)(inputs)
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, activation="relu")(x)
        x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1064, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    return keras.Model(inputs, _head(x), name="custom_cnn")


def build_vit(preset=VIT_PRESET):
    """ViT-B/16 pretrained on ImageNet (via KerasHub), fully fine-tuned."""
    import keras_hub

    inputs = _inputs()
    x = layers.Resizing(224, 224)(inputs)  # position embeddings are fixed to 224x224
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)  # ViT normalisation: [-1, 1]
    x = keras_hub.models.ViTBackbone.from_preset(preset)(x)  # (batch, tokens, hidden)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    return keras.Model(inputs, _head(x), name="vit")


BUILDERS = {
    "vgg16": build_vgg16,
    "custom_cnn": build_custom_cnn,
    "vit": build_vit,
}


def load_trained(name):
    if name == "vit":
        import keras_hub  # noqa: F401  registers ViTBackbone for deserialisation
    return keras.models.load_model(config.model_path(name))
