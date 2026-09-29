# DeepFake Image Detection

Detects GAN-generated (fake) face images with a weighted-average ensemble of three models: a CNN pretrained on ImageNet (VGG16), a CNN trained from scratch, and a Vision Transformer. The models look at images in different ways, so their errors overlap less than any single model's, and the ensemble is more accurate than each one alone.

---

## Models

| Model | What it is | Input handling |
|-------|------------|----------------|
| **VGG16** | ImageNet-pretrained, last 5 layers fine-tuned, GAP → Dense(1024) → Dropout → sigmoid | 256×256, caffe-style mean subtraction |
| **Custom CNN** | 3 conv/pool blocks → Dense(1064) → Dropout → sigmoid, trained from scratch | 256×256, scaled to [0, 1] |
| **ViT-B/16** | ImageNet-pretrained Vision Transformer ([KerasHub](https://keras.io/keras_hub/)), fully fine-tuned with AdamW | resized to 224×224, scaled to [-1, 1] |

Every model takes the same raw 256×256 RGB input and outputs **P(real)**. Preprocessing is built into each saved model, so a single data pipeline feeds all three.

### Ensemble

`ensemble.py` runs every trained model on the **validation** set and grid-searches weights that sum to 1 (step 0.05) to minimise log loss. It then reports each model and the ensemble on the held-out **test** set. The weights are saved to `artifacts/ensemble_weights.json`.

---

## Dataset

[140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces): 70k real faces (Flickr-Faces-HQ) and 70k StyleGAN-generated faces, split 100k / 20k / 20k for train / valid / test.

Expected layout:

```
real-vs-fake/
├── train/{fake,real}/
├── valid/{fake,real}/
└── test/{fake,real}/
```

The default path is the Kaggle mount (`/kaggle/input/140k-real-and-fake-faces/real_vs_fake/real-vs-fake`). To use a different location, set `DEEPFAKE_DATA_DIR` or pass `--data-dir`.

---

## Usage

```bash
pip install -r requirements.txt

# 1. Train each model (best checkpoint → artifacts/<model>.keras, plus a loss/accuracy plot)
python train.py --model vgg16
python train.py --model custom_cnn
python train.py --model vit --batch-size 32 --mixed-precision

# 2. Fit ensemble weights on the validation set and score everything on the test set
python ensemble.py

# 3. Classify new images
python predict.py path/to/face.jpg
```

`train.py` options: `--epochs`, `--lr`, `--batch-size`, `--mixed-precision`. Training uses early stopping and reduces the learning rate when validation loss plateaus. Default settings: VGG16 lr 1e-4 for 10 epochs, Custom CNN lr 1e-3 for 10 epochs, ViT lr 2e-5 for 5 epochs.

A GPU is strongly recommended. ViT-B/16 in particular is slow on CPU.

---

## Project structure

```
├── config.py        # paths, image size, class names
├── data.py          # tf.data pipelines + augmentation
├── models.py        # VGG16, Custom CNN, ViT definitions
├── train.py         # train one model
├── ensemble.py      # fit ensemble weights, evaluate on test set
├── predict.py       # classify individual images
└── requirements.txt
```

---

## Results

Earlier single-model runs on the test set:

| Model       | Accuracy |
|-------------|----------|
| VGG16       | 95.27%   |
| Custom CNN  | 95.02%   |
| ViT-B/16    | run `train.py --model vit` |
| **Ensemble**| run `ensemble.py` |

---

## Acknowledgments

- [DeepFakes and Beyond: A Survey of Face Manipulation and Fake Detection](https://arxiv.org/abs/2001.00179)
- [FaceForensics++: Learning to Detect Manipulated Facial Images](https://arxiv.org/abs/1901.08971)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [TensorFlow](https://www.tensorflow.org/), [Keras](https://keras.io/), [KerasHub](https://keras.io/keras_hub/), [scikit-learn](https://scikit-learn.org/)
