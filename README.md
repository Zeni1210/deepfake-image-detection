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

`ensemble.py` runs every trained model on the **validation** set and grid-searches weights that sum to 1 (step 0.05) to minimise log loss. It then reports each model and the ensemble on the held-out **test** set, alongside a plain equal-weight average (`EQUAL_AVG`) as a baseline. The weights are saved to `artifacts/ensemble_weights.json`.

### Does it generalise?

High accuracy on the 140k dataset alone proves little, because every fake in it comes from one generator (StyleGAN). A model can score well by memorising that generator's fingerprints and still fail on anything else. Two tools check for this:

- **Cross-dataset evaluation** (`evaluate.py`): scores every model and the ensemble on a folder of real and/or fake images from a *different* source (another GAN, a diffusion model, or hand-edited faces). Per-class accuracy shows whether errors come from missed fakes or false alarms on real photos.
- **Grad-CAM** (`gradcam.py`): heatmaps of the regions behind each model's decision, on its last conv layer for the CNNs and its final patch tokens for the ViT. A model that relies on facial regions (eyes, hair boundaries, skin texture) is more trustworthy than one that relies on backgrounds or image borders.

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

# 4. Test generalisation on images from another generator / dataset
python evaluate.py --name <test-set-name> --real-dir path/to/real --fake-dir path/to/fake

# 5. Grad-CAM heatmaps (saved to artifacts/gradcam/)
python gradcam.py path/to/face1.jpg path/to/face2.jpg
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
├── evaluate.py      # cross-dataset / cross-generator evaluation
├── gradcam.py       # Grad-CAM heatmaps for all models
└── requirements.txt
```

---

## Results

Results are pending a full training run.

**In-distribution** (140k test set, 20k images):

| Model       | Accuracy | AUC |
|-------------|----------|-----|
| VGG16       | –        | –   |
| Custom CNN  | –        | –   |
| ViT-B/16    | –        | –   |
| Equal average | –      | –   |
| **Weighted ensemble** | – | – |

**Cross-dataset** (unseen source, not used in training or weight fitting):

| Model       | Accuracy | Real acc. | Fake acc. | AUC |
|-------------|----------|-----------|-----------|-----|
| **Weighted ensemble** | – | – | – | – |

---

## Acknowledgments

- [DeepFakes and Beyond: A Survey of Face Manipulation and Fake Detection](https://arxiv.org/abs/2001.00179)
- [FaceForensics++: Learning to Detect Manipulated Facial Images](https://arxiv.org/abs/1901.08971)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [TensorFlow](https://www.tensorflow.org/), [Keras](https://keras.io/), [KerasHub](https://keras.io/keras_hub/), [scikit-learn](https://scikit-learn.org/)
