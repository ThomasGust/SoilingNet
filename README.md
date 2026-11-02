# SoilingNet

SoilingNet analyzes dirt and debris ("soiling") on solar panels. From one photo of a panel and a solar irradiance reading, it maps where the panel is soiled and with what, then estimates how much power the soiling is costing.

<p align="center">
<img src="figures/examples/images/IMG0.png" width="150"><img src="figures/examples/images/IMG1.png" width="150"><img src="figures/examples/images/IMG2.png" width="150"><img src="figures/examples/images/IMG3.png" width="150"><img src="figures/examples/images/IMG4.png" width="150">
<br>
<img src="figures/examples/predictions/OUT0.png" width="150"><img src="figures/examples/predictions/OUT1.png" width="150"><img src="figures/examples/predictions/OUT2.png" width="150"><img src="figures/examples/predictions/OUT3.png" width="150"><img src="figures/examples/predictions/OUT4.png" width="150">
</p>

*Top: panel images. Bottom: predicted segmentation masks. Black is background, red is clean panel surface, and the other colors are soiling classes.*

I did this project in 2022–23. It was one of the first machine learning projects I took on without following a tutorial. I wrote it up as a paper with Ivan Rodriguez (Brown University): [SoilingNetResearchPaper.pdf](SoilingNetResearchPaper.pdf).

## Why

Soiling can cut a panel's output by more than 25% in very dirty conditions, and large solar farms usually clean on a fixed schedule whether panels need it or not, using billions of gallons of water a year. Knowing which panels are soiled, with what, and how badly would let farms clean only where it matters. The paper's introduction cites the sources for these figures.

## What I built

**A labeled segmentation dataset.** The [DeepSolarEye](https://arxiv.org/abs/1710.03811) dataset has 45,754 images of a solar panel labeled with power loss and irradiance, but no masks showing where the soiling is. I hand-labeled segmentation masks for 184 of those images using segments.ai, covering 8 classes: background, clean panel surface, and six soiling types (red, brown, light gray, bird droppings, snow, and white specks). Augmentation (rotation, flips, grid distortion, and Gaussian blur) expanded the set to 1,104 image–mask pairs. DeepSolarEye simulates soiling with materials like dust and talcum powder, so classes like "bird droppings" and "snow" describe how the soiling looks, not where it came from.

**Soiling segmentation.** I trained and compared three segmentation architectures: FCN-32, U-Net with a ResNet-50 encoder, and SegNet with a ResNet-50 encoder, using the [keras-segmentation](https://github.com/divamgupta/image-segmentation-keras) library. Soiling pixels are rare compared with background and clean panel, so the loss weights the soiling classes 3 times more heavily.

**Power-loss classification.** DeepSolarEye records power loss as a continuous value, which I split into 4, 8, 12, or 16 equal-width severity classes. The classifier is a small PyTorch CNN that takes the panel image and its predicted soiling mask as a 4-channel input, with the irradiance reading added before the fully connected layers. Most images have little soiling, so training uses weighted sampling to show the model each severity class about equally often.

## Results reported in the paper

Segmentation, measured as mean intersection over union (Jaccard index) with a 20% test split:

| Model | Train | Test | Parameters |
| --- | --- | --- | --- |
| FCN-32 | 57.4 | 54.5 | 70.0M |
| U-Net (ResNet-50) | 82.0 | 71.8 | 16.4M |
| SegNet (ResNet-50) | 97.7 | 72.4 | 14.9M |

Power-loss classification:

| Severity classes | Top-1 accuracy | Top-2 accuracy |
| --- | --- | --- |
| 4 | 92.5% | 98.7% |
| 8 | 84.0% | 95.0% |
| 12 | 80.2% | 91.9% |
| 16 | 75.3% | 89.4% |

Training curves, confusion matrices, and more example predictions are in [figures](figures) and the paper's appendices.

### How much to trust these numbers

Looking back at this project with more experience, the evaluation has problems that make these numbers optimistic:

- **Segmentation:** the train/test split was made after augmentation, so rotated or flipped copies of a test image can also appear in the training set.
- **Classification:** in `power_loss_classification.py`, the train/test split doesn't take effect, because the split subsets are replaced with the full dataset. The accuracies above were then computed on all 45,754 images, which include the training images. DeepSolarEye's images are frames taken seconds apart, so even the images the model didn't train on are nearly identical to ones it did.
- **Generalization:** all images come from DeepSolarEye's experimental setup with simulated soiling, so how well the models transfer to real solar farms is untested.

A fair re-evaluation would split by original image before augmenting, and hold out whole time periods of DeepSolarEye footage for the classifier.

This was still a cool project though! It's an early example of me building an AI/deep learning project without just trying to follow along with a YouTube video.

## Pretrained weights

- [Segmentation weights](https://drive.google.com/file/d/1829M9qiGXcGVNZjZ3yax8MbA9fz9fvPB/view?usp=sharing) (`segmentation_weights.pt`)
- [Classification weights](https://drive.google.com/file/d/1q7K6P6Jm1_QeO9gCmn8DvIEFUu--pd5P/view?usp=sharing) (`classifier_weights.pt`)

## Repository layout

```text
kerasegmentation.py              Segmentation training (FCN-32, U-Net, SegNet) used in the paper
power_loss_classification.py     Severity classifier training
classifier_architectures/        Classifier model definitions
dataset.py                       Dataset loading, augmentation, and splitting
generate_confusion_matrices.py   Classifier evaluation
AugmentedDataset/                Hand-labeled images and masks, plus augmented copies
SplitDatasets/                   Train/test splits of the labeled data (20%, 50%, 70% test)
PanelImages/                     DeepSolarEye images
figures/                         Training curves, confusion matrices, and example predictions
```

## Credits

The panel images and power-loss labels come from DeepSolarEye:

> S. Mehta, A. P. Azad, S. A. Chemmengath, V. Raykar and S. Kalyanaraman, "[DeepSolarEye: Power Loss Prediction and Weakly Supervised Soiling Localization via Fully Convolutional Networks for Solar Panels](https://arxiv.org/abs/1710.03811)," 2018 IEEE Winter Conference on Applications of Computer Vision (WACV), pp. 333–342.

Segmentation model implementations are from [keras-segmentation](https://github.com/divamgupta/image-segmentation-keras).
