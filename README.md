# SoilingNet
## Sorry for how messy the code is right now, I should probably clean it up :<
This repository is for the SoilingNet Project. Soiling Net is an AI model to analyze soiling and power loss on photovoltaic panels, with the ultimate goal of making solar panel maintenance easier for everyone. 

Credit to the DeepSolarEye project for the dataset we use for this project:

S. Mehta, A. P. Azad, S. A. Chemmengath, V. Raykar and S. Kalyanaraman,
[DeepSolarEye: Power Loss Prediction and Weakly Supervised Soiling Localization via Fully    Convolutional Networks for Solar Panels,](https://arxiv.org/abs/1710.03811)" 2018 IEEE Winter Conference on Applications of Computer Vision (WACV), Lake Tahoe, NV, 2018, pp. 333-342.

We also introduced a subset of the DeepSolarEye dataset with hand labeled semantic segmentation masks for this project.

SoilingNet consists of 2 sub-systems, the first is a semantic segmentation model, which we are able to train in a fully supervised manner, that predicts soiling type and distribution from images of a solar panel. The second system is a classification model that produces a prediction for soiling impact severity.
