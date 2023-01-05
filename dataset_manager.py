import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from kerasegmentation import resnetsegnet, resnetunet, fcn32, put_pallete
from keras.utils import plot_model
import random
import torch

names = ["FCN32-20", "SegNet-20", "UNet-20"]

for name in names:
    with open(os.path.join("training_stats", name, 'accuracies.pkl'), "rb") as f:
        accuracies = pkl.load(f)
        accuracy = str(accuracies[-1]*100)
        print(accuracy)
    with open(os.path.join("training_stats", name, 'ious.pkl'), "rb") as f:
        ious = pkl.load(f)
        iou = str(ious[-1]*100)
        print(iou)
    print()

resnetsegnet.load_weights("segmenters_checkpoints\\segnet_20\\SEGNET.99")
resnetunet.load_weights("segmenters_checkpoints\\unet_20\\UNET.99")
fcn32.load_weights("segmenters_checkpoints\\fcn32_20\\FCN32.99")

def random_colorize(i, mod, n):
    num = random.randint(0, len(os.listdir("PanelImages")))

    img_path = os.path.join("PanelImages", os.listdir("PanelImages")[num])
    src_img = cv2.imread(img_path)
    mask = mod.predict_segmentation(img_path)
    print(np.unique(mask))
    cv2.imwrite(f"{n}IMG{i}.png", src_img)
    
    put_pallete(mask, f"{n}OUT{i}")
    

models = [(fcn32, "FCN32"), (resnetsegnet, "SEGNET"), (resnetunet, "UNET")]

for mod, n in models:
    for i in range(10):
        random_colorize(i, mod, n)