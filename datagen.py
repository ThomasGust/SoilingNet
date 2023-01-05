from dataset import DynamicSolarPanelSoilingDataset
import torchvision
from kerasegmentation import fcn32, resnetunet, resnetsegnet
import os
import cv2
import numpy as np
import shutil
import matplotlib.pyplot as plt
import pathlib
import tensorflow as tf

if __name__ == "__main__":
    shutil.rmtree("MASKDIR")
    os.mkdir("MASKDIR")
    #data = DynamicSolarPanelSoilingDataset(4, "PanelImages", None, every=25, transform=torchvision.transforms.ToTensor())

    resnetsegnet.load_weights("segmenters_checkpoints\\segnet_20\\SEGNET.99")

    names = os.listdir("PanelImages")

    for i, name in enumerate(names):

        if not pathlib.Path(F"MASKDIR\\MASK{i}.npy").exists():
            #img = cv2.imread(os.path.join("PanelImages", name))

            #img = np.expand_dims(cv2.resize(img, (224, 224)), 0)
            mask = resnetsegnet.predict_segmentation(f"PanelImages\\{name}")
            #mask = np.expand_dims(mask, 0)
            mask = cv2.resize(mask, (224,224), interpolation=cv2.INTER_NEAREST)
            mask = np.expand_dims(mask, 0)
            #print(mask.shape)
            #print(np.unique(mask))
            #mask = resnetsegnet(img)[0]
            #mask = tf.argmax(tf.math.softmax(tf.reshape(mask, (112, 112, 8)), axis=-1), axis=-1).numpy()
            #print(np.shape(mask))
            #mask = np.resize(mask, (192, 192, 1))
            #plt.imshow(mask)
            #plt.show()
            np.save(f"MASKDIR\\MASK{i}", mask)

        if i % 100 == 0:
            print(f"Generated mask for image {i}")
