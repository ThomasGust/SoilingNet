from keras_segmentation.models.unet import vgg_unet
from keras_segmentation.models.unet import resnet50_unet
from keras_segmentation.models.segnet import resnet50_segnet
from keras_segmentation.models.fcn import fcn_32
from dataset import DynamicSolarPanelSoilingDataset
from torch.utils.data import DataLoader, Dataset
import numpy as np
import tensorflow as tf
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
from imgaug import augmenters as iaa
import keras_segmentation
import os
import shutil
import keras
import matplotlib.pyplot as plt
import pickle as pkl
import random
import cv2

import collections
collections.Iterable = collections.abc.Iterable



fcn32 = fcn_32(8, input_height=224, input_width=224)
resnetunet = resnet50_unet(8, 224, 224)
resnetsegnet = resnet50_segnet(8, 224, 224)


def augmentation_stack():
    return  iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        iaa.Flipud(0.5), # horizontally flip 50% of all images
    ])


class ClassifierDataset(Dataset):

    def __init__(self):
        self.seg_preds = []
        self.imgs = []
        self.pl = []
    
    def __len__(self):
        return len(self.seg_preds)

    def __getitem__(self, idx):
        return self.seg_preds[idx], self.imgs[idx], self.pl[idx]

def build_dataset(m):
    train_dataset = DynamicSolarPanelSoilingDataset(imgs_path="C:\\Users\\Thomas\\OneDrive\\Apps\\Documents\\Visual studio code projects\\SolarPanelResearchProject\\DatasetTwo\\train\\Images", seg_labels_path="C:\\Users\\Thomas\\OneDrive\\Apps\\Documents\\Visual studio code projects\\SolarPanelResearchProject\\DatasetTwo\\train\\ProcessedLabels")
    trainloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    test_dataset = DynamicSolarPanelSoilingDataset("C:\\Users\\Thomas\\OneDrive\\Apps\\Documents\\Visual studio code projects\\SolarPanelResearchProject\\DatasetTwo\\test\\Images", seg_labels_path="C:\\Users\\Thomas\\OneDrive\\Apps\\Documents\\Visual studio code projects\\SolarPanelResearchProject\\DatasetTwo\\test\\ProcessedLabels")
    testloader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=2)

    cldstrain = ClassifierDataset()
    cldstest = ClassifierDataset()

    for sample in trainloader:
        img, _, clid = sample
        cldstrain.imgs.append(img)
        cldstrain.pl.append(clid)
        
        img = F.interpolate(img, size=(224, 224))
        
        img = img.permute(0,2,3,1)
        np_tensor = img.numpy()
        img = tf.convert_to_tensor(np_tensor)

        pred = m(img)

        pred = pred.numpy()
        pred = torch.tensor(pred)
        cldstrain.seg_preds.append(pred)

    for sample in testloader:
        img, _, clid = sample
        cldstest.imgs.append(img)
        cldstest.pl.append(clid)
        
        img = F.interpolate(img, size=(224, 224))

        img = img.permute(0,2,3,1)
        np_tensor = img.numpy()
        img = tf.convert_to_tensor(np_tensor)

        pred = m(img)

        pred = pred.numpy()
        pred = torch.tensor(pred)
        cldstest.seg_preds.append(pred)
    
    return cldstrain, cldstest



epochs = 100

"""
fcn32.train(train_images="DatasetTwo\\train\\Images",train_annotations="DatasetTwo\\train\\ProcessedLabels",checkpoints_path="segmenters_checkpoints\\fcn32\\FCN32", epochs=epochs)

resnetunet.train(train_images="DatasetTwo\\train\\Images", train_annotations="DatasetTwo\\train\\ProcessedLabels", checkpoints_path="segmenters_checkpoints\\resnetunet2\\RESUNET", epochs=epochs)

resnetsegnet.train(train_images="DatasetTwo\\train\\Images",
            train_annotations="DatasetTwo\\train\\ProcessedLabels",
            checkpoints_path="segmenters_checkpoints\\segunet1\\SEGUNET", epochs=epochs)
"""

class KerasSegmentationGraphCallback(keras.callbacks.Callback):

    def __init__(self, epochs, model_name):
        super().__init__()
        self.accuracies = []
        self.losses = []
        self.model_name = model_name
        self.epoch_range = range(epochs)

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        plt.plot(self.epoch_range, self.accuracies, 'r', label=f'Training Accuracy {self.model_name}')
        plt.title(f'Training Accuracy {self.model_name}')

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.savefig(f'figures\\accuracies\\TrainingAccuracy{self.model_name}.png')
        plt.close()

        plt.plot(self.epoch_range, self.losses, 'r', label=f'Loss {self.model_name}')
        plt.title(f'Training Loss {self.model_name}')

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(f'figures\\losses\\trainingLoss{self.model_name}.png')
        plt.close()

        if not os.path.exists(os.path.join("training_stats", self.model_name)):
            os.mkdir(os.path.join("training_stats", self.model_name))
        
        with open(os.path.join("training_stats", self.model_name, "accuracies.pkl"), "wb") as f:
            pkl.dump(self.accuracies, f)

        with open(os.path.join("training_stats", self.model_name, "losses.pkl"), "wb") as f:
            pkl.dump(self.losses, f)



    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        accuracy = logs.get("accuracy")
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        print(self.accuracies)
        print(self.losses)



def train_models(epochs=epochs, splits=[0.2, 0.5, 0.7]):
    for s in splits:
        per = int(s*100)

        path = os.path.join("SplitDatasets", f"Dataset{per}")

        if not os.path.exists(f"segmenters_checkpoints\\fcn32_{per}"): 
            os.mkdir(f"segmenters_checkpoints\\fcn32_{per}")
        else:
            shutil.rmtree(f"segmenters_checkpoints\\fcn32_{per}")
            os.mkdir(f"segmenters_checkpoints\\fcn32_{per}")

        fcn32.train(train_images=os.path.join(path, "train", "images"), train_annotations=os.path.join(path, "train", "labels"), checkpoints_path=f"segmenters_checkpoints\\fcn32_{per}\\FCN32", epochs=epochs, cbs=[KerasSegmentationGraphCallback(epochs, f"FCN32-{per}")])

        if not os.path.exists(f"segmenters_checkpoints\\unet_{per}"):
            os.mkdir(f"segmenters_checkpoints\\unet_{per}")
        else:
            shutil.rmtree(f"segmenters_checkpoints\\unet_{per}")
            os.mkdir(f"segmenters_checkpoints\\unet_{per}")

        resnetunet.train(train_images=os.path.join(path, "train", "images"), train_annotations=os.path.join(path, "train", "labels"), checkpoints_path=f"segmenters_checkpoints\\unet_{per}\\UNET", epochs=epochs, cbs=[KerasSegmentationGraphCallback(epochs, f"UNet-{per}")])

        if not os.path.exists(f"segmenters_checkpoints\\segnet_{per}"):
            os.mkdir(f"segmenters_checkpoints\\segnet_{per}")
        else:
            shutil.rmtree(f"segmenters_checkpoints\\segnet_{per}")
            os.mkdir(f"segmenters_checkpoints\\segnet_{per}")
        resnetsegnet.train(train_images=os.path.join(path, "train", "images"), train_annotations=os.path.join(path, "train", "labels"), checkpoints_path=f"segmenters_checkpoints\\segnet_{per}\\SEGNET", epochs=epochs, cbs=[KerasSegmentationGraphCallback(epochs, f"SegNet-{per}")])



def masking_function(img):
    print(img.shape)
    img[img >= 1] = 1
    return img

def put_pallete(img, path):
    t = torch.tensor(img)
    img = transforms.ToPILImage()(t.byte())
    img.putpalette([0,   0,   0,
                    255, 0, 0,
                    0, 255,   0,
                    0,   0, 255,
                    255,255,0,
                    0,255,255,
                    255,255,255]) 
    filename = f"{path}.png"
    img.save(filename)

def random_colorize():
    path = "Dataset\\labels"

    names = os.listdir(path)

    name = random.choice(names)

    img = cv2.imread(os.path.join(path, name), cv2.IMREAD_GRAYSCALE)
    print(np.unique(img))

    put_pallete(img, "ColorizedLabelExample.png")


random_colorize()

"""
resnetunet.load_weights('segmenters_checkpoints\\unet_20\\UNET.99')


for i in range(4):
    img = resnetunet.predict_segmentation(inp=f"examples\\inputs\\test{i+1}.png", out_fname=f'out{i+1}.png')
    print(img.shape)
    print(type(img))
    print(np.unique(img))
    put_pallete(img, f"out{i+1}")
"""

"""
if __name__ == "__main__":
    train_models(epochs=100)
"""
