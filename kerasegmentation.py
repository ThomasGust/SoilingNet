from keras_segmentation.models.unet import vgg_unet
from keras_segmentation.models.unet import resnet50_unet
from keras_segmentation.models.segnet import resnet50_segnet
from keras_segmentation.models.pspnet import pspnet
import matplotlib.pyplot as plt
from dataset import DynamicSolarPanelDustClassificationDataset
from torch.utils.data import DataLoader, Dataset
import numpy as np
import tensorflow as tf
import torch.nn.functional as F
import torch.nn as nn
import torch
import torchmetrics
import torchvision.transforms as transforms
from PIL import Image
import cv2
import keras_segmentation

vgg = vgg_unet(n_classes=7, input_height=224, input_width=224)
resnetunet = resnet50_unet(7, 224, 224)
resnetsegnet = resnet50_segnet(7, 224, 224)


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
    train_dataset = DynamicSolarPanelDustClassificationDataset(imgs_path="C:\\Users\\Thomas\\OneDrive\\Apps\\Documents\\Visual studio code projects\\SolarPanelResearchProject\\DatasetTwo\\train\\Images", seg_labels_path="C:\\Users\\Thomas\\OneDrive\\Apps\\Documents\\Visual studio code projects\\SolarPanelResearchProject\\DatasetTwo\\train\\ProcessedLabels")
    trainloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    test_dataset = DynamicSolarPanelDustClassificationDataset("C:\\Users\\Thomas\\OneDrive\\Apps\\Documents\\Visual studio code projects\\SolarPanelResearchProject\\DatasetTwo\\test\\Images", seg_labels_path="C:\\Users\\Thomas\\OneDrive\\Apps\\Documents\\Visual studio code projects\\SolarPanelResearchProject\\DatasetTwo\\test\\ProcessedLabels")
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

vgg.train(train_images="DatasetTwo\\train\\Images",train_annotations="DatasetTwo\\train\\ProcessedLabels",checkpoints_path="segmenters_checkpoints\\vgg_unet_1\\VGG", epochs=epochs)

resnetunet.train(train_images="DatasetTwo\\train\\Images", train_annotations="DatasetTwo\\train\\ProcessedLabels", checkpoints_path="segmenters_checkpoints\\resnetunet2\\RESUNET", epochs=epochs)

resnetsegnet.train(train_images="DatasetTwo\\train\\Images",
            train_annotations="DatasetTwo\\train\\ProcessedLabels",
            checkpoints_path="segmenters_checkpoints\\segunet1\\SEGUNET", epochs=epochs)


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

#vgg.load_weights('segmenters_checkpoints\\vgg_unet_1\\VGG.32')
#resnetunet.load_weights('segmenters_checkpoints\\resnetunet2\\RESUNET.35')
#resnetsegnet.load_weights('segmenters_checkpoints\\segunet1\\SEGUNET.23')

"""
for i in range(4):
    img = resnetunet.predict_segmentation(inp=f"test{i+1}.png", out_fname=f'out{i+1}.png')
    print(np.unique(img))
    put_pallete(img, f"out{i+1}")
"""

"""
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #cldstrain,cldstest = build_dataset(model)
    #torch.save(cldstrain, "cldstrain")
    #torch.save(cldstest, "cldstest")
    cldstrain = torch.load("cldstrain")
    cldstest = torch.load("cldstest")
    cldstrain = DataLoader(cldstrain, 128)
    cldstest = DataLoader(cldstest, 128)
    convnet = ConvNet2().to(device)

    accuracy = torchmetrics.Accuracy(num_classes=6).to(device)
    end_criterion = nn.CrossEntropyLoss().to(device)
    optim = torch.optim.SGD(convnet.parameters(), lr=0.01)
    num_epochs = 300

    accuracies = []
    
    for epoch in range(num_epochs):
        a = []
        for i, sample in enumerate(cldstrain):
            if i > 1:
                pass
            else:
                sp, imgs, pl = sample
                sp = sp.to(device)
                pl = pl.to(device)
                img = imgs.to(device)
                img = torch.squeeze(img, 1)
                #sp  = torch.squeeze(sp, 1)

                model_preds = convnet(sp)
                pl = pl.squeeze(1)
                loss = end_criterion(model_preds, pl)
                a.append(accuracy(model_preds, pl))

                optim.zero_grad()
                loss.backward()
                optim.step()

        t = []
        for i, sample in enumerate(cldstest):
            if i > 1:
                pass
            else:
                sp, imgs, pl = sample
                sp = sp.to(device)
                pl = pl.to(device)
                img = imgs.to(device)
                img = torch.squeeze(img, 1)
                #sp = torch.squeeze(sp, 1)

                model_preds = convnet(sp)

                pl = torch.squeeze(pl, 1)
                model_preds = model_preds.float()
                loss = end_criterion(model_preds, pl)
                t.append(accuracy(model_preds, pl))
        print()
        print(sum(a)/len(a))
        print(sum(t)/len(t))
        print()
"""