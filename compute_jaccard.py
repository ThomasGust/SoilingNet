from sklearn.metrics import jaccard_score
import pickle as pkl
from keras_segmentation.models.fcn import fcn_32
from keras_segmentation.models.unet import resnet50_unet
from keras_segmentation.models.segnet import resnet50_segnet
import os
from dataset import DynamicSolarPanelSoilingDatasetMask
from torch.utils.data import DataLoader
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt



def compute_jaccard():
    #dataset = DynamicSolarPanelSoilingDatasetMask(4, "PanelImages", segmentation_model=None, every=100)
    dataset = None
    dataloader = DataLoader(dataset, 64, shuffle=False)
    fcn32 = fcn_32(8, input_height=224, input_width=224)
    resnetunet = resnet50_unet(8, 224, 224)
    resnetsegnet = resnet50_segnet(8, 224, 224)

    epochs = range(0, 100)


    models = [("fcn32_20", "FCN32", fcn32), ("fcn32_50", "FCN32", fcn32), ("fcn32_70", "FCN32", fcn32), ("unet_20", "UNET", resnetunet), ("unet_50", "UNET", resnetunet), ("unet_70", "UNET", resnetunet), ("segnet_20", "SEGNET", resnetsegnet), ("segnet_50", "SEGNET", resnetsegnet), ("segnet_70", "SEGNET", resnetsegnet)]
    img_names = os.listdir("Dataset\\images")
    mask_names = os.listdir("Dataset\\labels")
    jaccard_indices = []
    for i, m in enumerate(models[:3]):
        model_jaccard_indices = []
        name, chkpt, model = m
        chkpt_path = os.path.join('segmenters_checkpoints', name, f'{chkpt}.99')

        model.load_weights(chkpt_path)
        
        print(f"{i}: LOADED")

        for i, name in enumerate(img_names):
            img = cv2.imread(os.path.join("Dataset\\images", name))
            img = cv2.resize(img, (224, 224))
            img = np.expand_dims(img, 0)
            pred = model(img)
            pred = pred[0]
            pred = np.reshape(pred, (256, 256, 8))
            pred = np.argmax(np.array(torch.softmax(torch.tensor(pred), dim=2)), axis=2)
            mask = cv2.imread(os.path.join("Dataset\\labels", mask_names[i]))
            mask = np.moveaxis(mask, 2, 0)
            mask = mask[0]
            mask  = cv2.resize(mask, (256, 256))
    
            plt.imshow(mask)
            plt.show()
            
            #mask = np.argmax(np.array(torch.softmax(torch.tensor(mask, dtype=torch.float), dim=0)), 0)
            #mask = np.resize(mask, (256, 256))

            print(mask.shape)
            print(pred.shape)

            print(np.unique(mask))
            print(np.unique(pred))
            jaccard = jaccard_score(mask, pred)
            model_jaccard_indices.append(jaccard)
        
        j = sum(model_jaccard_indices)/len(model_jaccard_indices)
        jaccard_indices.append(j)
        print(j)


if __name__ == "__main__":
    compute_jaccard()
