import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
import numpy as np
import shutil
from sklearn.model_selection import train_test_split


def tts(path, splits=[0.2, 0.5, 0.7]):
    if not os.path.exists("SplitDatasets"):
        os.mkdir("SplitDatasets")
    
    img_names = os.listdir(os.path.join(path, 'images'))
    label_names = os.listdir(os.path.join(path, "labels"))

    for i, s in enumerate(splits):

        X_train, X_test, y_train, y_test = train_test_split(img_names, label_names, random_state=314, test_size=s,shuffle=True)
        per = int(s*100)

        if os.path.exists(f"SplitDatasets\\Dataset{per}"):
            shutil.rmtree(f"SplitDatasets\\Dataset{per}")
        
        os.mkdir(f"SplitDatasets\\Dataset{per}")
        os.mkdir(f"SplitDatasets\\Dataset{per}\\train")
        os.mkdir(f"SplitDatasets\\Dataset{per}\\train\\images")
        os.mkdir(f"SplitDatasets\\Dataset{per}\\test")
        os.mkdir(f"SplitDatasets\\Dataset{per}\\test\\images")
        os.mkdir(f"SplitDatasets\\Dataset{per}\\train\\labels")
        os.mkdir(f"SplitDatasets\\Dataset{per}\\test\\labels")

        #train_dataset = training_datasets[i]
        test_dataset = zip(X_test, y_test)
        train_dataset = zip(X_train, y_train)
    
        for sample in train_dataset:
            img_name, label_name = sample

            img_path_src = os.path.join(path, "images", img_name)
            label_path_src = os.path.join(path, "labels", label_name)

            img_path_dst = os.path.join(f"SplitDatasets\\Dataset{per}\\train\\images", img_name)
            label_path_dst = os.path.join(f"SplitDatasets\\Dataset{per}\\train\\labels", label_name)

            shutil.copyfile(img_path_src, img_path_dst)
            shutil.copyfile(label_path_src, label_path_dst)
        
        for sample in test_dataset:
            img_name, label_name = sample

            img_path_src = os.path.join(path, "images", img_name)
            label_path_src = os.path.join(path, "labels", label_name)

            img_path_dst = os.path.join(f"SplitDatasets\\Dataset{per}\\test\\images", img_name)
            label_path_dst = os.path.join(f"SplitDatasets\\Dataset{per}\\test\\labels", label_name)

            shutil.copyfile(img_path_src, img_path_dst)
            shutil.copyfile(label_path_src, label_path_dst)


class SolarPanelSoilingDataset(Dataset):

    def __init__(self, dataset_path, transform=None):
        #self.imgs_path = os.path.join(dataset_path, "Images")
        self.imgs_path = dataset_path
        self.transform = transform

        self.img_names = os.listdir(self.imgs_path)

        self.imgs = []
        self.labels = []

        for i, name in enumerate(self.img_names):
            img = io.imread(os.path.join(self.imgs_path, name), plugin='matplotlib')
            #img = io.imread(os.path.join(self.imgs_path, name))
            self.imgs.append(img)

            label = int(str(name.split("_DIS_")[1].split(".j")[0]))
            #label = int(str(name.split("_DIS_")[1].split(".p")[0]))
            self.labels.append(label) 

            if i % 1000 == 0:
                print(np.max(np.array(img)))
                print(f"Loaded pair {i} from dataset")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        x = self.imgs[index]
        y = self.labels[index]

        y = torch.tensor(y)

        if self.transform:
            x = self.transform(x)
        
        return x, y

class DynamicSolarPanelSoilingDataset(Dataset):

    def __init__(self, n_classes, dataset_path, every=10, format='PNG', transform=None):
        self.n_classes = n_classes

        self.imgs = []
        self.labels = []
        self.irradiances = []

        self.transform = transform

        self.names = os.listdir(dataset_path)[::every]

        print(len(self.names))
        for i, name in enumerate(self.names):
            if format == 'PNG':
            #img = io.imread(os.path.join(dataset_path, name), plugin='matplotlib')
                img = io.imread(os.path.join(dataset_path, name))
            else:
                img = io.imread(os.path.join(dataset_path, name))
            
            self.imgs.append(img/255.0)

            smooth_label = float(name.split("_L_")[1].split("_I_")[0])
            hard_label = int(round(smooth_label*(self.n_classes-1)))

            irradiance = float(name.split("_I_")[1].split(".")[0])
            self.irradiances.append(float(int(round(irradiance*(self.n_classes-1)))))

            self.labels.append(hard_label)

            if i % 1000 == 0:
                print(f"Loaded sample {i}")
        print("DATASET LOADED")
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        irradiance = self.irradiances[idx]

        if self.transform:
            img = self.transform(img)
        img = img.float()
        
        label = torch.tensor(label)
        irradiance = torch.tensor(irradiance)

        return img, label, irradiance


class PartDynamicSolarPanelSoilingDataset(Dataset):

    def __init__(self, n_classes, dataset_path, every=10, format='PNG', transform=None):
        self.n_classes = n_classes

        self.imgs = []
        self.labels = []
        self.irradiances = []

        self.transform = transform

        self.names = os.listdir(dataset_path)[::every]

        print(len(self.names))
        for i, name in enumerate(self.names):
            smooth_label = float(name.split("_L_")[1].split("_I_")[0])
            hard_label = int(round(smooth_label*(self.n_classes-1)))

            irradiance = float(name.split("_I_")[1].split(".")[0])
            self.irradiances.append(float(int(round(irradiance*(self.n_classes-1)))))

            self.labels.append(hard_label)

            if i % 1000 == 0:
                print(f"Loaded sample {i}")
        print("DATASET LOADED")
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        irradiance = self.irradiances[idx]
        
        label = torch.tensor(label)
        irradiance = torch.tensor(irradiance)

        return label, irradiance




            
    
def discretize_power_loss(n):
    pth = "Solar_Panel_Soiling_Image_dataset\\PanelImages"
    labels = []

    names = os.listdir(pth)
    for name in names:
        p = os.path.join(pth, name)
        loss = float(str(str(p.split("_L_")[1]).split("_I_")[0]))
        labels.append(loss)
    print(names[0])
    
    nlabels = []
    for l in labels:
        nlabels.append(int(round((l*(n-1)))))
    print(nlabels[0])
    
    for i, name in enumerate(names):
        n = os.path.join(pth, name)

        os.rename(n, os.path.join(pth, f"{name}_DIS_{nlabels[i]}.jpg"))

#discretize_power_loss(5)