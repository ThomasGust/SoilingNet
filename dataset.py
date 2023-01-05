import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
from glob import glob
from albumentations import RandomRotate90, GridDistortion, HorizontalFlip, VerticalFlip, HueSaturationValue, GaussianBlur
import matplotlib.pyplot as plt
from kerasegmentation import resnetunet, resnetsegnet, fcn32

def load_data(path):
    images = os.listdir(os.path.join(path, "images"))
    masks = os.listdir(os.path.join(path, "labels"))
    return images, masks


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

            img_img = cv2.imread(img_path_src)
            img_label = cv2.imread(label_path_src, cv2.IMREAD_UNCHANGED)

            cv2.imwrite(img_path_dst, img_img)
            cv2.imwrite(label_path_dst, img_label)
        
        for sample in test_dataset:
            img_name, label_name = sample

            img_path_src = os.path.join(path, "images", img_name)
            label_path_src = os.path.join(path, "labels", label_name)

            img_path_dst = os.path.join(f"SplitDatasets\\Dataset{per}\\test\\images", img_name)
            label_path_dst = os.path.join(f"SplitDatasets\\Dataset{per}\\test\\labels", label_name)

            img_img = cv2.imread(img_path_src)
            img_label = cv2.imread(label_path_src, cv2.IMREAD_UNCHANGED)
            print(np.unique(img_label))

            cv2.imwrite(img_path_dst, img_img)
            cv2.imwrite(label_path_dst, img_label)

def augment_data(images, masks, save_path, augment=True):
    for x, y in tqdm(zip(images, masks), total=len(images)):
        name = x.split("/")[-1].split(".")

        image_name = name[0]
        image_extn = "png"

        name = y.split("/")[-1].split(".")
        mask_name = name[0]
        mask_extn = "png"

        x = cv2.imread(os.path.join("Dataset", "images", x), cv2.IMREAD_COLOR)
        y = cv2.imread(os.path.join("Dataset", "labels", y), cv2.IMREAD_UNCHANGED)

        if augment == True:
            aug = RandomRotate90(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented['image']
            y2 = augmented['mask']

            aug = GridDistortion(p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x5 = augmented['image']
            y5 = augmented['mask']

            aug = GaussianBlur(p=1.0)
            augmented = aug(image=x, mask=y)
            x7 = augmented['image']
            y7 = y

            save_images = [x, x2, x3, x4, x5, x7]
            save_masks =  [y, y2, y3, y4, y5, y7]

        else:
            save_images = [x]
            save_masks = [y]

        idx = 0
        for i, m in zip(save_images, save_masks):
            mask = np.moveaxis(m, 2, 0)
            mask = mask[0]
            print(np.unique(mask))
            #plt.imshow(mask)
            #plt.show()
            #i = cv2.resize(i, (224,224))
            #m = cv2.resize(m, (224, 224))

            if len(images) == 1:
                tmp_img_name = f"{image_name}.{image_extn}"
                tmp_mask_name = f"{mask_name}.{mask_extn}"

            else:
                tmp_img_name = f"{image_name}_{idx}.{image_extn}"
                tmp_mask_name = f"{mask_name}_{idx}.{mask_extn}"

            image_path = os.path.join(save_path, "images", tmp_img_name)
            mask_path = os.path.join(save_path, "labels", tmp_mask_name)

            cv2.imwrite(image_path, i)
            print("WROTE IMAGE")
            cv2.imwrite(mask_path, m)

            idx += 1

def get_labels(n_classes):
    labels = []
    irradiances = []

    file_names = os.listdir("ClassificationDataset")
    
    for name in file_names:
        smooth_label = float(name.split("_L_")[1].split("_I_")[0])
        hard_label = int(round(smooth_label*(n_classes-1)))
        labels.append(hard_label)

        irradiance = float(name.split("_I_")[1].split(".")[0])
        irradiances.append(float(int(round(irradiance*(n_classes-1)))))

    return labels, irradiances


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

    def __init__(self, n_classes, dataset_path, segmentation_model, every=10, format='PNG', transform=None):
        self.n_classes = n_classes

        self.imgs = []
        self.masks = []
        self.labels = []
        self.irradiances = []
        resnetsegnet.load_weights("segmenters_checkpoints\\segnet_20\\SEGNET.99")

        self.transform = transform

        self.names = os.listdir(dataset_path)[::every]

        """
        for i, name in enumerate(os.listdir("MASKDIR")[::every]):
            img = np.load(os.path.join("MASKDIR", f"{name}"))
            #print(np.shape(img))s
            img = np.resize(img, (1, 192, 192))
            #print(np.shape(img))
            self.masks.append(img)

            if i % 1000 == 0:
                print(f"Loaded mask {i}")
        """

        for i, name in enumerate(self.names):
            if format == 'PNG':
                img = io.imread(os.path.join(dataset_path, name))
            else:
                img = io.imread(os.path.join(dataset_path, name))

            self.img = np.moveaxis(cv2.resize(img, (224, 224)), 2, 0)
            #print(np.shape(self.img))
            self.imgs.append(img/255.0)

            smooth_label = float(name.split("_L_")[1].split("_I_")[0])
            hard_label = int(round(smooth_label*(self.n_classes-1)))

            irradiance = float(name.split("_I_")[1].split(".")[0])
            self.irradiances.append(float(int(round(irradiance*(self.n_classes-1)))))

            self.labels.append(hard_label)

            mask = resnetsegnet.predict_segmentation(os.path.join(dataset_path, name))
            self.masks.append(mask)

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


        img = np.concatenate((img, self.masks[idx]), axis=0)
        img = torch.tensor(img).float()
        
        label = torch.tensor(label)
        irradiance = torch.tensor(irradiance)

        return img, label, irradiance
    
    def save_images(self):
        os.mkdir("IMGDIR")
        
        for i, img in enumerate(self.imgs):
            cv2.imwrite(f"IMGDIR\\IMG{i}.png", img)

class DynamicSolarPanelSoilingDatasetMask(Dataset):

    def __init__(self, n_classes, dataset_path, segmentation_model, every=10, format='PNG', transform=None):
        self.n_classes = n_classes

        self.imgs = []
        self.masks = []
        self.labels = []
        self.irradiances = []

        self.transform = transform

        self.names = os.listdir(dataset_path)[::every]

        print(len(self.names))
        for i, name in enumerate(self.names):
            if format == 'PNG':
                img = io.imread(os.path.join(dataset_path, name))
            else:
                img = io.imread(os.path.join(dataset_path, name))

            self.img = cv2.resize(img, (224, 224))
            self.imgs.append(img/255.0)

            smooth_label = float(name.split("_L_")[1].split("_I_")[0])
            hard_label = int(round(smooth_label*(self.n_classes-1)))

            irradiance = float(name.split("_I_")[1].split(".")[0])
            self.irradiances.append(float(int(round(irradiance*(self.n_classes-1)))))

            self.labels.append(hard_label)

            if i % 1000 == 0:
                print(f"Loaded sample {i}")
        
        for i, name in enumerate(os.listdir("MASKDIR")[::every]):
            self.masks.append(np.load(os.path.join("MASKDIR", f"{name}")))

            if i % 1000 == 0:
                print(f"Loaded mask {i}")
        
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
        img = np.concatenate((img, self.masks[idx]), axis=0)
        
        label = torch.tensor(label)
        irradiance = torch.tensor(irradiance)

        return img, label, irradiance, torch.tensor(self.masks[idx])
    
    def save_images(self):
        os.mkdir("IMGDIR")
        
        for i, img in enumerate(self.imgs):
            cv2.imwrite(f"IMGDIR\\IMG{i}.png", img)


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