import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import shutil
import time


# pip install segments-ai
from segments import SegmentsClient, SegmentsDataset
from segments.utils import export_dataset

shutil.rmtree("Dataset\\images")
os.mkdir("Dataset\\images")

shutil.rmtree("Dataset\\labels")
os.mkdir("Dataset\\labels")
# Initialize a SegmentsDataset from the release file
client = SegmentsClient('b2cd4366770506719fc3955a8f78849ae86f82da')
release = client.get_release('AlteredCloth1/PanelImages', 'v0.2') # Alternatively: release = 'flowers-v1.0.json'
dataset = SegmentsDataset(release, labelset='ground-truth', filter_by=['labeled', 'reviewed'])

# Export to COCO panoptic format
export_dataset(dataset, export_format='semantic')



pth = "segments\\AlteredCloth1_PanelImages\\v0.2"
names = os.listdir(pth)

for name in names:
    if name.endswith('_semantic.png'):
        shutil.move(os.path.join(pth, name), f"Dataset\\labels\\{name}")
        os.rename(f"Dataset\\labels\\{name}", f"Dataset\\labels\\{name.split('_semantic')[0]}.png")
        img = cv2.imread(f"Dataset\\labels\\{name.split('_semantic')[0]}.png")
        print(np.unique(img))
    if name.endswith('.jpg'):
        try:
            shutil.move(os.path.join(pth, name), f"Dataset\\images\\{name}")
            os.rename(f"Dataset\\images\\{name}", f"Dataset\\images\\{name.split('.jpg')[0]}.png")   
        except Exception as e:
            print("EXCEPTION")    

