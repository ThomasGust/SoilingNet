import cv2
import numpy as np
import os

ref = cv2.imread("clean_reference_.png")
ref = cv2.resize(ref, dsize=(224, 224))
print(np.max(ref))
print(np.unique(ref))
for path in os.listdir("Dataset\\labels"):
    img = cv2.imread(os.path.join("Dataset", "labels", path))
    img = cv2.resize(img, (224, 224))
    new_img = ref + img
    cv2.imwrite(os.path.join("Dataset", "labels", path), new_img)
