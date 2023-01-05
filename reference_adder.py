import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
ref = cv2.imread("clean_reference_.png")
ref = cv2.resize(ref, dsize=(192, 192))

for path in os.listdir("Dataset\\labels"):
    img = cv2.imread(os.path.join("Dataset", "labels", path))
    print(np.unique(img))
    new_img = ref + img
    print(np.unique(new_img))
    cv2.imwrite(os.path.join("Dataset", "labels", path), new_img)

