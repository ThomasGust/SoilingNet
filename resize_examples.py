import cv2
import os

def resize_examples():
    path = "figures\\examples\\predictions"

    for n in os.listdir(path):
        img = cv2.imread(os.path.join(path, n))
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(path, n), img)

    path = "figures\\examples\\images"

    for n in os.listdir(path):
        img = cv2.imread(os.path.join(path, n))
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(path, n), img)

if __name__ == "__main__":
    resize_examples()