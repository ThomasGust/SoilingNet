from dataset import load_data, augment_data
import shutil
import os

if __name__ == "__main__":
    shutil.rmtree("AugmentedDataset\\images")
    shutil.rmtree("AugmentedDataset\\labels")
    os.mkdir("AugmentedDataset\\images")
    os.mkdir("AugmentedDataset\\labels")
    images, masks = load_data("Dataset")
    augment_data(images, masks, "AugmentedDataset")