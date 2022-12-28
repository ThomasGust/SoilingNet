from dataset import load_data, augment_data

if __name__ == "__main__":
    images, masks = load_data("Dataset")
    augment_data(images, masks, "AugmentedDataset")