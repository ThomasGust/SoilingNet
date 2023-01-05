from utils import get_confusion_matrix, plot_confusion_matrix
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from dataset import DynamicSolarPanelSoilingDataset
import numpy as np
import os
import torchmetrics


def get_labels(n_classes, every):
    labels = []

    file_names = os.listdir("PanelImages")
    
    for name in file_names:
        smooth_label = float(name.split("_L_")[1].split("_I_")[0])
        hard_label = int(round(smooth_label*(n_classes-1)))
        labels.append(hard_label)

    return labels[::every]

every = 1
device = torch.device('cuda:0')
ds = DynamicSolarPanelSoilingDataset(4, "PanelImages", segmentation_model=None, every=every, format='PNG', transform=transforms.ToTensor())

def build_confusion_matrix(c, s):
    model = torch.load(f'classifier_checkpoints\\ClassifierModel_{c}_{s}\\MODEL99.pt').to(device)

    dataloader = DataLoader(ds, 64)

    predictions = []
    _predictions = []
    labels = get_labels(c, every)

    with torch.no_grad():
        for i, (img, true, irradiance) in enumerate(dataloader):
            img = img.to(device)
            irradiance = irradiance.to(device)

            preds = model(img, irradiance)
            _preds = torch.softmax(preds, dim=1)
            preds = torch.argmax(_preds, dim=1)

            preds = preds.cpu().tolist()

            _preds = _preds.cpu().tolist()
            for p in _preds:
                _predictions.append(p)
            for p in preds:
                predictions.append(p)



    predictions = np.array(predictions)
    labels = np.array(labels)
    _predictions = np.array(_predictions)

    print(_predictions.shape)
    print(labels.shape)
    top_one_accuracy = torchmetrics.Accuracy(num_classes=c)(torch.tensor(predictions), torch.tensor(labels))
    top_two_accuracy = torchmetrics.Accuracy(num_classes=c, top_k=2)(torch.tensor(_predictions), torch.tensor(labels))
    #top_two_accuracy = torchmetrics.Accuracy(num_classes=c, top_k=2)(torch.tensor(_preds), torch.tensor(labels))

    print()
    print(top_one_accuracy)
    print(top_two_accuracy)
    print()
    #print(top_two_accuracy)

    cf = get_confusion_matrix(predictions, labels, per=True)
    plot_confusion_matrix(cf, c, s)

def generate_configs():
    #cs = [4, 8, 12, 16]
    cs = [4, 8, 12, 16]
    ss = [0]

    configs = []

    for c in cs:
        for s in ss:
            configs.append((c, s))
    
    return configs

def generate_confusion_matrices():
    configs = generate_configs()

    for i, config in enumerate(configs):
        c, s = config
        build_confusion_matrix(c, s)

        print(f"Generate confusion matrix for config {i+1}")

if __name__ == "__main__":
    generate_confusion_matrices()