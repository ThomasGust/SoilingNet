import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torchmetrics
import matplotlib.pyplot as plt
from dataset import DynamicSolarPanelSoilingDataset

def get_confusion_matrix(labels, predictions, per=True):
    K = len(np.unique(labels))
    result = np.zeros((K, K))

    for i in range(len(labels)):
        result[labels[i]][predictions[i]] += 1
    
    if per:
        result /= result.astype(np.float).sum(axis=0)
    
    return result

def plot_confusion_matrix(conf_matrix, c, s):
    c_dict = {4:'x-large', 8:'large', 12:'medium', 16:'x-small'}
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, alpha=0.7)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=f'{str(conf_matrix[i, j]*100):.4}%', va='center', ha='center', size=c_dict[c])
    
    plt.xlabel(f'Confusion Matrix Classifier Model-{c}-{s}', fontsize=18)
    plt.ylabel('Labels', fontsize=18)
    plt.title('Predictions', fontsize=18)
    plt.savefig(f'figures\\confusion_matrices\\CONFUSIONMATRIX_{c}_{s}.png')
    plt.close()

def get_labels(n_classes, every):
    labels = []


    file_names = os.listdir("PanelImages")
    
    for name in file_names:
        smooth_label = float(name.split("_L_")[1].split("_I_")[0])
        hard_label = int(round(smooth_label*(n_classes-1)))
        labels.append(hard_label)

    return labels[::every]

def build_confusion_matrix(c, s, ds, every, device):
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

    top_one_accuracy = torchmetrics.Accuracy(num_classes=c)(torch.tensor(predictions), torch.tensor(labels))
    top_two_accuracy = torchmetrics.Accuracy(num_classes=c, top_k=2)(torch.tensor(_predictions), torch.tensor(labels))

    print(top_one_accuracy)
    print(top_two_accuracy)

    cf = get_confusion_matrix(predictions, labels, per=True)
    plot_confusion_matrix(cf, c, s)

def generate_configs():
    cs = [4, 8, 12, 16]
    ss = [0]

    configs = []

    for c in cs:
        for s in ss:
            configs.append((c, s))
    
    return configs

def generate_confusion_matrices():
    every = 1
    device = torch.device('cuda:0')
    ds = DynamicSolarPanelSoilingDataset(4, "PanelImages", segmentation_model=None, every=every, format='PNG', transform=transforms.ToTensor())

    configs = generate_configs()

    for i, config in enumerate(configs):
        c, s = config
        build_confusion_matrix(c, s, ds, every, device)

        print(f"Generate confusion matrix for config {i+1}")