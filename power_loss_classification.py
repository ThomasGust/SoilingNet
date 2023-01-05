import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from dataset import DynamicSolarPanelSoilingDataset
import torch.nn.functional as F
from torch.utils.data import random_split
import time
import torchmetrics
import numpy as np
from torch.utils.data import WeightedRandomSampler
import matplotlib.pyplot as plt
from classifier_architectures import *
import pickle as pkl
import os
import shutil


device = torch.device('cuda:0')

def get_sampler(dataset):
    class_weights = []

    un = np.unique(np.array(dataset.labels))

    for u in un:
        class_weights.append(1/dataset.labels.count(u))

    sample_weights = [0] * len(dataset)

    for idx, (_, label, _) in enumerate(dataset):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight
    
    sample_weights[0] *= 1.5
    sample_weights[1] *= 1.5
    
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    return sampler

class ClassifierTrainingConfig:

    def __init__(self, model, name, output_dimensions, train_split, num_epochs=100, batch_size=64, learning_rate=(1e-4)):
        self.batch_size = batch_size
        self.lr = learning_rate
        self.num_epochs = num_epochs
        self.nc = output_dimensions

        print(train_split)
        self.model = model
        self.path = f"classifier_checkpoints\\{name}_{self.nc}_{int(round(train_split*10))}"
        self.name = f"{name}_{self.nc}_{int(round(train_split*10))}"
        self.train_split = train_split
        
        
def get_training_configs():
    models = [(ShortConv, 'ClassifierModel')]
    num_classes = [4, 8, 12, 16]
    splits = [0.0]
    configs = []

    for model in models:
        for num_class in num_classes:
            for split in splits:
                m, n = model
                configs.append(ClassifierTrainingConfig(m, n, num_class, split))
    
    return configs


def training_loop(config):
    NUM_CLASSES = config.nc
    BATCH_SIZE = config.batch_size
    LEARNING_RATE = config.lr
    NUM_EPOCHS = config.num_epochs
    save_path = config.path

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    model = config.model(NUM_CLASSES).to(device)

    ds = DynamicSolarPanelSoilingDataset(NUM_CLASSES, "PanelImages", segmentation_model=None, every=5, format='PNG', transform=transforms.ToTensor())

    train_size = int(config.train_split * len(ds))
    test_size = len(ds) - train_size
    ds, tds = torch.utils.data.random_split(ds, [train_size, test_size])

    ds = ds.dataset
    tds = tds.dataset

    sampler = get_sampler(ds)

    training_dataloader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler)
    testing_dataloader = DataLoader(tds, batch_size=BATCH_SIZE, shuffle=False)

    def time_convert(sec):
        mins = sec // 60
        sec = sec % 60
        hours = mins // 60
        mins = mins % 60
        return int(hours), int(mins), sec


    criterion = nn.CrossEntropyLoss()
    accuracy = torchmetrics.Accuracy().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    l = lambda epoch: 0.985 ** epoch
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, l)

    loss_hist = []
    acc_hist = []
    time_hist = []

    for epoch in range(NUM_EPOCHS):
        e_loss = []
        e_acc = []

        st = time.time()
        for i, (images, labels, irradiance) in enumerate(training_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            irradiance = irradiance.to(device)

            preds = model(images, irradiance)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            e_loss.append(loss.cpu().detach().item())
            e_acc.append(accuracy(preds, labels).detach().cpu().item())
        
        e_loss = sum(e_loss)/len(e_loss)
        e_acc = sum(e_acc)/len(e_acc)

        loss_hist.append(e_loss)
        acc_hist.append(e_acc)

        scheduler.step()

        saved = True

        #if epoch % 5 == 0 or epoch+1==NUM_EPOCHS: 
            #torch.save(model, f"{save_path}\\MODEL{epoch+1}.pt")
        torch.save(model, f"{save_path}\\MODEL{epoch}.pt")
        
        et = time.time()

        elapsed = et - st

        time_hist.append(elapsed)

        hours, minutes, seconds = time_convert(elapsed)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss [{e_loss}], Accuracy [{e_acc}], Saved [{str(saved)}], Time [{hours}:{minutes}:{seconds}]")
    

    hist = []
    hist.append(loss_hist)
    hist.append(acc_hist)
    hist.append(time_hist)

    print('Finished Training')

    epochs = range(NUM_EPOCHS)
    plt.plot(epochs, loss_hist, 'g', label='Train Loss Classifier')
    plt.title("Training Loss Classifier")
    plt.xlabel("Epochs")
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{save_path}\\LOSS.png")
    plt.close()

    plt.plot(epochs, acc_hist, 'g', label='Train Accuracy Classifier')
    plt.title("Training Accuracy Classifier")
    plt.xlabel("Epochs")
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"{save_path}\\ACCURACY.png")
    plt.close()

    if config.train_split != 0:
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            n_class_correct = [0 for i in range(NUM_CLASSES)]
            n_class_samples = [0 for i in range(NUM_CLASSES)]

            for images, labels, irradiances in testing_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                irradiances = irradiances.to(device)
                preds = model(images, irradiances)

                _, predicted = torch.max(preds, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

                for i in range(BATCH_SIZE):
                    try:
                        label = labels[i]
                        pred = predicted[i]
                        if (label == pred):
                            n_class_correct[label] += 1
                        n_class_samples[label] += 1
                    except IndexError as e:
                        pass
                
            acc = 100 * n_correct / n_samples
            print(f'Network Accuracy: {acc:.4f}%')

            test_accuracies = []
            for i in range(NUM_CLASSES):
                acc = 100 * n_class_correct[i] / n_class_samples[i]
                test_accuracies.append(acc)
                print(f'Accuracy of {[i]}:{acc:.4f}%')
        
        hist.append(test_accuracies)
    else:
        hist.append([])

    hist_path = os.path.join("training_stats", config.name)

    if not os.path.exists(hist_path):
        os.mkdir(hist_path)
    else:
        shutil.rmtree(hist_path)
        os.mkdir(hist_path)
    with open(f"{hist_path}\\hist.pkl", "wb") as f:
        pkl.dump(hist, f)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
if __name__ == "__main__":
    configs = get_training_configs()

    for config in configs:
        training_loop(config)

        