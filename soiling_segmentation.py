import os
import cv2
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
import numpy as np
from segmentation_model_architectures import UNET
import torch
from torch import nn
import torchmetrics
import time
from torch import optim
import matplotlib.pyplot as plt

def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    return int(hours), int(mins), sec


class SolarPanelDustClassificationDataset(Dataset):
    
    def __init__(self, imgs_path, seg_labels_path, transforms=None):
        self.imgs_path = imgs_path
        self.transforms = transforms
        self.imgs = []
        self.seglabels = []
        self.labels = []
        self.names = os.listdir(imgs_path)

        for n in self.names:
            img = cv2.imread(os.path.join(imgs_path, n))
            self.imgs.append(img)
            self.seglabels.append(cv2.imread(os.path.join(seg_labels_path, n)))
            self.labels.append(int(str(n.split("_DIS_")[1].split(".p")[0])))
        
        self.imgs = np.array(self.imgs)
        self.labels = np.array(self.labels)
        self.seglabels = np.array(self.seglabels)
      
    def __len__(self):
        return self.imgs.shape[0]
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        seglabel = self.seglabels[idx]
        label = self.labels[idx]
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        seg_tensor = torch.from_numpy(seglabel)
        seg_tensor = seg_tensor.permute(2, 0, 1)
        seg_tensor = torchvision.transforms.Grayscale()(seg_tensor)
        class_id = torch.tensor(label).long()
        return img_tensor.float(), seg_tensor.int(), class_id
    

BATCH_SIZE = 16

device = torch.device('cuda:0')

train_dataset_imgs_path = 'DatasetTwo\\train\\Images'
test_dataset_img_path = 'DatasetTwo\\test\\Images'

train_dataset_labels_path = 'DatasetTwo\\train\\ProcessedLabels'
test_dataset_labels_path = 'DatasetTwo\\test\\ProcessedLabels'

train_dataset = SolarPanelDustClassificationDataset(train_dataset_imgs_path, train_dataset_labels_path)
test_dataset = SolarPanelDustClassificationDataset(test_dataset_img_path, test_dataset_labels_path)

trainloader = DataLoader(train_dataset, BATCH_SIZE)
testloader = DataLoader(test_dataset, BATCH_SIZE)

model = UNET(3, 7).to(device)

criterion = nn.CrossEntropyLoss().to(device)
ji = torchmetrics.JaccardIndex(7).to(device)

optimizer = optim.Adam(model.parameters())
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: epoch*0.99)

train_accuracy_hist = []
test_accuracy_hist = []

train_loss_hist = []
test_loss_hist = []

NUM_EPOCHS = 200

for epoch in range(NUM_EPOCHS):
    print(f"Started epoch {epoch+1}")

    st = time.time()

    epoch_train_accuracy = []
    epoch_test_accuracy = []

    epoch_train_loss = []
    epoch_test_loss = []

    for sample in trainloader:
        img, seglabel, label = sample

        img = img.to(device)
        seglabel = seglabel.to(device)

        seglabel = seglabel.squeeze(1)

        preds = model(img)

        loss = criterion(preds, seglabel.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_loss.append(loss.cpu().detach().item())

        accuracy = 1
        epoch_train_accuracy.append(accuracy)

    for sample in testloader:
        img, seglabel, label = sample
        
        img = img.to(device)
        seglabel = seglabel.to(device)
        
        seglabel = seglabel.squeeze(1)

        preds = model(img)

        loss = criterion(preds, seglabel.long())

        accuracy = 1

        epoch_test_loss.append(loss.cpu().detach().item())
        epoch_test_accuracy.append(accuracy)
    
    train_accuracy = sum(epoch_train_accuracy)/len(epoch_train_accuracy)
    test_accuracy = sum(epoch_test_accuracy)/len(epoch_test_accuracy)

    train_loss = sum(epoch_train_loss)/len(epoch_train_loss)
    test_loss = sum(epoch_test_loss)/len(epoch_test_loss)

    train_accuracy_hist.append(train_accuracy)
    test_accuracy_hist.append(test_accuracy)
    
    train_loss_hist.append(train_loss)
    test_loss_hist.append(test_loss)

    if epoch % 10 == 0:
        torch.save(model, f'segmentation_checkpoints\\UNET\\BESTMODEL{epoch}')

    et = time.time()

    elapsed = st-et

    elapsed = time_convert(elapsed)

    print(f"Completed Epoch {epoch+1} | Train Loss: {train_loss} | Test Loss: {test_loss} | Train Jaccard Index: {train_accuracy} | Test Jaccard Index: {test_accuracy}")
    
    scheduler.step()


epochs=range(0, NUM_EPOCHS)
plt.plot(epochs, train_accuracy_hist, 'g', label='Training Accuracy')
plt.plot(epochs, test_accuracy_hist, 'b', label='Validation Accuracy')
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("ACCURACIES.png")
plt.close()

plt.plot(epochs, train_loss_hist, 'g', label='Training Loss')
plt.plot(epochs, test_loss_hist, 'b', label='Validation Loss')
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel('Loss')
plt.legend()
plt.savefig("LOSS.png")
plt.close()
