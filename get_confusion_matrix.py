import torch
from utils import get_confusion_matrix, plot_confusion_matrix
from dataset import DynamicSolarPanelSoilingDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix


device = torch.device('cuda:0')
ds = DynamicSolarPanelSoilingDataset(16, "Solar_Panel_Soiling_Image_dataset\\PanelImages", every=1, format='PNG', transform=transforms.ToTensor())
model = torch.load('classifier_checkpoints\\ShortConv_16_5\\MODEL36.pt').to(device)

dataloader = DataLoader(ds, 64)

predictions = []
labels = []

with torch.no_grad():
    for i, (img, true, irradiance) in enumerate(dataloader):
        img = img.to(device)
        irradiance = irradiance.to(device)

        true = true.tolist()

        for t in true:
            labels.append(t)

        preds = model(img, irradiance)
        preds = torch.softmax(preds, dim=1)
        preds = torch.argmax(preds, dim=1)

        preds = preds.cpu().tolist()

        for p in preds:
            predictions.append(p)



predictions = np.array(predictions)
labels = np.array(labels)

print(predictions.shape)
print(labels.shape)


torch.save(predictions, 'preds.pt')
torch.save(labels, 'labels.pt')
"""

predictions = torch.load("preds.pt")
labels = torch.load("labels.pt")
"""

cf = get_confusion_matrix(predictions, labels, per=True)
plot_confusion_matrix(cf)
