from classifier_architectures import ShortConv
from dataset import DynamicSolarPanelSoilingDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch


device = torch.device('cuda:0')
model = torch.load(f='classifier_checkpoints\\MODEL190.pt')
ds = DynamicSolarPanelSoilingDataset(4, "Solar_Panel_Soiling_Image_dataset\\PanelImages", every=1, format='PNG', transform=transforms.ToTensor())

testing_dataloader = DataLoader(ds, 32)

classes = ('CLEAN', 'SLIGHTLY DIRTY', "MODERATELY DIRTY", "VERY DIRTY", "EXTREMELY DIRTY")

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(5)]
    n_class_samples = [0 for i in range(5)]

    for images, labels, irradiances in testing_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        irradiances = irradiances.to(device)
        preds = model(images, irradiances)
        #print(preds)

        _, predicted = torch.max(preds, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(32):
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

    for i in range(4):
        acc = 100 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}:{acc:.4f}%')