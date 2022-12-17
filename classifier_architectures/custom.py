from torch import nn
import torch.nn.functional as F
import torch

class ConvNet(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6*8, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6*8, 16*8, 5)
        self.conv3 = nn.Conv2d(16*8, 32*8, 5)
        self.conv4 = nn.Conv2d(32*8, 64*8, 3)
        self.conv5 = nn.Conv2d(64*8, 128*8, 3)

        self.fc1 = nn.Linear(9216, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, nc)

        self.flatten = nn.Flatten()


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        return x


class ClassifierModel(nn.Module):

    def __init__(self, nc):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)

        self.pool = nn.MaxPool2d(3, 3)

        self.fc1 = nn.Linear(23104+1, 8196)
        self.fc2 = nn.Linear(8196, nc)

        self.f = nn.Flatten()
    
    def forward(self, x, i):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.f(x)

        x = torch.concat((x, i.unsqueeze(1)), 1)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x
    
class M(nn.Module):

    def __init__(self, nc):
        super().__init__()
        self.f = nn.Flatten()
        self.fc1 = nn.Linear(110592, 9000)
        self.fc2 = nn.Linear(9000, nc)
    
    def forward(self, x):
        x = self.f(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x