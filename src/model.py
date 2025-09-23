import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # [batch, 32, 16, 16]
        x = self.pool(F.relu(self.conv2(x)))   # [batch, 64, 8, 8]
        x = x.view(-1, 64 * 8 * 8)             # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
