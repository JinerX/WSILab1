import torch
import torch.nn as nn


class fullyConnectedModel(nn.Module):
    def __init__(self,
                 in_vals=784):
        super().__init__()

        self.fc1 = nn.Linear(in_features=in_vals,
                             out_features=16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=16,
                             out_features=10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class CnnModel(nn.Module):
    def __init__(self,
                 in_shape=(28, 28)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5)
        self.fc1 = nn.Linear(in_features=8 * 20 * 20, out_features=10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x