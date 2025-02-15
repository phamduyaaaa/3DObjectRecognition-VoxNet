import torch
import torch.nn as nn
import torch.nn.functional as F


class VoxNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv3d(1, 32, kernel_size=5, stride=2)  # (32, 14, 14, 14)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1)  # (64, 12, 12, 12)
        self.batchnorm1 = nn.BatchNorm3d(32)
        self.batchnorm2 = nn.BatchNorm3d(64)
        self.dropout = nn.Dropout(p=0.3)
        self.maxpooling = nn.MaxPool3d(kernel_size=2, stride=2)  # (64, 6, 6, 6)

        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 6 * 6 * 6, 1024)  # (1024)
        self.fc2 = nn.Linear(1024, num_classes)  # (num_classes)

    def forward(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))  # Conv1 -> ReLU -> BatchNorm
        x = F.relu(self.batchnorm2(self.conv2(x)))  # Conv2 -> ReLU -> BatchNorm
        #x = F.relu(self.conv1(x))
        #x = F.relu(self.conv2(x))
        x = self.maxpooling(x)  # MaxPool
        x = self.dropout(x)  # Dropout
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))  # FC1 -> ReLU
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model1 = VoxNet()
print(model1.fc1.weight.data.clone())
torch.save(model1.state_dict(),"init_weights.pth")
