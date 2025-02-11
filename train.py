import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model import VoxNet
from dataset import ModelNetDataset
from colorama import Fore
import random

# Cấu hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 300 #12h45p
batch_size = 16
learning_rate = 0.001
colors = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.WHITE]

# Load dataset
train_dataset = ModelNetDataset(root="ModelNet10", train=True)
test_dataset = ModelNetDataset(root="ModelNet10", train=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model
model = VoxNet(num_classes=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
min_loss = 9999999.0
epoch_save = 0
# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (points, labels) in enumerate(train_loader):
        points, labels = points.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(points)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(random.choice(colors)+ f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")
    # Lưu model
    if (running_loss/len(train_loader)) <= min_loss:
        min_loss = running_loss/len(train_loader)
        epoch_save = epoch
        torch.save(model.state_dict(), f"voxnet_model_best.pth")
        print(f"--> NEW UPDATE! min_loss = {min_loss} <--")
print(Fore.RED + f"Model saved. The best is {epoch_save}!")
