import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model import VoxNet
from dataset import ModelNetDataset
from colorama import Fore
import csv
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# Cấu hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
num_epochs = 100
batch_size = 16
learning_rate = 0.001
colors = [Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.WHITE]

# Load dataset
train_dataset = ModelNetDataset(root="ModelNet10", train=True)
test_dataset = ModelNetDataset(root="ModelNet10", train=False)
set_seed(42)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model
model = VoxNet(num_classes=10).to(device)
state_dict = torch.load("init_weights.pth", map_location=device, weights_only=True)
model.load_state_dict(state_dict, strict=False)
print(model)
print("LOAD INIT_WEIGHT")
print(model.fc1.weight.data.clone())
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
min_loss = float('inf')
epoch_save = 0

# Tạo file CSV và ghi tiêu đề
csv_filename = "training_log.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Loss", "Accuracy"])  # Ghi tiêu đề cột

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

    avg_loss = running_loss / len(train_loader)

    # Đánh giá độ chính xác trên tập kiểm tra
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for points, labels in test_loader:
            points, labels = points.to(device), labels.to(device)
            outputs = model(points)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total  # Tính accuracy (%)

    # Ghi log ra console
    print(random.choice(colors) + f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Lưu vào CSV
    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([epoch + 1, avg_loss, accuracy])

    # Lưu model nếu có improvement
    if avg_loss <= min_loss:
        min_loss = avg_loss
        epoch_save = epoch
        torch.save(model.state_dict(), "voxnet_model_best_default.pth")
        print(f"--> NEW UPDATE! min_loss = {min_loss:.4f}, Accuracy = {accuracy:.2f}% <--")

print(Fore.RED + f"Model saved. The best is {epoch_save}!")
