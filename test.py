import torch
from torch.utils.data import DataLoader
from model import VoxNet
from dataset import ModelNetDataset
import open3d as o3d
import numpy as np

def visualize_voxel(voxel_grid, label, predicted):
    coords = np.argwhere(voxel_grid > 0)  # (N, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords.astype(np.float32))
    o3d.visualization.draw_geometries([pcd], window_name=f"Label: {label}, Predicted: {predicted}")

# Cấu hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16

# Load dataset
test_dataset = ModelNetDataset(root="./ModelNet10", train=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load model
model = VoxNet(num_classes=10).to(device)
model.load_state_dict(torch.load("voxnet_model_best.pth"))
model.eval()

# Evaluation
correct = 0
total = 0
with torch.no_grad():
    for voxels, labels in test_loader:
        voxels, labels = voxels.to(device), labels.to(device) # torch.Size([16, 1, 32, 32, 32])
        outputs = model(voxels)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        voxel_np = voxels[0, 0].cpu().numpy()  # (32, 32, 32)
        #visualize_voxel(voxel_np, labels[0].item(), predicted[0].item())

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
