import os
import numpy as np
import torch
from torch.utils.data import Dataset
import open3d as o3d

'''
    Load ModelNet10 (.off)
    Chuyển Point Cloud thành Voxel Grid
    Trả về data dưới dạng PyTorch Dataset, chuẩn hóa về [0,1]
'''

def pointcloud_to_voxel(points, grid_size=32):
    voxels = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)

    norm_p = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0) + 1e-5)
    voxel_coords = (norm_p * (grid_size - 1)).astype(int)

    for x, y, z in voxel_coords:
        voxels[x, y, z] = 1.0  # Đánh dấu voxel là có điểm

    return torch.tensor(voxels).unsqueeze(0)  # Đổi (D, H, W) → (1, D, H, W)


class ModelNetDataset(Dataset):
    def __init__(self, root, num_points=1024, grid_size=32, train=True):
        self.root = root
        self.num_points = num_points
        self.grid_size = grid_size
        self.train = train
        self.files = []
        split = "train" if train else "test"
        class_folders = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        self.classes = {cls: i for i, cls in enumerate(class_folders)}

        for cls in class_folders:
            folder = os.path.join(root, cls, split)
            if os.path.exists(folder):  # Kiểm tra xem thư mục có tồn tại không
                for file in os.listdir(folder):
                    if file.endswith(".off"):
                        self.files.append((os.path.join(folder, file), self.classes[cls]))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path, label = self.files[idx]
        mesh = o3d.io.read_triangle_mesh(file_path)
        point_cloud = mesh.sample_points_uniformly(self.num_points)
        points = np.asarray(point_cloud.points, dtype=np.float32)

        # Chuyển đổi point cloud thành voxel grid
        voxel_grid = pointcloud_to_voxel(points, self.grid_size)

        return voxel_grid, torch.tensor(label).long()
