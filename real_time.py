import torch
import open3d as o3d
import numpy as np
from model import VoxNet

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VoxNet(num_classes=10).to(device)
model.load_state_dict(torch.load("voxnet_model_20epochs.pth"))
model.eval()

# Kết nối camera
camera = o3d.io.AzureKinectSensor()
camera.start()

while True:
    rgbd = camera.capture()
    if rgbd is None:
        continue

    # Chuyển RGBD thành Point Cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera.intrinsics)
    points = np.asarray(pcd.points, dtype=np.float32)

    # Tiền xử lý (chuẩn hóa)
    points = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0))

    # Chuyển đổi thành tensor
    points = torch.tensor(points).float().to(device).unsqueeze(0).unsqueeze(0)

    # Dự đoán
    with torch.no_grad():
        outputs = model(points)
        _, predicted = torch.max(outputs, 1)

    print(f"Detected Object: {predicted.item()}")
