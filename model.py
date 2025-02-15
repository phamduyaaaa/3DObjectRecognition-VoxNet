import torch.nn as nn

class VoxNet(nn.Module):
    def __init__(self, norm_type="groupnorm"):  # "batchnorm", "groupnorm", "layernorm"
        super(VoxNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1)

        if norm_type == "batchnorm":
            self.norm1 = nn.BatchNorm3d(32)
            self.norm2 = nn.BatchNorm3d(64)
        elif norm_type == "groupnorm":
            self.norm1 = nn.GroupNorm(4, 32)  # 4 nhóm cho 32 kênh
            self.norm2 = nn.GroupNorm(4, 64)  # 4 nhóm cho 64 kênh
        elif norm_type == "layernorm":
            self.norm1 = nn.LayerNorm([32, 14, 14, 14])  # Cần đúng shape, có thể phải chỉnh lại sau
            self.norm2 = nn.LayerNorm([64, 12, 12, 12])  # Cần đúng shape, kiểm tra lại sau
            
        self.dropout = nn.Dropout(p=0.3)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(13824, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)  # Áp dụng chuẩn hóa
        x = nn.ReLU()(x)

        x = self.conv2(x)
        x = self.norm2(x)  # Áp dụng chuẩn hóa
        x = nn.ReLU()(x)

        x = self.maxpool(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = nn.ReLU()(x)

        x = self.fc2(x)
        return x


#model1 = VoxNet()
#print(model1.fc1.weight.data.clone())
#torch.save(model1.state_dict(),"init_weights.pth")
