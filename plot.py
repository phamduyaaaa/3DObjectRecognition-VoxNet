import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
df = pd.read_csv('training_log_ScanObjectNN.csv')  # Thay tên file nếu cần

# Tạo figure với 2 subplot
fig, ax1 = plt.subplots(figsize=(10, 5))

# Vẽ Loss (màu đỏ, trục trái)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='r')
ax1.plot(df['Epoch'], df['Loss'], linestyle='-', color='r', label='Loss')
ax1.tick_params(axis='y', labelcolor='r')

# Vẽ Accuracy (màu xanh, trục phải)
ax2 = ax1.twinx()  # Tạo trục y thứ hai
ax2.set_ylabel('Accuracy (%)', color='b')
ax2.plot(df['Epoch'], df['Accuracy'], linestyle='-', color='b', label='Accuracy')
ax2.tick_params(axis='y', labelcolor='b')

# Tiêu đề và hiển thị
plt.title('Loss & Accuracy per Epoch')
fig.tight_layout()  # Điều chỉnh layout để không bị chồng chéo
plt.grid(True)
plt.show()
