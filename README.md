# Nhận diện Đối tượng 3D sử dụng VoxNet trên ModelNet10

### 📌 Tổng quan

#### Dự án này triển khai Nhận diện Đối tượng 3D bằng VoxNet, một mạng nơ-ron tích chập 3D (3D-CNN) dựa trên học sâu. Mô hình được huấn luyện và đánh giá trên tập dữ liệu ModelNet10, bao gồm các mô hình đối tượng 3D thuộc 10 danh mục khác nhau. Mục tiêu là phân loại các đối tượng dựa trên biểu diễn voxel của chúng.

### 🏗 Cấu trúc Dự án
```bash
3DObjectRecognition-VoxNet/
├── ModelNet10/           # Tập dữ liệu ModelNet10, bạn có thể tải từ kaggle miễn phí 
├── dataset.py            # Xử lý và tải dữ liệu
├── model.py              # Triển khai mô hình VoxNet
├── train.py              # Script huấn luyện
├── test.py               # Script đánh giá
├── real_time.py          # Script đánh giá trên thời gian thực
├── voxnet_model_20epochs # Weight sẵn 20 epochs, accuracy ~ 88%
├── voxnet_model_best.pth # Weight sẵn 160 epochs, acuracy ~ 90% (8 tiếng training).
└── requirements.txt      # Các thư viện phụ thuộc
```
### 📂 Tập dữ liệu: ModelNet10

#### [ModelNet10](https://www.kaggle.com/datasets/balraj98/modelnet10-princeton-3d-object-dataset) là tập dữ liệu phổ biến được sử dụng trong nghiên cứu nhận dạng đối tượng 3D.
#### Tập dữ liệu bao gồm 4890 mô hình 3D (3991-train/908-test).
#### Tập dữ liệu chứa 10 lớp đối tượng 3D khác nhau với định dạng .off:
```
ModelNet10/
├── bathub/
|  ├──train/
|  |  ├── bathtub_0001.off
|  |  ├── bathtub_0002.off
|  |  ├── ...
|  ├── test/
|  |  ├── bathtub_0501.off
|  |  ├── bathtub_0502.off
|  |  ├── ...
├── bed/
├── chair/
├── desk/
├── dresser/
├── monitor/
├── sofa/
├── table/
├── toilet/
```
#### Mỗi danh mục bao gồm các mẫu huấn luyện và kiểm tra ở định dạng .off, được chuyển đổi thành lưới voxel trước khi đưa vào mô hình VoxNet.
### 🧠 Cấu trúc VoxNet
#### Cấu trúc VoxNet này dựa trên bài báo [VoxNet: A 3D Convolutional Neural Network for Real-Time Object Recognition](https://www.ri.cmu.edu/pub_files/2015/9/voxnet_maturana_scherer_iros15.pdf). Tuy nhiên bài báo này đã được công bố từ năm 2015, nên tôi có bổ sung thêm Batchnorm3D - 1 layer được công bố cùng năm nhằm giúp mạng nơ-ron học tập tốt hơn.
```
├── 3D Conv1: 32 Kernel 5x5x5, stride = 2, ReLU
├── Batchnorm3D(32)
├── 3D Conv2: 32 Kernel 3x3x3, stride = 1, ReLU
├── Batchnorm3D(64)
├── MaxPooling3D: Kernel 2x2x2, stride = 2
├── Dropout = 0.3
├── Flatten, ReLU
├── FC1(64x6x6x6, 1024)
├── Dropout = 0.3
├── FC2(1024, num_classes = 10)
```
### 📊 Kết quả biểu đồ loss:
![loss_plot](https://github.com/user-attachments/assets/dca3dbe9-45f7-44cd-b2c6-c2ae49d4f928)
