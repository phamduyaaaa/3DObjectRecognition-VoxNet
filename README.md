![loss_plot](https://github.com/user-attachments/assets/6e4ace32-a1d6-4e36-98cf-2e8138699812)Nhận diện Đối tượng 3D sử dụng VoxNet trên ModelNet10

📌 Tổng quan

Dự án này triển khai Nhận diện Đối tượng 3D bằng VoxNet, một mạng nơ-ron tích chập 3D (3D-CNN) dựa trên học sâu. Mô hình được huấn luyện và đánh giá trên tập dữ liệu ModelNet10, bao gồm các mô hình đối tượng 3D thuộc 10 danh mục khác nhau. Mục tiêu là phân loại các đối tượng dựa trên biểu diễn voxel của chúng.

📂 Tập dữ liệu: ModelNet10

ModelNet10 là một tập con của ModelNet, chứa 10 danh mục đối tượng 3D:

Bathtub

Bed

Chair

Desk

Dresser

Monitor

Nightstand

Sofa

Table

Toilet

Mỗi danh mục bao gồm các mẫu huấn luyện và kiểm tra ở định dạng .off, được chuyển đổi thành lưới voxel trước khi đưa vào mô hình VoxNet.

🏗 Cấu trúc Dự án

3DObjectRecognition-VoxNet/
├── ModelNet10/                 # Tập dữ liệu ModelNet10 (sau tiền xử lý)     
├── metadata_modelnet10.csv     # Xử lý và tải dữ liệu
├── model.py         # Triển khai mô hình VoxNet
├── train.py          # Script huấn luyện
├── test.py         # Script đánh giá
├── real_time.py          # real-time
└── dataset.py        #
Kết quả:
![loss_plot](https://github.com/user-attachments/assets/dca3dbe9-45f7-44cd-b2c6-c2ae49d4f928)
