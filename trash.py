
import numpy as np
from PIL import Image

# 读取图像数据
image_path = '/Disk_16TB/zhouhaowei/code/network_GAS/TMI/experiments_out/2025-04-30/00-45-05/logs/1/wandb/latest-run/files/media/images/渲染/真实视角0_78_a49480187a8f53de8f8d.png'
image = Image.open(image_path)
image_data = np.array(image)

# 查看图像数据的形状
print(image_data.shape)

# 查看图像数据中的最大值和最小值
max_value = np.max(image_data)
min_value = np.min(image_data)
print(f"图像数据中的最大值为 {max_value}，最小值为 {min_value}")

# 统计1和0的像素数量
count_1 = np.sum(image_data == 1)
count_0 = np.sum(image_data == 0)
print(f"图像数据中像素值为1的数量为 {count_1}，像素值为0的数量为 {count_0}")
