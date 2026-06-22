import matplotlib.pyplot as plt
import numpy as np

import os

# 读取灰度图
image_path = "/Disk_16TB/zhouhaowei/code/network_GAS/TMI/save_results/reconstructions/fdk_reconstruction.npy"

image = np.load(image_path)
plt.imshow(image[128, :, :], cmap='gray')
plt.show()
plt.savefig("fdk_reconstruction.png", bbox_inches='tight', pad_inches=0)
# image_files = sorted(os.listdir(image_path))[:50]  # 获取文件夹中的前50个文件
# selected_files = image_files[::5]  # 均匀挑选10个文件

# # 创建图像
# fig, axs = plt.subplots(2, 5, figsize=(15, 6))  # 创建2行5列的子图

# for i, file in enumerate(selected_files):
#     image = np.load(os.path.join(image_path, file))
#     axs[i // 5, i % 5].imshow(image, cmap='gray')
#     axs[i // 5, i % 5].axis('off')  # 不显示坐标轴

# plt.tight_layout()
# plt.savefig("patient_00001_img_01_cone_proj_train.png", bbox_inches='tight', pad_inches=0)
# plt.close()
# # 读取灰度图
# image = np.load(image_path)

# # 创建图像
# plt.figure(figsize=(8, 8))
# plt.imshow(image, cmap='gray')
# plt.axis('off')  # 不显示坐标轴
# plt.title('投影图像: projection108')

# # 保存图像
# plt.tight_layout()
# plt.savefig("projection108_visualization.png", bbox_inches='tight', pad_inches=0)
# plt.show()

# # 显示图像的基本信息
# print(f"图像形状: {image.shape}")
# print(f"像素值范围: [{image.min()}, {image.max()}]")
# print(f"平均值: {image.mean():.4f}")
# print(f"标准差: {image.std():.4f}")
