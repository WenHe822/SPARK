import re
import matplotlib.pyplot as plt
import numpy as np # 添加 numpy 用于计算平均值

def parse_log_file(log_file_path):
    """
    解析日志文件，提取迭代次数和损失值。

    Args:
        log_file_path (str): 日志文件的路径。

    Returns:
        tuple: 包含迭代次数、总损失、投影损失和 SSIM 损失列表的元组。
    """
    iterations = []
    total_losses = []
    projection_losses = []
    ssim_losses = []

    # 正则表达式匹配包含损失信息的行
    log_pattern = re.compile(
        r"迭代 (\d+) - 总损失: ([\d.]+) 投影损失: ([\d.]+) SSIM损失: ([\d.]+)"
    )

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = log_pattern.search(line)
                if match:
                    iterations.append(int(match.group(1)))
                    total_losses.append(float(match.group(2)))
                    projection_losses.append(float(match.group(3)))
                    ssim_losses.append(float(match.group(4)))
    except FileNotFoundError:
        print(f"错误：找不到日志文件 {log_file_path}")
        return None
    except Exception as e:
        print(f"读取或解析日志文件时出错: {e}")
        return None

    return iterations, total_losses, projection_losses, ssim_losses

def calculate_windowed_averages(iterations, total_losses, projection_losses, ssim_losses, window_size=500):
    """
    计算窗口内的平均损失值。

    Args:
        iterations (list): 迭代次数列表。
        total_losses (list): 总损失列表。
        projection_losses (list): 投影损失列表。
        ssim_losses (list): SSIM 损失列表。
        window_size (int): 计算平均值的窗口大小。

    Returns:
        tuple: 包含平均迭代次数、平均总损失、平均投影损失和平均 SSIM 损失列表的元组。
    """
    avg_iterations = []
    avg_total_losses = []
    avg_projection_losses = []
    avg_ssim_losses = []

    num_points = len(iterations)
    for i in range(0, num_points, window_size):
        end_index = min(i + window_size, num_points)
        if i == end_index: # 防止空窗口
            continue

        # 使用窗口内最后一个迭代次数作为代表
        avg_iterations.append(iterations[end_index - 1])
        avg_total_losses.append(np.mean(total_losses[i:end_index]))
        avg_projection_losses.append(np.mean(projection_losses[i:end_index]))
        avg_ssim_losses.append(np.mean(ssim_losses[i:end_index]))

    return avg_iterations, avg_total_losses, avg_projection_losses, avg_ssim_losses

def plot_losses(iterations, total_losses, projection_losses, ssim_losses, output_filename="loss_curve_avg.png", is_averaged=False):
    """
    绘制损失曲线并保存到文件。

    Args:
        iterations (list): 迭代次数列表 (原始或平均后的)。
        total_losses (list): 总损失列表 (原始或平均后的)。
        projection_losses (list): 投影损失列表 (原始或平均后的)。
        ssim_losses (list): SSIM 损失列表 (原始或平均后的)。
        output_filename (str): 输出图片文件名。
        is_averaged (bool): 指示数据是否为平均值。
    """
    if not iterations:
        print("没有找到可绘制的损失数据。")
        return

    plt.figure(figsize=(12, 6))

    plt.plot(iterations, total_losses, label='总损失 (Total Loss)')
    plt.plot(iterations, projection_losses, label='投影损失 (Projection Loss)')
    plt.plot(iterations, ssim_losses, label='SSIM 损失 (SSIM Loss)')

    plt.xlabel('迭代次数 (Iteration)')
    plt.ylabel('损失值 (Loss Value)')
    title = '平均训练损失曲线 (Averaged Training Loss Curve)' if is_averaged else '训练损失曲线 (Training Loss Curve)'
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 保存图像
    try:
        plt.savefig(output_filename)
        print(f"损失曲线图已保存到 {output_filename}")
    except Exception as e:
        print(f"保存图像时出错: {e}")

    # 如果在支持图形显示的 环境中运行，取消注释下一行以显示绘图
    # plt.show()

if __name__ == "__main__":
    # 请将此路径替换为您的实际日志文件路径
    log_file = "/Disk_16TB/zhouhaowei/code/network_GAS/TMI/experiments_out/2025-04-20/01-34-14/train_network.log"
    window_size = 500 # 设置窗口大小

    parsed_data = parse_log_file(log_file)

    if parsed_data:
        iterations, total_losses, projection_losses, ssim_losses = parsed_data

        # 计算窗口平均值
        avg_iterations, avg_total, avg_proj, avg_ssim = calculate_windowed_averages(
            iterations, total_losses, projection_losses, ssim_losses, window_size=window_size
        )

        # 绘制平均损失曲线
        plot_losses(avg_iterations, avg_total, avg_proj, avg_ssim, output_filename=f"loss_curve_avg_{window_size}.png", is_averaged=True)

        # （可选）如果您仍想绘制原始损失曲线，可以取消注释以下行：
        # plot_losses(iterations, total_losses, projection_losses, ssim_losses, output_filename="loss_curve_raw.png", is_averaged=False)
