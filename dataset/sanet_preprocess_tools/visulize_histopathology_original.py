# 查看histopathology的数据

import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt


def load_case_ct(bmp_dir, case_name):
    """
    读取一个病例的所有BMP切片
    """
    case_path = os.path.join(bmp_dir, case_name)
    if not os.path.exists(case_path):
        raise FileNotFoundError(f"病例目录不存在: {case_path}")

    # 获取所有BMP文件并排序
    bmp_files = sorted([f for f in os.listdir(case_path) if f.endswith('.bmp')])

    slices = []
    for bmp_file in bmp_files:
        img_path = os.path.join(case_path, bmp_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            slices.append(img)

    return slices, bmp_files


def load_annotations(csv_path, case_name):
    """
    读取指定病例的标注坐标
    """
    df = pd.read_csv(csv_path)

    # 过滤出指定病例的标注
    # 根据CSV的实际列名调整，这里假设包含病例名称的列
    case_annotations = df[df['image'] == case_name]

    return case_annotations


def draw_nodule_on_slice(slice_img, x_min, y_min, x_max, y_max, color=(0, 255, 0), thickness=2):
    """
    在单个切片上绘制结节标记
    """
    img_copy = slice_img.copy()
    
    # 如果是灰度图，转换为彩色图以便显示彩色框
    if len(img_copy.shape) == 2:
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)
    
    # 绘制矩形框
    top_left = (int(x_min), int(y_min))
    bottom_right = (int(x_max), int(y_max))
    
    cv2.rectangle(img_copy, top_left, bottom_right, color, thickness)
    
    # 绘制中心点
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    cv2.circle(img_copy, (int(center_x), int(center_y)), 3, (0, 0, 255), -1)
    
    return img_copy


def visualize_case(bmp_dir, csv_path, case_name, box_size=30):
    """
    可视化一个病例的CT和结节标注
    """
    # 加载CT数据
    slices, bmp_files = load_case_ct(bmp_dir, case_name)
    print(f"加载病例: {case_name}, 共 {len(slices)} 个切片")

    # 加载标注
    annotations = load_annotations(csv_path, case_name)
    print(f"找到 {len(annotations)} 个结节标注")

    if len(annotations) == 0:
        print("未找到该病例的标注")
        return

    # 显示所有切片，有标注的切片绘制矩形框
    fig, axes = plt.subplots(1, min(len(slices), 10), figsize=(20, 4))
    if len(slices) == 1:
        axes = [axes]

    # 选择几个关键切片显示
    step = max(1, len(slices) // 10)
    selected_indices = list(range(0, len(slices), step))[:10]

    for idx, slice_idx in enumerate(selected_indices):
        if idx < len(axes):
            img = slices[slice_idx]

            # 检查当前切片是否有标注
            slice_has_annotation = False
            for _, anno in annotations.iterrows():
                # 从image列提取切片信息，如 '0001_41.bmp'
                image_name = anno['image']
                x_min = anno['x_min']
                x_max = anno['x_max']
                y_min = anno['y_min']
                y_max = anno['y_max']
                
                # 检查是否是当前切片的标注
                if bmp_files[slice_idx] == image_name:
                    img = draw_nodule_on_slice(img, x_min, y_min, x_max, y_max)
                    slice_has_annotation = True

            axes[idx].imshow(img, cmap='gray')
            axes[idx].set_title(f'Slice {slice_idx}')
            axes[idx].axis('off')

            if slice_has_annotation:
                axes[idx].set_title(f'Slice {slice_idx} (有结节)', color='red')

    # 隐藏多余的子图
    for idx in range(len(selected_indices), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.suptitle(f'病例: {case_name}', fontsize=16, y=1.02)
    plt.show()


def visualize_single_slice(bmp_dir, csv_path, case_name, slice_idx):
    """
    可视化单个切片
    """
    slices, bmp_files = load_case_ct(bmp_dir, case_name)
    if slice_idx >= len(slices):
        print(f"切片索引超出范围，最大索引为 {len(slices) - 1}")
        return
    img = slices[slice_idx].copy()
    slice_annotations = load_annotations(csv_path, case_name+f'_{slice_idx}.bmp')

    print(f"切片 {slice_idx} 找到 {len(slice_annotations)} 个结节")

    # 绘制所有结节
    for _, anno in slice_annotations.iterrows():
        x_min = anno['x_min']
        x_max = anno['x_max']
        y_min = anno['y_min']
        y_max = anno['y_max']
        img = draw_nodule_on_slice(img, x_min, y_min, x_max, y_max)

    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray')
    plt.title(f'病例: {case_name}, 切片: {slice_idx}')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # 配置路径
    BASE_DIR = '/data/医保大赛/dataset/Histopathology'
    BMP_DIR = os.path.join(BASE_DIR, 'BMP_3D')
    CSV_PATH = os.path.join(BASE_DIR, 'all_anno_3D.csv')

    # 示例：可视化一个病例
    CASE_NAME = '0024'# 替换为实际的病例名称
    slice_idx=26 #替换为实际的切片索引

    # 方式1：可视化整个病例的所有切片
    # visualize_case(BMP_DIR, CSV_PATH, CASE_NAME, box_size=30)

    # 方式2：可视化单个切片
    visualize_single_slice(BMP_DIR, CSV_PATH, CASE_NAME, slice_idx=slice_idx, box_size=30)

    print("请修改 CASE_NAME 为实际的病例名称后运行")
    print(f"BMP目录: {BMP_DIR}")
    print(f"标注文件: {CSV_PATH}")