'''
可视化SANet预处理后的数据，验证npy文件和标注是否正确
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_sanet_output(sanet_dir, case_pid=None, slice_idx=None):
    """
    可视化SANet预处理后的数据，验证npy文件和标注是否正确
    
    参数:
        sanet_dir: SANet输出目录路径
        case_pid: 要可视化的病例PID（如 1, 2, 5），None则随机选择一个有标注的病例
        slice_idx: 要可视化的切片索引，None则自动选择有结节的切片
    """
    sanet_dir = Path(sanet_dir)
    full_dir = sanet_dir / "full"
    anno_csv = sanet_dir / "split" / "all_anno.csv"
    
    # 加载标注
    ann = pd.read_csv(anno_csv)
    print(f"加载标注文件: {anno_csv}")
    print(f"总共 {len(ann)} 个结节标注")
    print(f"标注列: {list(ann.columns)}")
    print(ann.head())
    
    # 选择病例
    if case_pid is None:
        # 随机选择一个有标注的病例
        case_pid = ann['pid'].iloc[0]
        print(f"\n自动选择病例 PID={case_pid}")
    
    # 加载npy文件
    npy_file = full_dir / f"{case_pid:05d}_zoom.npy"
    if not npy_file.exists():
        print(f"错误: 找不到文件 {npy_file}")
        return
    
    volume = np.load(npy_file)
    print(f"\n加载病例: {npy_file}")
    print(f"Volume shape: {volume.shape} (应为 [1, D, H, W])")
    print(f"Value range: [{volume.min():.1f}, {volume.max():.1f}]")
    
    # 获取该病例的所有结节
    case_ann = ann[ann['pid'] == case_pid]
    print(f"\n病例 {case_pid} 有 {len(case_ann)} 个结节")
    print(case_ann)
    
    # 选择切片
    if slice_idx is None:
        # 选择第一个结节的中心切片
        first_nodule = case_ann.iloc[0]
        slice_idx = int((first_nodule['zmin'] + first_nodule['zmax']) / 2)
        print(f"\n自动选择切片 idx={slice_idx} (结节中心)")
    
    if slice_idx >= volume.shape[1]:
        print(f"错误: 切片索引 {slice_idx} 超出范围 [0, {volume.shape[1]-1}]")
        return
    
    # 获取切片图像
    img = volume[0, slice_idx]  # [H, W]
    
    # 找出该切片上的所有结节
    slice_nodules = case_ann[(case_ann['zmin'] <= slice_idx) & (case_ann['zmax'] >= slice_idx)]
    print(f"\n切片 {slice_idx} 上有 {len(slice_nodules)} 个结节")
    
    # 可视化
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img, cmap='gray')
    
    # 绘制结节边界框
    colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'orange', 'lime', 'pink', 'purple']
    for idx, (_, nodule) in enumerate(slice_nodules.iterrows()):
        color = colors[idx % len(colors)]
        xmin, xmax = int(nodule['xmin']), int(nodule['xmax'])
        ymin, ymax = int(nodule['ymin']), int(nodule['ymax'])
        nodule_id = int(nodule['nodule_id'])
        
        # 绘制矩形框
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                            fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        
        # 标注结节ID
        ax.text(xmin, ymin - 5, f'Nodule {nodule_id}', 
                color=color, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        print(f"  结节 {nodule_id}: z[{int(nodule['zmin'])}-{int(nodule['zmax'])}], "
              f"x[{xmin}-{xmax}], y[{ymin}-{ymax}]")
    
    ax.set_title(f'PID={case_pid}, Slice={slice_idx}, '
                f'{len(slice_nodules)} nodule(s) in this slice',
                fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    # 显示结节在Z轴的分布
    if len(case_ann) > 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        for idx, (_, nodule) in enumerate(case_ann.iterrows()):
            color = colors[int(nodule['nodule_id']) % len(colors)]
            ax.plot([nodule['zmin'], nodule['zmax']], 
                   [int(nodule['nodule_id']), int(nodule['nodule_id'])],
                   color=color, linewidth=3, marker='o', markersize=8)
            ax.text(nodule['zmin'], int(nodule['nodule_id']) + 0.1,
                   f'Nodule {int(nodule["nodule_id"])}', 
                   color=color, fontweight='bold')
        
        ax.set_xlabel('Z-axis (slice index)', fontsize=12)
        ax.set_ylabel('Nodule ID', fontsize=12)
        ax.set_title(f'PID={case_pid}: Nodule distribution along Z-axis', fontsize=14)
        ax.set_yticks(range(len(case_ann)))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # 配置路径
    SANET_DIR = '/data/医保大赛/code/SANet/data/LNDB'
    
    # 方式1: 随机选择一个病例可视化
    visualize_sanet_output(SANET_DIR)
    
    # 方式2: 指定病例PID
    # visualize_sanet_output(SANET_DIR, case_pid=24)
    
    # 方式3: 指定病例和切片
    visualize_sanet_output(SANET_DIR, case_pid=1, slice_idx=260)