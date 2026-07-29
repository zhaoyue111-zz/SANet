# -*- coding: utf-8 -*-
'''
统计一个文件夹下所有 以 _ct.nii.gz / _CT.nii.gz 结尾 的 NIfTI 文件层厚信息，并统计 z-spacing > 2.5 mm 的病例数（适用于luna16和lndb数据集）
'''
from pathlib import Path
import argparse
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


def find_ct_nii_files(root: Path):
    files = []
    for p in root.rglob("*.nii.gz"):
        if "LNDB" in str(p):
            name = p.name.lower()
            if name.endswith("_ct.nii.gz"):
                files.append(p)
        else:
            files.append(p)

    return sorted(files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"/media/SENSETIME\yangtingting/T7/医保大赛数据/LUNA16/raw/subset9_nii", help="包含 *_ct.nii.gz 文件的根目录")
    parser.add_argument("--threshold", type=float, default=2.5, help="层厚阈值，默认 2.5mm")
    parser.add_argument("--out-csv", default="ct_spacing_summary.csv", help="输出统计 CSV")
    args = parser.parse_args()

    root = Path(args.root)
    files = find_ct_nii_files(root)

    if not files:
        raise RuntimeError(f"没有在 {root} 下找到 *_ct.nii.gz 文件")

    rows = []

    for p in tqdm(files, desc="Checking spacing"):
        try:
            img = sitk.ReadImage(str(p))

            # SimpleITK 顺序是 x, y, z
            sx, sy, sz = img.GetSpacing()
            nx, ny, nz = img.GetSize()

            rows.append({
                "file": str(p),
                "case_id": p.name.replace(".nii.gz", ""),
                "spacing_x": sx,
                "spacing_y": sy,
                "spacing_z_slice_thickness": sz,
                "size_x": nx,
                "size_y": ny,
                "size_z_slices": nz,
                "is_thicker_than_threshold": sz > args.threshold,
            })

        except Exception as e:
            rows.append({
                "file": str(p),
                "case_id": p.name.replace(".nii.gz", ""),
                "spacing_x": None,
                "spacing_y": None,
                "spacing_z_slice_thickness": None,
                "size_x": None,
                "size_y": None,
                "size_z_slices": None,
                "is_thicker_than_threshold": None,
                "error": str(e),
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    valid = df[df["spacing_z_slice_thickness"].notna()].copy()
    thick = valid[valid["spacing_z_slice_thickness"] > args.threshold]

    print("\n========== 统计结果 ==========")
    print(f"总文件数: {len(df)}")
    print(f"成功读取: {len(valid)}")
    print(f"读取失败: {len(df) - len(valid)}")
    print(f"z-spacing > {args.threshold} mm 的病例数: {len(thick)}")
    print(f"z-spacing <= {args.threshold} mm 的病例数: {len(valid) - len(thick)}")

    print("\n========== z-spacing 描述统计 ==========")
    print(valid["spacing_z_slice_thickness"].describe())

    print("\n========== z-spacing 取值分布 ==========")
    print(valid["spacing_z_slice_thickness"].value_counts().sort_index())

    print(f"\n详细结果已保存到: {args.out_csv}")

    if len(thick) > 0:
        thick_csv = Path(args.out_csv).with_name(
            Path(args.out_csv).stem + f"_gt_{args.threshold}mm.csv"
        )
        thick.to_csv(thick_csv, index=False)
        print(f"层厚 > {args.threshold}mm 的病例列表已保存到: {thick_csv}")


if __name__ == "__main__":
    main()


