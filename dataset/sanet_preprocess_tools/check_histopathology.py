'''
检查Histopathology数据集的层厚信息
'''
from pathlib import Path
import SimpleITK as sitk
import pandas as pd

mhd_root = Path(r"/media/SENSETIME\yangtingting/T7/医保大赛数据/Histopathology/MHD_3D")

rows = []
for p in sorted(mhd_root.glob("*.mhd")):
    img = sitk.ReadImage(str(p))
    sx, sy, sz = img.GetSpacing()
    nx, ny, nz = img.GetSize()
    rows.append({
        "file": p.name,
        "pid": p.stem,
        "spacing_x": sx,
        "spacing_y": sy,
        "spacing_z_slice_thickness": sz,
        "size_x": nx,
        "size_y": ny,
        "size_z_slices": nz,
    })

df = pd.DataFrame(rows)
print(df["spacing_z_slice_thickness"].describe())
print(df["spacing_z_slice_thickness"].value_counts().sort_index())

df.to_csv("histopathology_mhd_spacing_summary.csv", index=False)
print("saved histopathology_mhd_spacing_summary.csv")