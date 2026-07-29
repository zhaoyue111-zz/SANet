
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import pandas as pd

root = Path(r"/media/SENSETIME\yangtingting/T7/医保大赛数据/LNDb/LNDb")

rows = []

for p in sorted(root.glob("*_ct.nii.gz")):
    img = sitk.ReadImage(str(p))
    arr = sitk.GetArrayFromImage(img)

    rows.append({
        "file": p.name,
        "dtype": str(arr.dtype),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "p0.5": float(np.percentile(arr, 0.5)),
        "p1": float(np.percentile(arr, 1)),
        "p50": float(np.percentile(arr, 50)),
        "p99": float(np.percentile(arr, 99)),
        "p99.5": float(np.percentile(arr, 99.5)),
        "spacing": img.GetSpacing(),
        "size": img.GetSize(),
    })

df = pd.DataFrame(rows)
print(df[["file", "dtype", "min", "max", "p0.5", "p99.5"]].head())
print("\n整体范围：")
print(df[["min", "max", "p0.5", "p99.5"]].describe())

df.to_csv("lndb_ct_intensity_range.csv", index=False)