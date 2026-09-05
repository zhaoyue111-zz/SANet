"""Weighted GT positive sampling for thinFPR train (annotations.csv lookup).

Each matched category adds ``boost`` (default 5) to sampling weight, stacked:

  1. ``main_lesion == yes`` → +5
  2. ``noduleType == 磨玻璃结节`` and ``detectResult == 恶性`` → +5
  3. ``detectResult == 恶性`` and ``symptoms`` contains ``空洞`` → +5
  4. ``detectResult == 恶性`` and ``symptoms`` contains ``空泡征`` → +5

(``空泡`` is **not** a sign label; only ``空泡征`` counts for rule 4.)

``weight = boost × (#matched categories)``; no match → 1.
Max four categories → weight 20 when ``boost=5``.

Lookup key: ``(patient_id, studyInstanceUID, seriesInstanceUID, lesion_id)``.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dataset import utils_competition as gtscol

DEFAULT_GT_BOOST_FACTOR = 5.0
GGO_NODULE_TYPE = "磨玻璃结节"
CAVITY_SIGN = "空洞"
VACUOLE_SIGN = "空泡征"

ANNOTATION_ID_COLS = (
    "patient_id",
    "studyInstanceUID",
    "seriesInstanceUID",
    "lesion_id",
    "nodule_id",
)
ANNOTATION_BOOST_COLS = ANNOTATION_ID_COLS + (
    "main_lesion",
    "detectResult",
    "noduleType",
    "symptoms",
)


def norm_lesion_id(val: Any) -> str:
    text = gtscol.cell({"lesion_id": val}, gtscol.LESION_NAMES)
    if not text:
        return ""
    try:
        f = float(text)
        if f == int(f):
            return str(int(f))
    except ValueError:
        pass
    return text


def nodule_lookup_key(row: Any) -> tuple[str, str, str, str]:
    lesion = norm_lesion_id(gtscol.lesion_id(row) or gtscol.nodule_id(row))
    return (
        gtscol.patient_id(row),
        gtscol.study_uid(row),
        gtscol.series_uid(row),
        lesion,
    )


def parse_symptoms_cell(raw: Any) -> list[str]:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text or text in ("[]",) or text.lower() in ("nan", "none", "null"):
        return []
    inner = text.strip("[]").replace("'", "").replace('"', "").strip()
    if inner.lower() in ("", "无", "nan", "none", "null"):
        return []
    labels: list[str] = []
    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                labels = [str(x).strip() for x in parsed]
            elif isinstance(parsed, str):
                labels = [parsed.strip()]
        except (SyntaxError, ValueError):
            labels = []
    if not labels:
        labels = [
            t.strip()
            for t in text.replace(";", ",").replace("|", ",").split(",")
            if t.strip()
        ]
    out: list[str] = []
    for lab in labels:
        lab = lab.strip(" []'\"")
        if lab and lab not in ("无", "NEGATIVE"):
            out.append(lab)
    return out


def is_main_lesion(val: Any) -> bool:
    return str(val or "").strip().lower() == "yes"


def is_malignant(val: Any) -> bool:
    return str(val or "").strip() == "恶性"


def gt_boost_reasons(ann_row: pd.Series | None) -> list[str]:
    """Human-readable reasons for boost (empty if none)."""
    if ann_row is None:
        return []
    reasons: list[str] = []
    if is_main_lesion(ann_row.get("main_lesion")):
        reasons.append("main_lesion")
    detect = ann_row.get("detectResult")
    if is_malignant(detect):
        ntype = str(ann_row.get("noduleType", "") or "").strip()
        if ntype == GGO_NODULE_TYPE:
            reasons.append("ggo+malignant")
        symptoms = set(parse_symptoms_cell(ann_row.get("symptoms")))
        if CAVITY_SIGN in symptoms:
            reasons.append("mal+cavity")
        if VACUOLE_SIGN in symptoms:
            reasons.append("mal+vacuole")
    return reasons


def annotation_gt_boost_weight(
    ann_row: pd.Series | None,
    *,
    boost: float = DEFAULT_GT_BOOST_FACTOR,
) -> float:
    """``weight = boost × (#matched categories)``; no match → 1."""
    if ann_row is None:
        return 1.0
    n = len(gt_boost_reasons(ann_row))
    if n == 0:
        return 1.0
    return float(boost) * float(n)


def load_annotation_lookup(path: Path | str | None) -> dict[tuple[str, str, str, str], pd.Series] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        print(f"[gt_boost] warning: annotations csv not found, disabled: {p}")
        return None
    df = pd.read_csv(p)
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in ANNOTATION_ID_COLS if c not in df.columns]
    if missing:
        print(f"[gt_boost] warning: missing id columns {missing}, disabled: {p}")
        return None
    lookup: dict[tuple[str, str, str, str], pd.Series] = {}
    dup = 0
    skipped = 0
    for _, row in df.iterrows():
        key = nodule_lookup_key(row)
        if not key[0] or not key[3]:
            skipped += 1
            continue
        if key in lookup:
            dup += 1
        lookup[key] = row
    print(
        f"[gt_boost] annotations lookup: {len(lookup)} nodules from {p}"
        + (f" ({dup} duplicate keys)" if dup else "")
        + (f" ({skipped} rows skipped missing id)" if skipped else "")
    )
    return lookup


def summarize_gt_boost(
    gts: pd.DataFrame,
    ann_lookup: dict[tuple[str, str, str, str], pd.Series],
    *,
    boost: float = DEFAULT_GT_BOOST_FACTOR,
) -> dict[str, int]:
    """Count train gts rows eligible for each boost reason."""
    stats = {
        "gts_rows": len(gts),
        "matched_ann": 0,
        "boost_rows": 0,
        "main_lesion": 0,
        "ggo+malignant": 0,
        "mal+cavity": 0,
        "mal+vacuole": 0,
        "miss_ann": 0,
        "weight_x5": 0,
        "weight_x10": 0,
        "weight_x15": 0,
        "weight_x20": 0,
    }
    for _, row in gts.iterrows():
        key = nodule_lookup_key(row)
        ann = ann_lookup.get(key)
        if ann is None:
            stats["miss_ann"] += 1
            continue
        stats["matched_ann"] += 1
        reasons = gt_boost_reasons(ann)
        if not reasons:
            continue
        stats["boost_rows"] += 1
        for r in reasons:
            stats[r] += 1
        w = annotation_gt_boost_weight(ann, boost=boost)
        if w >= boost * 4 - 1e-6:
            stats["weight_x20"] += 1
        elif w >= boost * 3 - 1e-6:
            stats["weight_x15"] += 1
        elif w >= boost * 2 - 1e-6:
            stats["weight_x10"] += 1
        elif w >= boost - 1e-6:
            stats["weight_x5"] += 1
    print(
        f"[gt_boost] train gts={stats['gts_rows']} ann_match={stats['matched_ann']} "
        f"boost_rows={stats['boost_rows']} "
        f"(main={stats['main_lesion']} ggo+mal={stats['ggo+malignant']} "
        f"mal+空洞={stats['mal+cavity']} mal+空泡征={stats['mal+vacuole']} "
        f"weight×5={stats['weight_x5']} ×10={stats['weight_x10']} "
        f"×15={stats['weight_x15']} ×20={stats['weight_x20']} "
        f"miss_ann={stats['miss_ann']})"
    )
    return stats


def sample_locs_weighted(
    locs: np.ndarray,
    weights: np.ndarray,
    num: int,
) -> np.ndarray:
    """Weighted random choice of positive patch centers (with replacement if needed)."""
    if num <= 0 or len(locs) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape[0] != len(locs):
        raise ValueError(f"weights length {w.shape[0]} != locs {len(locs)}")
    if not np.all(np.isfinite(w)) or np.all(w <= 0):
        replace = len(locs) < num
        idx = np.random.choice(len(locs), size=num, replace=replace)
        return locs[idx]
    p = w / w.sum()
    replace = len(locs) < num
    idx = np.random.choice(len(locs), size=num, replace=replace, p=p)
    return locs[idx]
