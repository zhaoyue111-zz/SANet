"""Shared competition utilities: runtime env, CSV column helpers, training log.

Merged from `runtime_env_competition`, `gt_columns_competition`,
`training_log_competition`.
"""

from __future__ import annotations

import collections
import datetime
import json
import os
import socket
import warnings
from datetime import timezone
from typing import Dict, IO, Any, Mapping, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Runtime environment (call before heavy imports)
# ---------------------------------------------------------------------------

def apply_runtime_env() -> None:
    os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS") or "1"
    os.environ["MKL_NUM_THREADS"] = os.environ.get("MKL_NUM_THREADS") or "1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    warnings.filterwarnings("ignore", category=FutureWarning)


def _can_bind_port(port: int) -> bool:
    """True if both IPv4 and (when available) IPv6 can bind `port`."""
    import socket

    sockets: list[Any] = []
    try:
        families = [(socket.AF_INET, ("0.0.0.0", port))]
        if getattr(socket, "has_ipv6", False):
            families.append((socket.AF_INET6, ("::", port)))
        for family, addr in families:
            try:
                sock = socket.socket(family, socket.SOCK_STREAM)
            except OSError:
                continue
            sockets.append(sock)
            try:
                if family == socket.AF_INET6:
                    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            except OSError:
                pass
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(addr)
        return True
    except OSError:
        return False
    finally:
        for sock in sockets:
            try:
                sock.close()
            except OSError:
                pass


def find_free_tcp_port() -> int:
    """Ask the OS for an ephemeral free TCP port."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def ensure_master_port(*, preferred: int | None = None, verbose: bool = True) -> int:
    """Ensure `MASTER_PORT` is free for torchrun / DDP.

    - If `MASTER_PORT` is set and free -> keep it.
    - If set but busy -> pick a free port (warn).
    - If unset -> try `preferred` (default 29500), else a free port.
    """
    env_raw = os.environ.get("MASTER_PORT", "").strip()
    candidates: list[int] = []
    if env_raw:
        try:
            candidates.append(int(env_raw))
        except ValueError:
            pass
    if preferred is not None:
        candidates.append(int(preferred))
    candidates.append(29500)

    chosen: int | None = None
    for port in candidates:
        if port <= 0 or port > 65535:
            continue
        if _can_bind_port(port):
            chosen = port
            break

    if chosen is None:
        chosen = find_free_tcp_port()

    prev = env_raw or None
    os.environ["MASTER_PORT"] = str(chosen)
    if verbose and (prev is None or str(chosen) != prev):
        why = "unset" if prev is None else f"busy ({prev})"
        print(f"[ddp] MASTER_PORT={chosen}  ({why} -> auto)")
    return chosen


# ---------------------------------------------------------------------------
# JSON config loader
# ---------------------------------------------------------------------------

def json_file_to_pyobj(filename):
    def _json_object_hook(d):
        return collections.namedtuple("X", d.keys())(*d.values())

    def json2obj(data):
        return json.loads(data, object_hook=_json_object_hook)

    with open(filename, encoding="utf-8") as f:
        return json2obj(f.read())


# ---------------------------------------------------------------------------
# Canonical CSV column names (= input/annotations.csv)
# ---------------------------------------------------------------------------

PATIENT_NAMES = ("patient_id",)
STUDY_NAMES = ("studyInstanceUID",)
SERIES_NAMES = ("seriesInstanceUID",)
LESION_NAMES = ("lesion_id",)
NODULE_NAMES = ("nodule_id",)
BBOX_NAME_PAIRS = (
    ("bbox_min_x",),
    ("bbox_max_x",),
    ("bbox_min_y",),
    ("bbox_max_y",),
    ("bbox_min_z",),
    ("bbox_max_z",),
)


def _mapping(row: Any) -> Mapping[str, Any]:
    if isinstance(row, dict):
        return row
    if hasattr(row, "_asdict"):
        return row._asdict()
    if isinstance(row, pd.Series):
        return row
    return dict(row)


def cell(row: Any, names: str) -> str:
    m = _mapping(row)
    for name in names:
        if name not in m:
            continue
        val = m[name]
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        text = str(val).strip()
        if text and text.lower() not in ("nan", "none", "null"):
            return text
    return ""


def patient_id(row: Any) -> str:
    return cell(row, PATIENT_NAMES)


def study_uid(row: Any) -> str:
    return cell(row, STUDY_NAMES)


def series_uid(row: Any) -> str:
    return cell(row, SERIES_NAMES)


def lesion_id(row: Any) -> str:
    return cell(row, LESION_NAMES)


def nodule_id(row: Any) -> str:
    return cell(row, NODULE_NAMES) or lesion_id(row)


def pick_df_col(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def bbox_xyz(row: Any) -> tuple[float, float, float, float, float, float] | None:
    """xmin, xmax, ymin, ymax, zmin, zmax on the original `image_original` grid."""
    m = _mapping(row)
    vals: list[float] = []
    for names in BBOX_NAME_PAIRS:
        raw = cell(m, names)
        if not raw:
            return None
        try:
            vals.append(float(raw))
        except (TypeError, ValueError):
            return None
    return tuple(vals)  # type: ignore[return-value]


def has_explicit_center(row: Any) -> bool:
    """True when CSV has finite `center_x/y/z`; else use bbox."""
    m = _mapping(row)
    for name in ("center_x", "center_y", "center_z"):
        raw = cell(m, name)
        if not raw:
            return False
        try:
            float(raw)
        except (TypeError, ValueError):
            return False
    return True


def center_xyz(row: Any) -> tuple[float, float, float] | None:
    """Prefer `center_x/y/z`; else bbox midpoint. Returns (x, y, z)."""
    m = _mapping(row)
    if has_explicit_center(m):
        return (
            float(cell(m, "center_x")),
            float(cell(m, "center_y")),
            float(cell(m, "center_z")),
        )
    bb = bbox_xyz(row)
    if bb is None:
        return None
    xmin, xmax, ymin, ymax, zmin, zmax = bb
    return (
        0.5 * (xmin + xmax),
        0.5 * (ymin + ymax),
        0.5 * (zmin + zmax),
    )


def canonicalize_gts_df(df: pd.DataFrame) -> pd.DataFrame:
    """Strip header whitespace; columns must already use `annotations.csv` names."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


# ---------------------------------------------------------------------------
# Competition JSONL training.log
# ---------------------------------------------------------------------------

def utc_timestamp() -> str:
    dt = datetime.datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{int(dt.microsecond / 1000):03d}Z"


def official_data_source(split: str) -> str:
    mapping = {
        "train": "official/train_v1",
        "val": "official/val_v1",
        "test": "official/test_v1",
    }
    if split not in mapping:
        raise ValueError(f"unknown split for data_source: {split!r}")
    return mapping[split]


class TrainingLogCompetition:
    """Append one JSON object per line to `training.log`."""

    def __init__(self, path, pretrained_from: Optional[str] = None):
        self.path = path
        self.pretrained_from = pretrained_from
        self.step = 0
        self._file: Optional[IO[str]] = open(path, "a", encoding="utf-8")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def write(
        self,
        *,
        epoch: Optional[int],
        phase: str,
        mode: str,
        data_source: str,
        step: Optional[int] = None,
        loss: Optional[float] = None,
        lr: Optional[float] = None,
        checkpoint: Optional[str] = None,
        include_pretrained: bool = False,
    ) -> None:
        if self._file is None:
            return
        entry = {
            "timestamp": utc_timestamp(),
            "epoch": epoch,
            "step": self.step if step is None else step,
            "phase": phase,
            "mode": mode,
            "data_source": data_source,
            "loss": float(loss) if loss is not None else None,
            "lr": float(lr) if lr is not None else None,
            "checkpoint": checkpoint,
        }
        if include_pretrained and self.pretrained_from is not None:
            entry["pretrained_from"] = self.pretrained_from
        self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._file.flush()

    def log_phase(
        self,
        *,
        epoch: int,
        phase: str,
        mode: str,
        data_source: str,
        loss: float,
        lr: float,
        checkpoint: Optional[str] = None,
        include_pretrained: bool = False,
    ) -> None:
        self.write(
            epoch=epoch,
            phase=phase,
            mode=mode,
            data_source=data_source,
            loss=loss,
            lr=lr,
            checkpoint=checkpoint,
            include_pretrained=include_pretrained,
        )
        self.step += 1
