"""
kansas_loader.py — Kansas State University Bed-BCG 数据集加载器

数据集结构 (破解后):
  - 文件: Bed_System_Database.mat (MATLAB v7.3 / HDF5)
  - 40 受试者, 每受试者 20 个同步信号, 1000 Hz
  - 在 #refs# 下, 每个受试者的数据是 shape=(20,1) 的 object cell
  - cell 内 20 个引用按固定列名顺序对应信号:
      [0]  PPG        [10] LC_COP1
      [1]  Resp       [11] LC_BCG1
      [2]  HR         [12] LC_COP2
      [3]  ECG  ★     [13] LC_BCG2
      [4]  Film0 ★    [14] LC_COP3
      [5]  Film1 ★    [15] LC_BCG3
      [6]  Film2 ★    [16] reBAP
      [7]  Film3 ★    [17] IBI
      [8]  LC_COP0    [18] SV
      [9]  LC_BCG0 ★  [19] dp_dt

  ★ = BCG→ECG 重建会用到的信号 (ECG 目标 + 4 Film + 4 LC_BCG = 9 个)

元数据 (40 维 float64, 在 #refs# 下):
  - 0G: 年龄 [27, 22, 19, ...]
  - aH: 身高 (cm)
  - bH: 体重 (kg)

用法:
  records = load_kansas(Path("data/kansas_bcg"))
  # records[i] = {subject_id, fs, ecg, bcg_film[4], bcg_lc[4], hr, ppg, resp, ...}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import h5py
except ImportError:
    h5py = None  # 延迟到 load_kansas 调用时再报错

# 固定列名顺序
COLUMN_NAMES = [
    "PPG", "Resp", "HR", "ECG",
    "Film0", "Film1", "Film2", "Film3",
    "LC_COP0", "LC_BCG0", "LC_COP1", "LC_BCG1",
    "LC_COP2", "LC_BCG2", "LC_COP3", "LC_BCG3",
    "reBAP", "IBI", "SV", "dp_dt",
]
IDX = {name: i for i, name in enumerate(COLUMN_NAMES)}
FS = 1000  # Hz


def _read_signal(f, ref):
    """解引用一个对象引用, 返回 1D float64 数组。"""
    ds = f[ref]
    arr = np.asarray(ds[()], dtype=np.float64).squeeze()
    return arr


def _find_subject_cells(f):
    """在 #refs# 下找到所有 (20,1) object cell (每受试者一个)。"""
    refs = f["#refs#"]
    cells = []
    for k in refs.keys():
        o = refs[k]
        if not (isinstance(o, h5py.Dataset) and o.dtype == object):
            continue
        if o.shape != (20, 1):
            continue
        raw = o[()]
        try:
            d = f[raw[0, 0]]
            if isinstance(d, h5py.Dataset) and d.dtype == np.float64 and d.size > 1000:
                cells.append(k)
        except Exception:
            pass
    return cells


def _read_metadata(f):
    """读取 40 维受试者元数据 (年龄/身高/体重)。"""
    refs = f["#refs#"]
    meta = {}
    for key, name in [("0G", "age"), ("aH", "height_cm"), ("bH", "weight_kg")]:
        if key in refs:
            meta[name] = np.asarray(refs[key][()]).ravel().tolist()
    return meta


def load_kansas(root: Path, load_metadata: bool = True) -> list[dict]:
    """加载 Kansas 数据集。

    Args:
        root: 数据集根目录 (含 Dataset Files/Bed_System_Database.mat)
        load_metadata: 是否读取年龄/身高/体重

    Returns:
        list[dict], 每个受试者:
          {"subject_id", "fs", "ecg", "bcg_film": [4], "bcg_lc": [4],
           "ppg", "resp", "hr", "n_samples", ...}
    """
    import h5py
    root = Path(root)
    mat_path = root / "Dataset Files" / "Bed_System_Database.mat"
    if not mat_path.exists():
        # 兼容: 文件直接在 root 下
        mat_path = root / "Bed_System_Database.mat"
    if not mat_path.exists():
        raise FileNotFoundError(
            f"Bed_System_Database.mat not found under {root} "
            f"(looked in Dataset Files/ and root)"
        )

    print(f"[Kansas loader] Opening {mat_path.name} ...")
    f = h5py.File(mat_path, "r")
    cell_keys = _find_subject_cells(f)
    print(f"[Kansas loader] Found {len(cell_keys)} subjects")
    if len(cell_keys) != 40:
        print(f"  WARNING: expected 40 subjects, got {len(cell_keys)}")

    meta = _read_metadata(f) if load_metadata else {}

    records = []
    for i, k in enumerate(sorted(cell_keys), start=1):
        raw = f["#refs#"][k][()]  # (20,1) object refs
        rec = {"subject_id": f"S{i:02d}", "fs": FS,
               "cell_key": k, "n_samples": None}

        # ECG (目标)
        rec["ecg"] = _read_signal(f, raw[IDX["ECG"], 0])
        rec["n_samples"] = len(rec["ecg"])

        # BCG: 4 个 PVDF Film 传感器 + 4 个称重传感器 LC_BCG
        rec["bcg_film"] = [_read_signal(f, raw[IDX[f"Film{j}"], 0]) for j in range(4)]
        rec["bcg_lc"] = [_read_signal(f, raw[IDX[f"LC_BCG{j}"], 0]) for j in range(4)]

        # 其它参考信号
        rec["ppg"] = _read_signal(f, raw[IDX["PPG"], 0])
        rec["resp"] = _read_signal(f, raw[IDX["Resp"], 0])
        rec["hr"] = _read_signal(f, raw[IDX["HR"], 0])
        rec["ibi"] = _read_signal(f, raw[IDX["IBI"], 0])

        # 元数据
        if meta:
            for mk in ["age", "height_cm", "weight_kg"]:
                if mk in meta and i - 1 < len(meta[mk]):
                    rec[mk] = meta[mk][i - 1]

        records.append(rec)
        print(f"  {rec['subject_id']}: n={rec['n_samples']} "
              f"({rec['n_samples']/FS:.1f}s), "
              f"age={rec.get('age','?')}")

    f.close()
    print(f"[Kansas loader] Loaded {len(records)} subjects")
    return records


def summarize_kansas(root: Path) -> dict:
    """探测数据集是否就绪 (不加载信号)。"""
    root = Path(root)
    mat_path = root / "Dataset Files" / "Bed_System_Database.mat"
    if not mat_path.exists():
        mat_path = root / "Bed_System_Database.mat"
    if not mat_path.exists():
        return {"available": False, "reason": f"Bed_System_Database.mat not found under {root}"}
    return {"available": True, "mat_path": str(mat_path), "fs_hz": FS,
            "n_subjects_expected": 40, "signals_per_subject": 20}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/kansas_bcg"))
    a = ap.parse_args()
    summary = summarize_kansas(a.data.resolve())
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary.get("available"):
        print("\nLoading first 2 subjects (quick check)...")
        import h5py
        f = h5py.File(summary["mat_path"], "r")
        cells = _find_subject_cells(f)
        print(f"Found {len(cells)} subject cells")
        # 验证第一个受试者
        raw = f["#refs#"][sorted(cells)[0]][()]
        ecg = _read_signal(f, raw[IDX["ECG"], 0])
        film0 = _read_signal(f, raw[IDX["Film0"], 0])
        print(f"  Subject 1 ECG: shape={ecg.shape}, range=[{ecg.min():.2f},{ecg.max():.2f}]")
        print(f"  Subject 1 Film0: shape={film0.shape}, range=[{film0.min():.2f},{film0.max():.2f}]")
        f.close()