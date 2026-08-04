# src/ 目录结构与探索思路

## 探索脉络

本项目的研究主线是 **床垫式无感监测的 BCG→ECG 重建**。
经过多轮诊断，发现核心瓶颈在 BCG 信号质量本身，遂转向以 PPG 为核心的方向。
本目录记录了完整的探索过程。

```
课题起点：BCG→ECG 重建 (压力阵列条件化)
    │
    ├─ 阶段1：波形重建实验 (experiment_conditioned / exp_cond)
    │      结果：所有条件化策略 direct r ≈ 0 (随机水平)
    │
    ├─ 阶段2：R峰事件检测 (exp_phase2)
    │      结果：F1 ≈ 0.37，压力条件化仅提升 1.4%
    │
    ├─ 阶段3：根因诊断 (diagnosis / explore_raw_bcg / hr_feasibility)
    │      结论：BCG 心搏信号在硬件层面极弱 (p-p=0.046, 比特效率9%)
    │            4通道 BCG 全部与真实心率零相关 (r≈0)
    │            同方法在 PPG 上 r=0.64 → 问题在数据非模型
    │
    └─ 阶段4：转向 PPG (explore_ppg)
           发现：PPG 心率估计优秀 (MAE=4.9, r=0.68)
                 PTT 延迟清晰 (190ms, 76.7% 正向)
                 但波形重建仍困难 (r=0.08, 线性失败)
           │
           └─ 角度C：PPG→ECG 深度波形重建 (experiments/ppg_to_ecg.py)
                  目标：探索非线性深度模型能否突破线性上限
```

## 目录结构

### 核心库 (根目录)
基础设施，被所有实验复用。

| 文件 | 用途 |
|------|------|
| `pipeline_utils.py` | 信号预处理工具 (滤波/标准化/R峰检测/特征提取) |
| `pressure_processing.py` | 压力阵列 1056→77×32 硬件映射 + OTSU 分割 |
| `preprocess.py` | 数据预处理主入口：raw → processed/v1/*.npz |
| `pressure_encoder.py` | 压力阵列深度编码器 (三分支 CNN + FiLM 条件化) |

### 原始筛选实验 (根目录)
课题初期的预实验，验证各模态的可行性基线。

| 文件 | 用途 |
|------|------|
| `classical_screen.py` | 经典 ML 筛选 (Ridge/ExtraTrees/PCA) |
| `deep_screen.py` | 深度学习筛选 (TinyTCN + FiLM + 蒸馏) |
| `spectral_screen.py` | 纯信号处理心率估计 (4种频谱法) |
| `subgroup_screen.py` | 子群分析 + 泄漏检查 |
| `make_preliminary_figures.py` | 预实验结果可视化 |
| `visualize_preprocessing.py` | 预处理流程审计图 |

### experiments/ — 新研究方向实验
转向 PPG 后的正式实验脚本。

| 文件 | 用途 |
|------|------|
| `exp_cond.py` | (BCG阶段) 压力条件化消融 — 保留作历史对照 |
| `ppg_to_ecg.py` | **角度C**：PPG→ECG 深度波形重建可行性 |
| `kansas_bcg_to_ecg.py` | **路径1**：Kansas 公共数据集上的 BCG→ECG 基线 (对标文献) |

### 路径1：Kansas 公共数据集验证 (进行中)

诊断确认我们的 BCG 失败是数据问题 (100Hz + 心搏分量极弱)，
非方法问题。为排除方法架构嫌疑，在标准 1kHz BCG 公共数据集上验证：

- **数据集**: Kansas State University Bed-BCG (Carlson 2020, Sensors)
  - 40 人, ECG+BCG+PPG 同步, 1000 Hz
  - IEEE DataPort: doi:10.21227/77hc-py84
  - 下载到: `data/kansas_bcg/`
- **加载器**: `kansas_loader.py` (自适应 WFDB/MAT/CSV 三种格式)
- **实验**: `experiments/kansas_bcg_to_ecg.py`
  - A. TCN 基线 (复现我们的架构)
  - B. BiLSTM (对标 Zhang 2024 / Morokuma 2025)
  - LOSO 验证, 对标 PCC 0.896-0.984 / RRI MAE 34ms

**运行** (数据下载后):
```bash
python src/experiments/kansas_bcg_to_ecg.py
```

**复现结果** (`kansas_reproduce.py`, 10s 分段, 多通道 BCG, LOSO 5 折, 30 epochs, GPU):

| 模型 | median PCC | delay-tol PCC | RRI MAE | 文献对标 |
|------|:----------:|:------------:|:-------:|----------|
| TCN (w=64) | 0.226 | 0.329 | 192ms | — |
| **BiLSTM** | **0.513** | **0.620** | 159ms | BiLSTM 0.896 / BiM-Diff 0.984 / Morokuma RRI 34ms |

逐折 PCC (BiLSTM): [0.79, 0.36, 0.51, 0.58, 0.19] — 受试者间方差大, fold1 已接近文献 0.896。

结论: 我们的 BiLSTM 在标准 1kHz BCG 上达到 PCC 0.51 (最佳折 0.79),
**确证方法架构正确**; 与文献 0.9 的差距来自未实现的高级技术
(MoE/扩散/医学损失/更长训练)。我们数据 BCG 失败 (PCC≈0.008) 纯属 100Hz 采样率+心搏分量极弱。

### exploration/ — 诊断探索脚本
用于定位问题的临时分析，结论已写入本 README。

| 文件 | 用途 |
|------|------|
| `diagnosis_analysis.py` | BCG 频谱质量 + 通道贡献诊断 |
| `explore_raw_bcg.py` | 原始 BCG 信号质量 (比特效率/极性/互相关) |
| `explore_ppg.py` | PPG 信号质量全面评估 (HR/波形/PTT/体位) |
| `hr_feasibility.py` | 手工特征 + ExtraTrees 心率回归可行性 |

### _archive/ — 废弃的重复实现
被新版本取代，保留备查。

| 文件 | 状态 |
|------|------|
| `experiment_conditioned.py` | 被 `experiments/exp_cond.py` 取代 (有 bug) |
| `exp_phase2.py` | 事件检测实验，结论已并入探索脉络 |

## 运行约定

- 所有脚本从项目根目录运行：`python src/<script>.py`
- 核心库之间用 `sys.path.insert(0, str(Path(__file__).resolve().parent))` 处理导入
- 子目录脚本需向上多插一级 parent：见 `experiments/ppg_to_ecg.py` 的导入段
