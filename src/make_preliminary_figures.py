from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pipeline_utils import POSTURE_NAMES


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "v1"
RESULTS = ROOT / "experiments" / "preliminary"
FIGURES = RESULTS / "figures"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def quality_control_figure() -> None:
    with np.load(DATA / "subjects" / "S001.npz") as z:
        score = (
            z["quality"][:, 0] * 10.0
            + z["quality"][:, 8] * 5.0
            + np.nan_to_num(z["quality"][:, 1], nan=1.0)
        )
        candidates = np.flatnonzero(z["pressure_valid"])
        index = int(candidates[np.argmin(score[candidates])])
        bcg = z["bcg"][index]
        ecg = z["ecg_100"][index]
        ppg = z["ppg"][index]
        rpeaks = np.flatnonzero(z["rpeak_mask_100"][index])
        pressure_map = z["pressure_mean_map"][index]

    start, stop = 0, 1500
    time = np.arange(stop - start) / 100.0
    fig = plt.figure(figsize=(13, 9))
    grid = fig.add_gridspec(4, 2, width_ratios=[3.2, 1.2], hspace=0.45, wspace=0.28)
    axes = [fig.add_subplot(grid[row, 0]) for row in range(4)]
    labels = [
        ("BCG/胸廓压力：稳健去趋势", bcg[0]),
        ("呼吸分量 0.08–0.70 Hz", bcg[1]),
        ("心搏候选分量 0.80–8.00 Hz", bcg[2]),
        ("PPG-IR 0.40–8.00 Hz", ppg[0]),
    ]
    colors = ["#355070", "#2a9d8f", "#e76f51", "#8a5a9b"]
    for axis, (label, values), color in zip(axes, labels, colors):
        axis.plot(time, values[start:stop], color=color, linewidth=0.8)
        axis.set_ylabel(label)
        axis.grid(alpha=0.15)
    axes[-1].set_xlabel("时间（秒）")

    ecg_axis = fig.add_subplot(grid[0:2, 1])
    ecg_axis.plot(time, ecg[start:stop], color="#bc4749", linewidth=0.8)
    visible = rpeaks[(rpeaks >= start) & (rpeaks < stop)]
    ecg_axis.scatter(
        (visible - start) / 100.0,
        ecg[visible],
        s=14,
        color="#003049",
        label="R峰",
        zorder=3,
    )
    ecg_axis.set_title("ECG 与自动 R 峰（15秒）")
    ecg_axis.legend(frameon=False)
    ecg_axis.grid(alpha=0.15)

    pressure_axis = fig.add_subplot(grid[2:4, 1])
    image = pressure_axis.imshow(pressure_map, cmap="magma", aspect="auto")
    pressure_axis.set_title("30秒平均压力分布（硬件映射后）")
    pressure_axis.set_xlabel("阵列列")
    pressure_axis.set_ylabel("阵列行")
    fig.colorbar(image, ax=pressure_axis, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"预处理质量示例：匿名受试者 S001，窗口 {index + 1}",
        fontsize=15,
        y=0.98,
    )
    fig.savefig(FIGURES / "preprocessing_quality_example.png", bbox_inches="tight")
    plt.close(fig)


def screening_summary_figure() -> None:
    classical = json.loads(
        (RESULTS / "classical_results.json").read_text(encoding="utf-8")
    )
    deep = json.loads((RESULTS / "deep_results.json").read_text(encoding="utf-8"))
    subgroup = json.loads(
        (RESULTS / "subgroup_results.json").read_text(encoding="utf-8")
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    hr_labels = ["固定频谱峰", "谐波/自相关最佳", "BCG工程特征", "训练集心率中位数", "PPG峰检测"]
    spectral = json.loads(
        (RESULTS / "spectral_results.json").read_text(encoding="utf-8")
    )
    hr_values = [
        classical["direct_heart_rate"]["bcg_cardiac_psd_0.8_3Hz"]["mae_bpm"],
        spectral["all"]["autocorrelation"]["mae_bpm"],
        classical["learned_heart_rate"]["bcg_extra_trees"]["mae_bpm"],
        subgroup["bcg_leakage_check"]["train_median_dummy_groupkfold"]["mae_bpm"],
        classical["direct_heart_rate"]["ppg_ir_peak_detector"]["mae_bpm"],
    ]
    axes[0, 0].barh(
        hr_labels,
        hr_values,
        color=["#d1495b", "#d1495b", "#edae49", "#777777", "#2a9d8f"],
    )
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel("心率 MAE（bpm，越低越好）")
    axes[0, 0].set_title("心率估计筛选")
    for row, value in enumerate(hr_values):
        axes[0, 0].text(value + 0.3, row, f"{value:.1f}", va="center")

    event_labels = ["无信号周期基线", "BCG", "BCG+压力", "BCG+压力+蒸馏", "PPG教师"]
    event_values = [
        subgroup["periodic_event_no_signal_baseline"]["event_f1_100ms"],
        deep["event_detection"]["bcg_event"]["test_all"]["event_f1_100ms"],
        deep["event_detection"]["bcg_pressure_event"]["test_all"]["event_f1_100ms"],
        deep["event_detection"]["bcg_pressure_distilled"]["test_all"]["event_f1_100ms"],
        deep["event_detection"]["ppg_teacher"]["test_all"]["event_f1_100ms"],
    ]
    axes[0, 1].bar(
        event_labels,
        event_values,
        color=["#777777", "#d1495b", "#edae49", "#9b5de5", "#2a9d8f"],
    )
    axes[0, 1].set_ylim(0, 0.8)
    axes[0, 1].set_ylabel("R峰 F1（±100 ms）")
    axes[0, 1].set_title("事件级深度模型")
    axes[0, 1].tick_params(axis="x", rotation=22)
    for row, value in enumerate(event_values):
        axes[0, 1].text(row, value + 0.02, f"{value:.2f}", ha="center")

    waveform_labels = ["线性BCG", "深度BCG", "深度BCG+压力", "线性PPG", "深度PPG"]
    waveform_values = [
        classical["linear_waveform_mapping"]["bcg_linear"]["median_direct_correlation"],
        deep["waveform_reconstruction"]["bcg_waveform"]["test_all"][
            "median_direct_correlation"
        ],
        deep["waveform_reconstruction"]["bcg_pressure_waveform"]["test_all"][
            "median_direct_correlation"
        ],
        classical["linear_waveform_mapping"]["ppg_linear_upper_bound"][
            "median_direct_correlation"
        ],
        deep["waveform_reconstruction"]["ppg_waveform_upper_bound"]["test_all"][
            "median_direct_correlation"
        ],
    ]
    axes[1, 0].bar(
        waveform_labels,
        waveform_values,
        color=["#d1495b", "#d1495b", "#edae49", "#2a9d8f", "#2a9d8f"],
    )
    axes[1, 0].axhline(0, color="#333333", linewidth=0.8)
    axes[1, 0].set_ylim(-0.04, 0.20)
    axes[1, 0].set_ylabel("ECG 波形中位 Pearson r")
    axes[1, 0].set_title("跨模态波形重建")
    axes[1, 0].tick_params(axis="x", rotation=22)
    for row, value in enumerate(waveform_values):
        axes[1, 0].text(row, value + 0.008, f"{value:.2f}", ha="center")

    matrix = np.asarray(classical["pressure_posture"]["confusion_matrix"], dtype=float)
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    image = axes[1, 1].imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    axes[1, 1].set_xticks(range(5), POSTURE_NAMES, rotation=25, ha="right")
    axes[1, 1].set_yticks(range(5), POSTURE_NAMES)
    axes[1, 1].set_xlabel("预测体位")
    axes[1, 1].set_ylabel("真实体位")
    axes[1, 1].set_title(
        f"压力阵列体位识别（平衡准确率 "
        f"{classical['pressure_posture']['balanced_accuracy']:.2f}）"
    )
    for i in range(5):
        for j in range(5):
            axes[1, 1].text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] > 0.5 else "#222222",
                fontsize=9,
            )
    fig.colorbar(image, ax=axes[1, 1], fraction=0.046, pad=0.04)
    fig.suptitle("粗略预实验路线筛选汇总", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIGURES / "screening_summary.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    quality_control_figure()
    screening_summary_figure()
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
