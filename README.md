# BCG2ECG

床垫式无感监测的 BCG→ECG 重建研究项目。

## 项目目标

从床垫式传感器（BCG 心冲击图、阵列压力、PPG）重建标准 ECG 信号，
实现无感、非接触的连续心电监测。

## 目录结构

```
├── src/                    # 全部源代码
│   ├── preprocess.py       # 数据预处理 (raw → processed)
│   ├── pipeline_utils.py   # 信号处理工具库
│   ├── pressure_encoder.py # 压力阵列深度编码器 (三分支 CNN + FiLM)
│   ├── kansas_loader.py    # Kansas 公共 BCG 数据集加载器
│   ├── experiments/        # 实验脚本 (PPG→ECG, Kansas 复现等)
│   ├── exploration/        # 诊断探索脚本
│   └── _archive/           # 废弃的重复实现
├── experiments/            # 实验结果 (JSON + 图表)
├── data/                   # 信号数据 (不入库, gitignore)
└── *.md                    # 研究方向与实验报告文档
```

## 核心发现

- **采样率决定可行性**: 100Hz BCG (PCC≈0) vs 250Hz Kansas BCG (PCC=0.85)
- **Med2ECG 复现成功**: 在 Kansas 数据集上达到 PCC 0.85, RRI MAE 11.9ms
- **增强损失超越原版**: 可微时延对齐 + 频域损失, 经偏移校正后 PCC 0.90
- **受试者方差 > 方法差异**: LOSO 单折 -0.18 到 0.85, 跨人泛化是核心难题

## 依赖

见 `requirements-preliminary.txt`。核心: numpy, scipy, scikit-learn, torch。
