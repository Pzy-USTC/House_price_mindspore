# House_price_mindspore

基于华为 MindSpore 框架的加州房价预测系统。

## 项目概述

使用 MindSpore 构建带自注意力的多层感知机（MLPAttention），在加州房价数据集（sklearn 内置，20640 条，8 特征）上训练。训练在华为云 ModelArts 平台完成，推理用纯 numpy 实现，通过 Streamlit 提供可视化交互界面。

## 技术栈

| 组件 | 技术 |
|------|------|
| 深度学习框架 | MindSpore（华为昇思） |
| 训练平台 | 华为云 ModelArts（Notebook + Ascend 训练作业） |
| 数据集 | sklearn.datasets.fetch_california_housing |
| 前端 | Streamlit |
| 其他 | numpy, scikit-learn, plotly, pandas |

## 项目结构

```
House_price_mindspore/
├── src/
│   ├── __init__.py
│   ├── data.py          # 数据加载与预处理
│   ├── model.py         # MindSpore 模型定义（MLPAttention）
│   └── train.py         # 训练脚本（ModelArts 上运行）
├── predict.py           # 推理模块（纯 numpy，供 Streamlit 使用）
├── app.py               # Streamlit 可视化前端
├── requirements.txt
├── .gitignore
├── checkpoints/         # 模型 checkpoint（.ckpt）
├── artifacts/           # 训练产物（weights.json, metrics.json, scaler_params.json）
└── README.md
```

## 模型结构

**MLPAttention（带自注意力的多层感知机）：**

1. **编码器**：`Dense(8→128) → ReLU → Dropout(0.2) → Dense(128→128) → ReLU`
2. **注意力**：对编码特征做 Q/K/V 投影，单 token 简化版，`Sigmoid(q*k)` 门控
3. **融合**：拼接 `[原始编码, 注意力输出] → Dense(256→128) → ReLU → Dropout → Dense(128→1)`
4. **输出**：`.squeeze(-1)`

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> 注意：MindSpore 需在华为云 ModelArts 环境或本地安装对应版本。

### 2. 训练模型（在 ModelArts 上运行）

```bash
python -m src.train --epochs 200 --lr 0.001 --batch_size 32
```

训练完成后，`artifacts/` 目录下会生成：
- `weights.json` — 模型权重（numpy JSON，供推理用）
- `metrics.json` — 评估指标（MSE/RMSE/MAE）
- `scaler_params.json` — StandardScaler 参数

### 3. 启动可视化界面

```bash
streamlit run app.py
```

## 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 200 | 训练轮数 |
| `--lr` | 0.001 | 学习率 |
| `--batch_size` | 32 | 批大小 |
| `--dropout` | 0.2 | Dropout 概率 |
| `--hidden_dim` | 128 | 隐藏层维度 |
| `--save_dir` | checkpoints | checkpoint 保存目录 |
| `--seed` | 42 | 随机种子 |

## 注意事项

1. MindSpore API：`nn.Dense`（非 `nn.Linear`）、`nn.Cell`（非 `nn.Module`）、`construct`（非 `forward`）
2. 训练在华为云 ModelArts 上运行，脚本自动检测设备（Ascend 优先，回退 CPU）
3. `predict.py` 纯 numpy 实现，Streamlit 运行环境无需安装 MindSpore
4. 代码注释使用中文
# House_price_mindspore