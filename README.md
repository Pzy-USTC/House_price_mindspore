# House_price_mindspore

基于华为 MindSpore 框架的加州房价预测系统，支持云端训练、本地可视化和嵌入式设备部署。

## 项目概述

使用 MindSpore 构建带自注意力的多层感知机（MLPAttention），在加州房价数据集（sklearn 内置，20640 条，8 特征）上训练。训练在华为云 ModelArts 平台完成，推理用纯 numpy 实现，提供 Streamlit 和 Flask 两种可视化前端，可部署到 PC 或香橙派等嵌入式设备。

## 技术栈

| 组件 | 技术 |
|------|------|
| 深度学习框架 | MindSpore 2.4.10（华为昇思） |
| 训练平台 | 华为云 ModelArts（Notebook + Ascend 910） |
| 数据集 | sklearn.datasets.fetch_california_housing |
| 可视化前端 | Streamlit / Flask |
| 推理部署 | 纯 numpy（跨平台，无需 MindSpore） |
| 其他依赖 | numpy, scikit-learn, plotly, pandas |

## 项目结构

```
House_price_mindspore/
├── src/
│   ├── __init__.py
│   ├── data.py          # 数据加载与预处理
│   ├── model.py         # MindSpore 模型定义（MLPAttention）
│   └── train.py         # 训练脚本（ModelArts 上运行）
├── predict.py           # 推理模块（纯 numpy，供 Streamlit 使用）
├── web_server.py        # Flask 轻量级 Web 服务（适合香橙派）
├── app.py               # Streamlit 可视化前端（完整版）
├── requirements.txt
├── .gitignore
├── 实验报告.md           # 实验报告（含部署说明和界面截图）
├── checkpoints/         # 模型 checkpoint（.ckpt）
├── artifacts/           # 训练产物（weights.json, metrics.json, scaler_params.json）
└── README.md
```

## 模型结构

**MLPAttention（带自注意力的多层感知机）：**

```
输入 (batch, 8)
    │
    ▼
┌─────────────────────────────┐
│      编码器 Encoder         │
│  Dense(8→128) → ReLU        │
│  → Dropout(0.2)             │
│  → Dense(128→128) → ReLU    │
└─────────────┬───────────────┘
              │ (batch, 128)
              ▼
──────────────────────────────┐
│     注意力模块 Attention   │
│  Q = Dense(encoded)        │
│  K = Dense(encoded)        │
│  V = Dense(encoded)        │
│  attn = Sigmoid(Q*K) * V   │
└─────────────┬──────────────┘
              │ (batch, 128)
              ▼
┌─────────────────────────────┐
│         融合层 Fusion        │
│  concat[encoded, attn]      │
│  → Dense(256→128) → ReLU    │
│  → Dropout(0.2)             │
│  → Dense(128→1)             │
└────────────────────────────┘
              │
              ▼
         输出 (batch,)
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> 注意：MindSpore 需在华为云 ModelArts 环境或本地安装对应版本。推理端（predict.py / web_server.py）仅需 numpy，无需 MindSpore。

### 2. 训练模型（在 ModelArts 上运行）

```bash
python -m src.train --epochs 200 --lr 0.001 --batch_size 32
```

训练完成后，`artifacts/` 目录下会生成：

| 文件 | 说明 |
|------|------|
| `weights.json` | 模型权重（numpy JSON，约 2.1 MB） |
| `metrics.json` | 评估指标（MSE/RMSE/MAE） |
| `scaler_params.json` | StandardScaler 参数 |

### 3. 启动可视化界面

**Streamlit 完整版**（PC 端，功能完整）：

```bash
streamlit run app.py
```

**Flask 轻量版**（香橙派等嵌入式设备）：

```bash
pip install flask
python web_server.py --port 8080
```

浏览器访问 `http://localhost:8080`

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

## 香橙派部署

### 步骤一：从 ModelArts 下载训练产物

```bash
cd House_price_mindspore
tar -czf artifacts.tar.gz artifacts/
# 通过 JupyterLab 界面下载
```

### 步骤二：传输文件到香橙派

```bash
scp -r web_server.py predict.py src/ artifacts/ user@香橙派IP:/home/user/House_price/
```

### 步骤三：香橙派上启动

```bash
pip3 install numpy flask
cd /home/user/House_price
python3 web_server.py --port 8080
```

### 步骤四：浏览器访问

```
http://香橙派IP:8080
```

## 注意事项

1. MindSpore API：`nn.Dense`（非 `nn.Linear`）、`nn.Cell`（非 `nn.Module`）、`construct`（非 `forward`）
2. 训练在华为云 ModelArts 上运行，脚本自动检测设备（Ascend 优先，回退 CPU）
3. `predict.py` 纯 numpy 实现，Streamlit 运行环境无需安装 MindSpore
4. 代码注释使用中文
5. MindSpore 2.4.10 已移除 `dataset_sink_mode` 参数，训练脚本已适配
