"""推理模块 - 纯 numpy 实现，不依赖 MindSpore

从 weights.json 加载权重，用 numpy 做前向传播，
提供 predict_price(raw_features, artifacts_dir) 端到端函数，
供 Streamlit 前端调用。

模型结构与 src/model.py 的 MLPAttention 完全一致：
- 编码器：Dense(8→128) → ReLU → Dropout(推理无操作) → Dense(128→128) → ReLU
- 注意力：Q/K/V 投影，Sigmoid(q*k) 门控
- 融合：concat[编码, 注意力] → Dense(256→128) → ReLU → Dropout(无操作) → Dense(128→1)
"""
import json
import os

import numpy as np


# ---------------------------------------------------------------------------
# numpy 基础运算
# ---------------------------------------------------------------------------

def _dense(x, w, b):
    """全连接层：y = x @ w.T + b

    MindSpore nn.Dense 的权重 shape 为 (out, in)，
    前向计算 y = x @ w.T + b，这里用 numpy 复现。
    """
    return x @ w.T + b


def _relu(x):
    """ReLU 激活函数"""
    return np.maximum(x, 0)


def _sigmoid(x):
    """Sigmoid 激活函数（带数值裁剪防溢出）"""
    x = np.clip(x, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# 前向传播
# ---------------------------------------------------------------------------

def _forward(weights, x):
    """前向传播（纯 numpy 实现）

    参数:
        weights: dict，键名为参数名，值为 list/numpy 数组
        x: numpy 数组，shape (batch, 8) 或 (8,)

    返回:
        numpy 数组，shape (batch,)
    """
    # 确保输入为 2D
    if x.ndim == 1:
        x = x.reshape(1, -1)

    # ---- 编码器 ----
    # Dense(8→128) → ReLU → Dropout(推理无操作) → Dense(128→128) → ReLU
    h = _dense(x,
               np.array(weights["enc_fc1.weight"]),
               np.array(weights["enc_fc1.bias"]))
    h = _relu(h)
    # Dropout 在推理时无操作
    h = _dense(h,
               np.array(weights["enc_fc2.weight"]),
               np.array(weights["enc_fc2.bias"]))
    h = _relu(h)
    encoded = h

    # ---- 注意力（单token简化版，Sigmoid 门控）----
    q = _dense(encoded,
               np.array(weights["q_proj.weight"]),
               np.array(weights["q_proj.bias"]))
    k = _dense(encoded,
               np.array(weights["k_proj.weight"]),
               np.array(weights["k_proj.bias"]))
    v = _dense(encoded,
               np.array(weights["v_proj.weight"]),
               np.array(weights["v_proj.bias"]))
    attn = _sigmoid(q * k) * v

    # ---- 融合 ----
    # concat[encoded, attn] → Dense(256→128) → ReLU → Dropout(无操作) → Dense(128→1)
    combined = np.concatenate([encoded, attn], axis=-1)
    h = _dense(combined,
               np.array(weights["fus_fc1.weight"]),
               np.array(weights["fus_fc1.bias"]))
    h = _relu(h)
    # Dropout 在推理时无操作
    h = _dense(h,
               np.array(weights["fus_fc2.weight"]),
               np.array(weights["fus_fc2.bias"]))
    # squeeze(-1)：(batch, 1) → (batch,)
    out = h.squeeze(-1)
    return out


# ---------------------------------------------------------------------------
# 端到端预测
# ---------------------------------------------------------------------------

def predict_price(raw_features, artifacts_dir="artifacts"):
    """端到端房价预测函数

    参数:
        raw_features: 原始特征值，shape (8,) 单条 或 (n, 8) 批量
        artifacts_dir: 权重和 scaler 参数目录

    返回:
        单条输入返回 float（预测房价，十万美元）
        批量输入返回 numpy 数组 (n,)
    """
    # 加载权重
    weights_path = os.path.join(artifacts_dir, "weights.json")
    with open(weights_path, "r", encoding="utf-8") as f:
        weights = json.load(f)

    # 加载 scaler 参数
    scaler_path = os.path.join(artifacts_dir, "scaler_params.json")
    with open(scaler_path, "r", encoding="utf-8") as f:
        scaler = json.load(f)

    # 转为 numpy 数组
    x = np.array(raw_features, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    # 标准化：(x - mean) / scale
    mean = np.array(scaler["mean"], dtype=np.float32)
    scale = np.array(scaler["scale"], dtype=np.float32)
    x_scaled = (x - mean) / scale

    # 前向传播
    pred = _forward(weights, x_scaled)

    # 单条返回标量，批量返回数组
    if pred.size == 1:
        return float(pred.item())
    return pred


# ---------------------------------------------------------------------------
# 命令行测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 使用默认特征值测试
    from src.data import FEATURE_NAMES, FEATURE_RANGES

    default_features = np.array(
        [FEATURE_RANGES[name][2] for name in FEATURE_NAMES],
        dtype=np.float32,
    )
    print("输入特征:")
    for name, val in zip(FEATURE_NAMES, default_features):
        print(f"  {name}: {val}")

    pred = predict_price(default_features, "artifacts")
    print(f"\n预测房价: {pred:.4f} 十万美元 ({pred * 10:.2f} 万美元)")
