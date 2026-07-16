"""训练脚本 - 在华为云 ModelArts 上运行

功能：
- 自动检测 Ascend 设备，回退 CPU
- 使用 NumpySlicesDataset 加载数据
- MSELoss + Adam 优化器
- LossMonitor, TimeMonitor, ModelCheckpoint 回调
- 训练后评估 MSE/RMSE/MAE，导出权重为 numpy JSON

运行方式（项目根目录下）：
    python -m src.train --epochs 200 --lr 0.001 --batch_size 32
"""
import argparse
import json
import os
import sys

import numpy as np

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_data, get_scaler, save_scaler
from src.model import build_model


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="MindSpore 房价预测训练")
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout 概率")
    parser.add_argument("--hidden_dim", type=int, default=128, help="隐藏层维度")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="checkpoint 保存目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def setup_context():
    """自动检测设备：Ascend 优先，回退 CPU"""
    from mindspore import context

    # 尝试使用 Ascend 设备
    try:
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=0)
        # 创建一个小 Tensor 测试 Ascend 是否真正可用
        import mindspore as ms
        _ = ms.Tensor(np.array([1.0]), ms.float32)
        print("检测到 Ascend 设备，使用 Ascend 训练")
    except Exception:
        # 回退 CPU
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
        print("未检测到 Ascend 设备，使用 CPU 训练")


def create_dataset(X, y, batch_size=32, shuffle=True):
    """创建 MindSpore Dataset

    参数:
        X: 特征 numpy 数组 (n, 8)
        y: 标签 numpy 数组 (n,)
        batch_size: 批大小
        shuffle: 是否打乱
    返回:
        MindSpore Dataset 对象
    """
    from mindspore.dataset import NumpySlicesDataset
    dataset = NumpySlicesDataset(
        (X.astype(np.float32), y.astype(np.float32)),
        column_names=["features", "labels"],
        shuffle=shuffle,
    )
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset


def evaluate(net, X_test, y_test, scaler):
    """在测试集上评估模型

    返回:
        dict: {"MSE": ..., "RMSE": ..., "MAE": ...}
    """
    import mindspore as ms

    # 设置为推理模式（关闭 Dropout）
    net.set_train(False)
    # 标准化
    X_scaled = scaler.transform(X_test).astype(np.float32)
    X_tensor = ms.Tensor(X_scaled, ms.float32)
    # 前向推理
    y_pred = net(X_tensor).asnumpy()
    # 计算指标
    mse = float(np.mean((y_pred - y_test) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_pred - y_test)))
    return {"MSE": mse, "RMSE": rmse, "MAE": mae}


def export_weights(net, path):
    """导出模型权重为 numpy JSON（供 Streamlit 推理用）

    参数:
        net: MindSpore 网络
        path: JSON 文件保存路径
    """
    weights = {}
    for name, param in net.parameters_dict().items():
        arr = param.asnumpy()
        weights[name] = arr.tolist()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(weights, f, ensure_ascii=False)
    print(f"权重已保存到 {path}")


def train(args):
    """训练主流程"""
    # 设置随机种子
    np.random.seed(args.seed)

    # 设置设备
    setup_context()

    # 加载数据
    print("正在加载数据...")
    X_train, X_test, y_train, y_test = load_data(seed=args.seed)
    print(f"训练集: {X_train.shape[0]} 条, 测试集: {X_test.shape[0]} 条")

    # 标准化
    scaler = get_scaler(X_train)
    X_train_scaled = scaler.transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    # 创建训练数据集
    train_dataset = create_dataset(
        X_train_scaled, y_train, args.batch_size, shuffle=True
    )

    # 导入 MindSpore 训练组件
    from mindspore import nn, Model
    from mindspore.train.callback import (
        LossMonitor,
        TimeMonitor,
        ModelCheckpoint,
        CheckpointConfig,
    )

    # 构建模型
    net = build_model(
        input_dim=X_train.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    loss_fn = nn.MSELoss()
    opt = nn.Adam(net.trainable_params(), learning_rate=args.lr)
    model = Model(net, loss_fn, opt)

    # 训练回调
    os.makedirs(args.save_dir, exist_ok=True)
    dataset_size = train_dataset.get_dataset_size()
    print(f"数据集大小: {dataset_size} 批次/epoch")
    config_ckpt = CheckpointConfig(
        save_checkpoint_steps=dataset_size,
        keep_checkpoint_max=5,
    )
    ckpt_cb = ModelCheckpoint(
        prefix="house_price",
        directory=args.save_dir,
        config=config_ckpt,
    )
    callbacks = [LossMonitor(), TimeMonitor(), ckpt_cb]

    # 开始训练
    print(f"开始训练: epochs={args.epochs}, lr={args.lr}, "
          f"batch_size={args.batch_size}")
    print("=" * 50)
    # MindSpore 2.4.x 移除了 dataset_sink_mode 参数
    model.train(
        args.epochs,
        train_dataset,
        callbacks=callbacks,
    )
    print("训练完成！")

    # ---- 评估 ----
    print("正在评估模型...")
    metrics = evaluate(net, X_test, y_test, scaler)
    print(f"评估指标: MSE={metrics['MSE']:.4f}, "
          f"RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}")

    # ---- 保存结果 ----
    artifacts_dir = "artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. 保存评估指标
    metrics_path = os.path.join(artifacts_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"指标已保存到 {metrics_path}")

    # 2. 导出模型权重为 numpy JSON
    weights_path = os.path.join(artifacts_dir, "weights.json")
    export_weights(net, weights_path)

    # 3. 保存 scaler 参数
    scaler_path = os.path.join(artifacts_dir, "scaler_params.json")
    save_scaler(scaler, scaler_path)
    print(f"Scaler 参数已保存到 {scaler_path}")

    print("全部完成！")


if __name__ == "__main__":
    args = parse_args()
    train(args)
