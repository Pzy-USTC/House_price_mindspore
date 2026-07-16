"""数据加载与预处理模块

使用 sklearn 的加州房价数据集，提供：
- 数据加载与训练/测试集划分
- StandardScaler 标准化
- scaler 参数的 JSON 保存/加载
- FEATURE_RANGES 字典（供 Streamlit 滑块使用）
"""
import json
import os

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 特征名称（顺序与数据集列一致）
FEATURE_NAMES = [
    "MedInc",       # 收入中位数
    "HouseAge",     # 房龄
    "AveRooms",     # 平均房间数
    "AveBedrooms",  # 平均卧室数
    "Population",   # 人口
    "AveOccup",     # 平均入住人数
    "Latitude",     # 纬度
    "Longitude",    # 经度
]

# 每个特征的 (min, max, default)，用于 Streamlit 滑块
# 范围基于数据集分布，去除极端离群值以提供合理的滑块区间
FEATURE_RANGES = {
    "MedInc":       (0.5, 15.0, 5.0),
    "HouseAge":     (1.0, 52.0, 20.0),
    "AveRooms":     (0.8, 10.0, 5.0),
    "AveBedrooms":  (0.1, 3.0, 1.0),
    "Population":   (100.0, 5000.0, 1500.0),
    "AveOccup":     (0.5, 5.0, 3.0),
    "Latitude":     (32.5, 42.0, 35.5),
    "Longitude":    (-124.0, -114.0, -120.0),
}


def load_data(seed=42):
    """加载加州房价数据集，划分训练/测试集（80/20）

    参数:
        seed: 随机种子
    返回:
        X_train, X_test, y_train, y_test
        特征 shape (n, 8)，目标 shape (n,)（房价，单位十万美元）
    """
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target  # 房价（十万美元）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    return X_train, X_test, y_train, y_test


def get_scaler(X_train):
    """拟合 StandardScaler

    参数:
        X_train: 训练集特征
    返回:
        拟合后的 StandardScaler 对象
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def save_scaler(scaler, path):
    """将 scaler 的 mean 和 scale 保存为 JSON

    参数:
        scaler: StandardScaler 对象
        path: JSON 文件路径
    """
    data = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "var": scaler.var_.tolist(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_scaler(path):
    """从 JSON 加载 scaler 参数

    参数:
        path: JSON 文件路径
    返回:
        字典 {mean, scale, var}，各为 list
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
