"""Streamlit 可视化前端

基于 MindSpore 房价预测系统的可视化交互界面。
四个区域：
1. 预测结果 - 大字号显示预测房价
2. 模型评估 - MSE/RMSE/MAE 指标卡片
3. 特征重要性 - 扰动法计算 + plotly 水平柱状图
4. 数据集概览 - 目标变量分布直方图 + 特征相关性热力图

运行方式（项目根目录下）：
    streamlit run app.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.datasets import fetch_california_housing

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import FEATURE_NAMES, FEATURE_RANGES
from predict import predict_price

# ---------------------------------------------------------------------------
# 页面配置
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="房价预测系统",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 加州房价预测系统")
st.markdown("基于华为 MindSpore 深度学习框架的房价预测可视化系统")

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

ARTIFACTS_DIR = "artifacts"
WEIGHTS_PATH = os.path.join(ARTIFACTS_DIR, "weights.json")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")

# 检查模型权重是否存在
has_weights = os.path.exists(WEIGHTS_PATH)


# ---------------------------------------------------------------------------
# 侧边栏：特征滑块
# ---------------------------------------------------------------------------

st.sidebar.header("输入特征")
st.sidebar.markdown("调整以下特征值进行房价预测")

input_features = {}
for name in FEATURE_NAMES:
    lo, hi, default = FEATURE_RANGES[name]
    # 计算步长：范围的 1/100
    step = round((hi - lo) / 100.0, 4)
    if step == 0:
        step = 0.01
    input_features[name] = st.sidebar.slider(
        name,
        min_value=float(lo),
        max_value=float(hi),
        value=float(default),
        step=step,
    )

# 转为特征数组（按 FEATURE_NAMES 顺序）
raw_features = np.array(
    [input_features[name] for name in FEATURE_NAMES],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# 区域一：预测结果
# ---------------------------------------------------------------------------

st.subheader("预测结果")

if has_weights:
    pred = predict_price(raw_features, ARTIFACTS_DIR)
    # 大字号显示预测房价
    st.markdown(
        f"<div style='text-align:center; padding:30px; "
        f"background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); "
        f"border-radius:15px; color:white; margin-bottom:20px;'>"
        f"<h1 style='font-size:48px; margin:0;'>💰 {pred * 10:.2f} 万美元</h1>"
        f"<p style='font-size:18px; margin-top:10px; opacity:0.9;'>"
        f"预测房价 = {pred:.4f} 十万美元"
        f"</p></div>",
        unsafe_allow_html=True,
    )

    # 展示当前输入特征
    with st.expander("当前输入特征详情", expanded=False):
        feat_df = pd.DataFrame(
            {"特征": FEATURE_NAMES, "值": [raw_features[i] for i in range(8)]}
        )
        st.dataframe(feat_df, use_container_width=True, hide_index=True)
else:
    st.warning(
        "⚠️ 模型权重未找到。请先在 ModelArts 上运行训练脚本 "
        "(`python -m src.train`)，生成 `artifacts/weights.json` 后刷新本页面。"
    )


# ---------------------------------------------------------------------------
# 区域二：模型评估
# ---------------------------------------------------------------------------

st.subheader("模型评估")

if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="MSE（均方误差）", value=f"{metrics['MSE']:.4f}")
    with col2:
        st.metric(label="RMSE（均方根误差）", value=f"{metrics['RMSE']:.4f}")
    with col3:
        st.metric(label="MAE（平均绝对误差）", value=f"{metrics['MAE']:.4f}")
    st.caption("注：目标变量单位为十万美元，误差越小模型越精确。")
else:
    st.warning("评估指标文件未找到。请先训练模型。")


# ---------------------------------------------------------------------------
# 区域三：特征重要性（扰动法 permutation importance）
# ---------------------------------------------------------------------------

st.subheader("特征重要性（扰动法）")

if has_weights:
    # 加载数据集用于扰动法计算
    housing = fetch_california_housing()
    # 随机采样 200 条数据作为评估样本
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(housing.data), size=200, replace=False)
    X_sample = housing.data[sample_idx].astype(np.float32)

    # 基准预测（原始特征）
    baseline_preds = predict_price(X_sample, ARTIFACTS_DIR)

    # 对每个特征做扰动（打乱该列值），计算预测变化
    importances = []
    progress_bar = st.progress(0, text="正在计算特征重要性...")
    for i, name in enumerate(FEATURE_NAMES):
        X_perturbed = X_sample.copy()
        # 打乱该特征列的顺序
        X_perturbed[:, i] = rng.permutation(X_perturbed[:, i])
        perturbed_preds = predict_price(X_perturbed, ARTIFACTS_DIR)
        # 重要性 = 预测变化的平均绝对值
        importance = float(np.mean(np.abs(perturbed_preds - baseline_preds)))
        importances.append({"特征": name, "重要性": importance})
        progress_bar.progress((i + 1) / len(FEATURE_NAMES))

    progress_bar.empty()

    df_imp = pd.DataFrame(importances).sort_values("重要性")

    # plotly 水平柱状图
    fig = px.bar(
        df_imp,
        x="重要性",
        y="特征",
        orientation="h",
        title="特征重要性（扰动法 Permutation Importance）",
        color="重要性",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(yaxis_title="特征", xaxis_title="重要性（预测变化量）")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "扰动法：对每个特征打乱其值，测量模型预测的变化量。"
        "变化越大说明该特征越重要。"
    )
else:
    st.warning("需要模型权重才能计算特征重要性。请先训练模型。")


# ---------------------------------------------------------------------------
# 区域四：数据集概览
# ---------------------------------------------------------------------------

st.subheader("数据集概览")

# 加载数据集
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=FEATURE_NAMES)
df["房价"] = housing.target  # 目标变量

col4, col5 = st.columns(2)

with col4:
    st.markdown("**目标变量（房价）分布**")
    fig_hist = px.histogram(
        df,
        x="房价",
        nbins=50,
        title="房价分布直方图",
        color_discrete_sequence=["#667eea"],
    )
    fig_hist.update_layout(xaxis_title="房价（十万美元）", yaxis_title="频数")
    st.plotly_chart(fig_hist, use_container_width=True)

with col5:
    st.markdown("**特征相关性热力图**")
    corr = df.corr()
    fig_heat = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        title="特征相关性热力图",
        zmin=-1,
        zmax=1,
    )
    fig_heat.update_layout(height=500)
    st.plotly_chart(fig_heat, use_container_width=True)

# 数据集统计信息
with st.expander("数据集统计信息", expanded=False):
    st.dataframe(df.describe(), use_container_width=True)

st.caption(
    "数据来源：sklearn.datasets.fetch_california_housing，"
    "共 20640 条记录，8 个特征。"
)
