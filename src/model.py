"""MindSpore 模型定义

MLPAttention：带自注意力的多层感知机

结构：
- 编码器：Dense(8→128) → ReLU → Dropout → Dense(128→128) → ReLU
- 注意力：对编码特征做 Q/K/V 注意力（单token简化版，Sigmoid 门控）
- 融合：拼接 [原始编码, 注意力输出] → Dense(256→128) → ReLU → Dropout → Dense(128→1)
- 输出 squeeze(-1)

注意：MindSpore 用 nn.Dense（非 nn.Linear）、nn.Cell（非 nn.Module）、construct（非 forward）
"""

# 检测 MindSpore 是否可用
try:
    import mindspore as ms  # noqa: F401
    from mindspore import nn, ops
    _HAS_MINDSPORE = True
except ImportError:
    _HAS_MINDSPORE = False


def build_model(input_dim=8, hidden_dim=128, dropout=0.2):
    """构建模型

    当 MindSpore 未安装时 raise RuntimeError
    """
    if not _HAS_MINDSPORE:
        raise RuntimeError(
            "MindSpore 未安装，无法构建模型。请先安装 mindspore。"
        )
    return MLPAttention(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


if _HAS_MINDSPORE:

    class MLPAttention(nn.Cell):
        """带自注意力的多层感知机

        编码器提取特征 → 注意力模块做门控增强 → 融合层输出预测值
        """

        def __init__(self, input_dim=8, hidden_dim=128, dropout=0.2):
            super().__init__()
            # ---- 编码器 ----
            # Dense(8→128) → ReLU → Dropout → Dense(128→128) → ReLU
            self.enc_fc1 = nn.Dense(input_dim, hidden_dim)
            self.enc_relu1 = nn.ReLU()
            self.enc_dropout1 = nn.Dropout(p=dropout)
            self.enc_fc2 = nn.Dense(hidden_dim, hidden_dim)
            self.enc_relu2 = nn.ReLU()

            # ---- 注意力 Q/K/V 投影（单token简化版，Sigmoid 门控）----
            self.q_proj = nn.Dense(hidden_dim, hidden_dim)
            self.k_proj = nn.Dense(hidden_dim, hidden_dim)
            self.v_proj = nn.Dense(hidden_dim, hidden_dim)
            self.sigmoid = nn.Sigmoid()

            # ---- 融合层 ----
            # 拼接 [原始编码, 注意力输出] → Dense(256→128) → ReLU → Dropout → Dense(128→1)
            self.fus_fc1 = nn.Dense(hidden_dim * 2, hidden_dim)
            self.fus_relu = nn.ReLU()
            self.fus_dropout = nn.Dropout(p=dropout)
            self.fus_fc2 = nn.Dense(hidden_dim, 1)

        def construct(self, x):
            # 编码器前向
            encoded = self.enc_fc1(x)
            encoded = self.enc_relu1(encoded)
            encoded = self.enc_dropout1(encoded)
            encoded = self.enc_fc2(encoded)
            encoded = self.enc_relu2(encoded)

            # 注意力（单token简化版：Sigmoid 门控）
            q = self.q_proj(encoded)
            k = self.k_proj(encoded)
            v = self.v_proj(encoded)
            attn = self.sigmoid(q * k) * v

            # 融合前向
            combined = ops.concat((encoded, attn), axis=-1)
            out = self.fus_fc1(combined)
            out = self.fus_relu(out)
            out = self.fus_dropout(out)
            out = self.fus_fc2(out)
            # 输出 squeeze(-1)，shape (batch,)
            return out.squeeze(-1)
