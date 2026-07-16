"""轻量级 Web 推理服务 - 适合香橙派等嵌入式设备

提供简单的 HTML 页面，通过 Flask 提供推理 API 和可视化界面。
依赖：flask, numpy（不需要 MindSpore、Streamlit）

运行方式：
    python3 web_server.py --port 8080
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import predict_price
from src.data import FEATURE_NAMES, FEATURE_RANGES

try:
    from flask import Flask, render_template_string, request, jsonify
except ImportError:
    print("请先安装 Flask: pip3 install flask")
    sys.exit(1)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# HTML 页面模板
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>房价预测系统</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
               background: #f0f2f5; padding: 20px; }
        .container { max-width: 600px; margin: 0 auto; }
        h1 { text-align: center; color: #333; margin-bottom: 20px; font-size: 24px; }
        .card { background: white; border-radius: 12px; padding: 20px; margin-bottom: 16px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .card h2 { font-size: 16px; color: #666; margin-bottom: 12px; }
        .slider-group { margin-bottom: 12px; }
        .slider-group label { display: flex; justify-content: space-between;
                              font-size: 14px; color: #444; margin-bottom: 4px; }
        .slider-group label span { color: #1a73e8; font-weight: bold; }
        .slider-group input[type=range] { width: 100%; height: 6px;
                                          -webkit-appearance: none; background: #e0e0e0;
                                          border-radius: 3px; outline: none; }
        .slider-group input[type=range]::-webkit-slider-thumb {
            -webkit-appearance: none; width: 18px; height: 18px;
            background: #1a73e8; border-radius: 50%; cursor: pointer; }
        .btn { width: 100%; padding: 14px; background: linear-gradient(135deg, #667eea, #764ba2);
               color: white; border: none; border-radius: 8px; font-size: 16px;
               cursor: pointer; margin-top: 8px; }
        .btn:hover { opacity: 0.9; }
        .result { text-align: center; padding: 24px; }
        .result .price { font-size: 36px; font-weight: bold; color: #1a73e8; }
        .result .unit { font-size: 14px; color: #999; margin-top: 4px; }
        .metrics { display: flex; justify-content: space-around; text-align: center; }
        .metrics .item .value { font-size: 20px; font-weight: bold; color: #333; }
        .metrics .item .label { font-size: 12px; color: #999; margin-top: 2px; }
        .loading { text-align: center; color: #999; font-size: 14px; display: none; }
    </style>
</head>
<body>
<div class="container">
    <h1>🏠 加州房价预测系统</h1>

    <div class="card">
        <h2>输入特征</h2>
        {% for name in features %}
        <div class="slider-group">
            <label>{{ name }} <span id="val_{{ loop.index0 }}">{{ defaults[loop.index0] }}</span></label>
            <input type="range" id="slider_{{ loop.index0 }}"
                   min="{{ ranges[loop.index0][0] }}" max="{{ ranges[loop.index0][1] }}"
                   value="{{ defaults[loop.index0] }}" step="0.1"
                   oninput="document.getElementById('val_{{ loop.index0 }}').textContent = parseFloat(this.value).toFixed(1)">
        </div>
        {% endfor %}
        <button class="btn" onclick="predict()"> 预测房价</button>
        <p class="loading" id="loading">计算中...</p>
    </div>

    <div class="card result" id="resultCard" style="display:none;">
        <div class="price" id="priceText"></div>
        <div class="unit" id="priceUnit"></div>
    </div>

    {% if metrics %}
    <div class="card">
        <h2>模型评估</h2>
        <div class="metrics">
            <div class="item"><div class="value">{{ "%.4f"|format(metrics.MSE) }}</div><div class="label">MSE</div></div>
            <div class="item"><div class="value">{{ "%.4f"|format(metrics.RMSE) }}</div><div class="label">RMSE</div></div>
            <div class="item"><div class="value">{{ "%.4f"|format(metrics.MAE) }}</div><div class="label">MAE</div></div>
        </div>
    </div>
    {% endif %}
</div>

<script>
function predict() {
    document.getElementById('loading').style.display = 'block';
    document.getElementById('resultCard').style.display = 'none';

    var features = [];
    for (var i = 0; i < {{ features|length }}; i++) {
        features.push(parseFloat(document.getElementById('slider_' + i).value));
    }

    fetch('/api/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({features: features})
    })
    .then(r => r.json())
    .then(data => {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('resultCard').style.display = 'block';
        document.getElementById('priceText').textContent = (data.price_wan).toFixed(2) + ' 万美元';
        document.getElementById('priceUnit').textContent = '预测房价 = ' + data.price.toFixed(4) + ' 十万美元';
    })
    .catch(err => {
        document.getElementById('loading').style.display = 'none';
        alert('预测失败: ' + err);
    });
}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """主页：特征滑块 + 预测按钮"""
    ranges = [FEATURE_RANGES[name] for name in FEATURE_NAMES]
    defaults = [r[2] for r in ranges]
    ranges_formatted = [(r[0], r[1]) for r in ranges]

    # 加载评估指标（如果有）
    metrics = None
    metrics_path = os.path.join("artifacts", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

    return render_template_string(
        HTML_TEMPLATE,
        features=FEATURE_NAMES,
        ranges=ranges_formatted,
        defaults=defaults,
        metrics=metrics,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """推理 API"""
    data = request.json
    features = np.array(data["features"], dtype=np.float32)
    price = predict_price(features, "artifacts")
    return jsonify({
        "price": float(price),
        "price_wan": round(float(price) * 10, 2),
    })


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="轻量级房价预测 Web 服务")
    parser.add_argument("--port", type=int, default=8080, help="服务端口")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="绑定地址")
    args = parser.parse_args()

    print(f"启动 Web 服务: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
