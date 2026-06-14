"""
后端 API 服务 —— 跑在你的 PC 上
提供 /detect 接口，接收图片并返回裂缝检测结果

启动方式: uvicorn backend_api:app --host 0.0.0.0 --port 8000
启动后在浏览器打开 http://localhost:8000/docs 可以上传图片测试
"""
import os, sys, io, base64, time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ═════════════════ 配置 ═════════════════
MODEL_PATH = 'runs/segment/crack_seg4/weights/best.pt'  # v8n 100ep 最佳模型
CONF_DEFAULT = 0.25
IOU_DEFAULT = 0.7

# ═════════════════ 初始化 ═════════════════
print("=" * 60)
print("  墙面裂缝检测 - 后端 API 服务")
print("=" * 60)

# 检查模型
if not os.path.exists(MODEL_PATH):
    print(f"[WARN] 未找到模型: {MODEL_PATH}")
    # 搜索其他可能的模型
    alt_models = list(Path('runs/segment').rglob('**/best.pt'))
    if alt_models:
        MODEL_PATH = str(alt_models[0])
        print(f"[INFO] 使用替代模型: {MODEL_PATH}")
    else:
        print("[ERROR] 没有找到任何训练好的模型，将使用预训练权重")
        MODEL_PATH = 'yolov8n-seg.pt'

print(f"[INFO] 加载模型: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] 计算设备: {device.upper()}")

# ═════════════════ FastAPI 应用 ═════════════════
app = FastAPI(
    title="墙面裂缝检测 API",
    description="基于 YOLOv8 实例分割的墙面裂缝检测后端服务",
    version="1.0.0"
)


@app.get("/", response_class=HTMLResponse)
def root():
    """根路径：返回简单的 HTML 页面用于手动测试"""
    return """
    <!DOCTYPE html>
    <html lang="zh">
    <head>
        <meta charset="UTF-8">
        <title>墙面裂缝检测 API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            h1 { color: #333; }
            .card { border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin: 20px 0; }
            input, button { padding: 10px; margin: 5px 0; }
            button { background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }
            #result { margin-top: 20px; white-space: pre-wrap; background: #f5f5f5; padding: 15px; border-radius: 4px; }
            img { max-width: 100%; margin-top: 10px; border: 2px solid #ddd; }
        </style>
    </head>
    <body>
        <h1>🧱 墙面裂缝检测 API</h1>
        <div class="card">
            <h3>📤 上传图片测试</h3>
            <input type="file" id="fileInput" accept="image/*">
            <button onclick="detect()">🔍 开始检测</button>
            <div id="result"></div>
        </div>
        <div class="card">
            <h3>📖 API 文档</h3>
            <p>访问 <a href="/docs">/docs</a> 查看完整 Swagger API 文档</p>
            <p>访问 <a href="/redoc">/redoc</a> 查看 ReDoc 文档</p>
        </div>
        <script>
            async function detect() {
                const file = document.getElementById('fileInput').files[0];
                if (!file) { alert('请先选择图片'); return; }
                const formData = new FormData();
                formData.append('file', file);
                const resultDiv = document.getElementById('result');
                resultDiv.textContent = '正在检测...';
                try {
                    const resp = await fetch('/detect', { method: 'POST', body: formData });
                    const data = await resp.json();
                    let html = '<h4>检测结果：</h4>';
                    html += '<p>裂缝数量: <b>' + data.crack_count + '</b></p>';
                    html += '<p>严重程度: <b>' + data.severity + '</b></p>';
                    html += '<p>裂缝面积占比: <b>' + data.crack_ratio + '%</b></p>';
                    html += '<p>平均置信度: <b>' + data.avg_confidence + '</b></p>';
                    html += '<p>推理耗时: <b>' + data.inference_time + 'ms</b></p>';
                    if (data.crack_count > 0) {
                        html += '<p>各裂缝置信度: ' + JSON.stringify(data.confidences) + '</p>';
                    }
                    if (data.annotated_image) {
                        html += '<img src="data:image/jpeg;base64,' + data.annotated_image + '" alt="检测结果">';
                    }
                    resultDiv.innerHTML = html;
                } catch (err) {
                    resultDiv.textContent = '请求失败: ' + err.message;
                }
            }
        </script>
    </body>
    </html>
    """


@app.post("/detect")
async def detect(
    file: UploadFile = File(..., description="墙面图片文件（支持 jpg/png/bmp）"),
    conf: float = Query(CONF_DEFAULT, ge=0.1, le=0.9, description="置信度阈值（0.1-0.9）"),
    iou: float = Query(IOU_DEFAULT, ge=0.1, le=0.9, description="NMS IOU 阈值（0.1-0.9）"),
    return_image: bool = Query(True, description="是否返回标注后的图片（base64）"),
):
    """
    裂缝检测接口

    - **file**: 上传的墙面图片文件
    - **conf**: 置信度阈值，低于此值的检测结果会被过滤（默认 0.25）
    - **iou**: NMS 的 IOU 阈值，用于去重（默认 0.7）
    - **return_image**: 是否在响应中返回标注后的 base64 图片

    返回 JSON 格式的检测结果，包含裂缝数量、置信度、严重程度等。
    """
    # 1. 验证文件类型
    if not file.content_type or not file.content_type.startswith('image/'):
        return JSONResponse(
            status_code=400,
            content={"error": "请上传图片文件", "detail": f"不支持的类型: {file.content_type}"}
        )

    # 2. 读取图片
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(
            status_code=400,
            content={"error": "无法解析图片文件"}
        )

    original_h, original_w = img.shape[:2]

    # 3. 模型推理
    t_start = time.time()
    results = model.predict(
        source=img,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
    )
    t_end = time.time()
    inference_time = round((t_end - t_start) * 1000, 1)  # 毫秒

    r = results[0]

    # 4. 提取检测结果
    num_cracks = len(r.boxes) if r.boxes is not None else 0
    confidences = []
    boxes_list = []
    severity = "无裂缝"
    crack_ratio = 0.0
    avg_confidence = 0.0

    if num_cracks > 0:
        # 置信度
        conf_arr = r.boxes.conf.cpu().numpy()
        confidences = [round(float(c), 4) for c in conf_arr]
        avg_confidence = round(float(conf_arr.mean()), 4)

        # 边界框
        boxes_arr = r.boxes.xyxy.cpu().numpy()
        boxes_list = [[round(float(v), 1) for v in box] for box in boxes_arr]

        # 裂缝面积占比（基于分割掩码）
        if r.masks is not None:
            total_mask_area = 0
            for mask in r.masks.data:
                total_mask_area += mask.sum().item()
            total_img_area = original_h * original_w
            crack_ratio = round((total_mask_area / total_img_area) * 100, 2)

        # 严重程度分级
        if crack_ratio > 5.0:
            severity = "🔴 严重"
        elif crack_ratio > 2.0:
            severity = "🟡 中等"
        elif crack_ratio > 0:
            severity = "🟢 轻微"
        else:
            severity = "⚪ 极轻微（无掩码）"

    # 5. 生成标注图片
    img_base64 = None
    if return_image:
        annotated = r.plot()
        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_base64 = base64.b64encode(buffer).decode('utf-8')

    # 6. 组装响应
    response = {
        "success": True,
        "filename": file.filename,
        "image_size": {"width": original_w, "height": original_h},
        "crack_count": num_cracks,
        "severity": severity,
        "crack_ratio": crack_ratio,
        "avg_confidence": avg_confidence,
        "confidences": confidences,
        "boxes": boxes_list,
        "inference_time_ms": inference_time,
        "device": device.upper(),
        "model": os.path.basename(MODEL_PATH),
        "parameters": {"conf": conf, "iou": iou},
    }

    if img_base64:
        response["annotated_image"] = img_base64

    return response


@app.get("/health")
def health_check():
    """健康检查接口"""
    return {
        "status": "ok",
        "model_loaded": os.path.exists(MODEL_PATH),
        "model_path": MODEL_PATH,
        "device": device.upper(),
    }


# ═════════════════ 启动入口 ═════════════════
if __name__ == '__main__':
    import uvicorn
    print("\n" + "=" * 60)
    print("  启动 API 服务...")
    print("  本地访问: http://localhost:8000")
    print("  接口文档: http://localhost:8000/docs")
    print("  树莓派需访问: http://<你PC的IP地址>:8000")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)