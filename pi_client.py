"""
树莓派客户端 — 拍照 → 发送到 PC 后端 → 显示结果

硬件要求: 树莓派 + Camera Module / USB 摄像头
软件依赖: pip install requests

用法:
  1. 修改下面的 PC_IP 为你的 PC 实际 IP 地址
  2. PC 上先启动: uvicorn backend_api:app --host 0.0.0.0 --port 8000
  3. 树莓派运行: python pi_client.py
"""

import time
import base64
import json
import os
from io import BytesIO

import requests
import cv2
import numpy as np

# ═════════════════ 配置 — 修改这里的 IP ═════════════════
PC_IP = '192.168.1.100'        # PC 的局域网 IP 地址
API_URL = f'http://{PC_IP}:8000/detect'
HEALTH_URL = f'http://{PC_IP}:8000/health'

CAMERA_ID = 0                   # 摄像头编号（内置=0, USB=1）
SAVE_DIR = 'captures'           # 拍照保存目录
AUTO_INTERVAL = 5               # 自动模式拍照间隔（秒），0=手动模式
# ═══════════════════════════════════════════════════════

os.makedirs(SAVE_DIR, exist_ok=True)


def check_connection():
    """检查 PC 后端是否在线"""
    try:
        resp = requests.get(HEALTH_URL, timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            print(f"PC 后端在线: {data['model_path']} | Device: {data['device']}")
            return True
    except requests.exceptions.ConnectionError:
        print(f"无法连接到 PC 后端 ({API_URL})")
        print("请确认:")
        print("  1. PC 上已启动: uvicorn backend_api:app --host 0.0.0.0 --port 8000")
        print(f"  2. PC IP 地址正确 (当前: {PC_IP})")
        print("  3. 防火墙已放行 8000 端口")
        print("  4. 树莓派能 ping 通 PC")
    except Exception as e:
        print(f"健康检查失败: {e}")
    return False


def send_image(filepath):
    """发送图片到 PC 后端进行检测"""
    with open(filepath, 'rb') as f:
        resp = requests.post(API_URL, files={'file': f}, timeout=30)

    if resp.status_code != 200:
        print(f"API 返回错误: {resp.status_code} {resp.text}")
        return None

    return resp.json()


def save_annotated(data, save_path):
    """解码 base64 标注图并保存"""
    if 'annotated_image' not in data:
        return None
    img_bytes = base64.b64decode(data['annotated_image'])
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is not None:
        cv2.imwrite(save_path, img)
    return img


def print_result(data):
    """打印检测结果"""
    print("\n" + "=" * 50)
    print("  裂缝检测结果")
    print("=" * 50)
    print(f"  图片: {data['filename']}")
    print(f"  尺寸: {data['image_size']['width']} x {data['image_size']['height']}")
    print(f"  裂缝数量: {data['crack_count']}")
    print(f"  严重程度: {data['severity']}")
    print(f"  裂缝面积占比: {data['crack_ratio']}%")
    print(f"  平均置信度: {data['avg_confidence']}")
    print(f"  推理耗时: {data['inference_time_ms']}ms")
    if data['confidences']:
        print(f"  各裂缝置信度: {[f'{c:.3f}' for c in data['confidences']]}")
    print("=" * 50)


def manual_mode(cap):
    """手动模式: 按 Enter 拍照并检测"""
    print("\n手动模式 — 按 Enter 拍照, 输入 q 退出")
    while True:
        cmd = input("\n>>> ").strip()
        if cmd.lower() == 'q':
            break

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f'{SAVE_DIR}/crack_{timestamp}.jpg'
        result_filename = f'{SAVE_DIR}/result_{timestamp}.jpg'

        # 拍照
        ret, frame = cap.read()
        if not ret:
            print("摄像头读取失败")
            continue
        cv2.imwrite(filename, frame)
        print(f"已拍摄: {filename}")

        # 发送检测
        data = send_image(filename)
        if data is None:
            continue

        print_result(data)
        save_annotated(data, result_filename)
        print(f"标注图已保存: {result_filename}")


def auto_mode(cap):
    """自动模式: 每 N 秒拍照并检测"""
    print(f"自动模式 — 每 {AUTO_INTERVAL} 秒拍照一次, Ctrl+C 退出")
    try:
        while True:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f'{SAVE_DIR}/crack_{timestamp}.jpg'
            result_filename = f'{SAVE_DIR}/result_{timestamp}.jpg'

            ret, frame = cap.read()
            if not ret:
                print("摄像头读取失败")
                time.sleep(AUTO_INTERVAL)
                continue

            cv2.imwrite(filename, frame)
            print(f"\n[{time.strftime('%H:%M:%S')}] 已拍摄: {filename}")

            try:
                data = send_image(filename)
                if data:
                    print_result(data)
                    save_annotated(data, result_filename)
            except requests.exceptions.Timeout:
                print("请求超时")
            except Exception as e:
                print(f"检测失败: {e}")

            time.sleep(AUTO_INTERVAL)
    except KeyboardInterrupt:
        print("\n停止自动模式")


def main():
    print("=" * 50)
    print("  墙面裂缝检测 — 树莓派客户端")
    print(f"  后端地址: {API_URL}")
    print("=" * 50)

    # 1. 检查连接
    if not check_connection():
        return

    # 2. 打开摄像头
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"无法打开摄像头 (ID={CAMERA_ID})")
        print("请检查摄像头连接, 或修改 CAMERA_ID")
        return

    print(f"摄像头已就绪 (ID={CAMERA_ID}), 分辨率: {int(cap.get(3))}x{int(cap.get(4))}")

    # 3. 选择模式
    if AUTO_INTERVAL > 0:
        auto_mode(cap)
    else:
        manual_mode(cap)

    cap.release()
    print("\n客户端已关闭")


if __name__ == '__main__':
    main()
