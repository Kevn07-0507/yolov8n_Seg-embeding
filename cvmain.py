"""
基于传统计算机视觉的裂缝检测脚本
使用Canny边缘检测、自适应阈值分割和形态学处理
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import font_config  # 使用之前配置的中文显示模块

class CVCrackDetector:
    def __init__(self):
        pass

    def detect(self, image_path):
        """核心检测流程"""
        # 1. 读取图像
        img = cv2.imread(image_path)
        if img is None:
            return None

        # 2. 转换为灰度图并去噪
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3. 自适应阈值分割（处理亮度不均）
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)

        # 4. Canny边缘检测
        edges = cv2.Canny(blurred, 30, 150)

        # 5. 结合阈值和边缘结果
        combined = cv2.bitwise_and(thresh, edges)

        # 6. 形态学处理（多轮闭运算连接断点）
        kernel_small = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_small)

        
        # 7. 轮廓查找与过滤
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 在原图上绘制检测结果
        result_img = img.copy()
        crack_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # 过滤过小的噪声点
            if area > 50:
                cv2.drawContours(result_img, [cnt], -1, (0, 0, 255), 2)
                crack_count += 1

        return {
            'original': cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            'gray': gray,
            'edges': edges,
            'morph': morph,
            'result': cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
            'count': crack_count
        }

    def visualize(self, image_path):
        """可视化结果"""
        results = self.detect(image_path)
        if results is None:
            print(f"无法读取图片: {image_path}")
            return

        plt.figure(figsize=(15, 10))

        titles = ['原始图像', '灰度图', '边缘检测 (Canny)', '形态学处理', '检测结果']
        images = [results['original'], results['gray'], results['edges'],
                  results['morph'], results['result']]

        for i in range(5):
            plt.subplot(2, 3, i+1)
            plt.imshow(images[i], cmap='gray' if len(images[i].shape) == 2 else None)
            plt.title(titles[i])
            plt.axis('off')

        plt.suptitle(f"传统CV检测结果 - 发现裂缝区域: {results['count']}")
        plt.tight_layout()
        plt.show()

def main():
    print("\n" + "="*50)
    print("传统计算机视觉裂缝检测工具")
    print("="*50)

    test_img = input("请输入图片路径 (直接回车自动选择): ").strip()

    if not test_img:
        default_dir = "crack-seg/test/images"
        if os.path.exists(default_dir):
            files = [f for f in os.listdir(default_dir) if f.endswith(('.jpg', '.png'))]
            if files:
                test_img = os.path.join(default_dir, files[0])
                print(f"自动选择测试图片: {test_img}")

    if test_img and os.path.exists(test_img):
        detector = CVCrackDetector()
        detector.visualize(test_img)
    else:
        print("未找到有效图片路径。")

if __name__ == "__main__":
    main()
