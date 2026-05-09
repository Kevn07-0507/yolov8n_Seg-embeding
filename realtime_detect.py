"""
实时裂缝检测脚本
使用摄像头或视频流进行实时裂缝检测
"""
from ultralytics import YOLO
import cv2
import time
import numpy as np

class RealtimeCrackDetector:
    def __init__(self, model_path='runs/segment/crack_seg/weights/best.pt', conf_threshold=0.25):
        """
        初始化实时检测器

        Args:
            model_path: 模型路径
            conf_threshold: 置信度阈值
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

    def process_frame(self, frame):
        """
        处理单帧图像

        Args:
            frame: 输入帧

        Returns:
            处理后的帧
        """
        # 进行预测
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            verbose=False
        )

        # 获取标注后的图像
        annotated_frame = results[0].plot()

        # 计算FPS
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            self.fps = self.frame_count / elapsed_time

        # 在图像上显示FPS和检测信息
        cv2.putText(annotated_frame, f'FPS: {self.fps:.1f}',
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示检测到的裂缝数量
        num_cracks = len(results[0].boxes)
        cv2.putText(annotated_frame, f'Cracks: {num_cracks}',
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return annotated_frame, results[0]

    def detect_from_camera(self, camera_id=0):
        """
        从摄像头进行实时检测

        Args:
            camera_id: 摄像头ID
        """
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"无法打开摄像头 {camera_id}")
            return

        print("按 'q' 退出, 按 's' 保存当前帧")

        while True:
            ret, frame = cap.read()

            if not ret:
                print("无法读取摄像头画面")
                break

            # 处理帧
            annotated_frame, results = self.process_frame(frame)

            # 显示结果
            cv2.imshow('实时裂缝检测', annotated_frame)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 保存当前帧
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f'capture_{timestamp}.jpg'
                cv2.imwrite(filename, annotated_frame)
                print(f"已保存: {filename}")

        cap.release()
        cv2.destroyAllWindows()

    def detect_from_video(self, video_path, output_path=None):
        """
        从视频文件进行检测

        Args:
            video_path: 视频文件路径
            output_path: 输出视频路径（可选）
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            return

        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"视频信息: {width}x{height} @ {fps}fps, 总帧数: {total_frames}")

        # 创建视频写入器
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print("处理中... 按 'q' 退出")

        frame_idx = 0
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_idx += 1

            # 处理帧
            annotated_frame, results = self.process_frame(frame)

            # 显示进度
            progress = (frame_idx / total_frames) * 100
            print(f"\r进度: {progress:.1f}% ({frame_idx}/{total_frames})", end='')

            # 写入输出视频
            if writer:
                writer.write(annotated_frame)

            # 显示结果
            cv2.imshow('视频裂缝检测', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print("\n处理完成!")

        cap.release()
        if writer:
            writer.release()
            print(f"输出视频已保存: {output_path}")
        cv2.destroyAllWindows()

def main():
    print("实时裂缝检测工具")
    print("="*60)

    # 检查模型
    model_path = 'runs/segment/crack_seg/weights/best.pt'
    if not os.path.exists(model_path):
        print(f"未找到训练好的模型: {model_path}")
        print("使用预训练模型...")
        model_path = 'yolov8n-seg.pt'

    # 创建检测器
    detector = RealtimeCrackDetector(model_path=model_path, conf_threshold=0.25)

    print("\n请选择检测模式:")
    print("1. 摄像头实时检测")
    print("2. 视频文件检测")

    choice = input("\n请输入选项 (1-2): ").strip()

    if choice == '1':
        camera_id = input("请输入摄像头ID (默认: 0): ").strip()
        camera_id = int(camera_id) if camera_id else 0
        detector.detect_from_camera(camera_id)

    elif choice == '2':
        video_path = input("请输入视频文件路径: ").strip()
        if os.path.exists(video_path):
            save_output = input("是否保存输出视频? (y/n): ").strip().lower()
            output_path = None
            if save_output == 'y':
                output_path = input("请输入输出视频路径 (默认: output.mp4): ").strip()
                output_path = output_path if output_path else 'output.mp4'
            detector.detect_from_video(video_path, output_path)
        else:
            print(f"视频文件不存在: {video_path}")
    else:
        print("无效的选项!")

if __name__ == '__main__':
    import os
    main()
