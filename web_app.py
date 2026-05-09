"""
Web界面 - 使用Gradio创建交互式界面
支持图片上传、实时检测和结果展示
"""
import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
import os
from PIL import Image
import torch

class CrackDetectorApp:
    def __init__(self, model_path='runs/segment/crack_seg/weights/best.pt'):
        """
        初始化应用

        Args:
            model_path: 模型路径
        """
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
            self.model_loaded = True
            self.model_name = os.path.basename(model_path)
        else:
            print(f"模型不存在: {model_path}，使用预训练模型")
            self.model = YOLO('yolov8n-seg.pt')
            self.model_loaded = False
            self.model_name = 'yolov8n-seg.pt (预训练)'

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def detect_image(self, image, conf_threshold, iou_threshold):
        """
        检测单张图片

        Args:
            image: 输入图片
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值

        Returns:
            标注后的图片和检测信息
        """
        if image is None:
            return None, "请上传图片"

        # 进行预测
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False
        )

        result = results[0]

        # 获取标注后的图片
        annotated_img = result.plot()
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        # 生成检测信息
        num_cracks = len(result.boxes)

        info = f"### 检测结果\n\n"
        info += f"**模型**: {self.model_name}\n\n"
        info += f"**设备**: {self.device.upper()}\n\n"
        info += f"**检测到的裂缝数量**: {num_cracks}\n\n"

        if num_cracks > 0:
            confidences = result.boxes.conf.cpu().numpy()
            info += f"**平均置信度**: {confidences.mean():.3f}\n\n"
            info += f"**最高置信度**: {confidences.max():.3f}\n\n"
            info += f"**最低置信度**: {confidences.min():.3f}\n\n"

            # 计算裂缝面积
            if result.masks is not None:
                total_area = 0
                for mask in result.masks.data:
                    total_area += mask.sum().item()

                img_area = image.shape[0] * image.shape[1]
                crack_ratio = (total_area / img_area * 100)
                info += f"**裂缝面积占比**: {crack_ratio:.2f}%\n\n"

                # 严重程度评估
                if crack_ratio > 5.0:
                    severity = "🔴 严重"
                elif crack_ratio > 2.0:
                    severity = "🟡 中等"
                else:
                    severity = "🟢 轻微"

                info += f"**严重程度**: {severity}\n\n"

            # 详细信息
            info += "---\n\n"
            info += "### 详细检测信息\n\n"
            for idx, (box, conf) in enumerate(zip(result.boxes.xyxy, confidences)):
                x1, y1, x2, y2 = box.cpu().numpy()
                width = x2 - x1
                height = y2 - y1
                info += f"**裂缝 {idx+1}**:\n"
                info += f"- 位置: ({int(x1)}, {int(y1)}) - ({int(x2)}, {int(y2)})\n"
                info += f"- 尺寸: {int(width)} x {int(height)} 像素\n"
                info += f"- 置信度: {conf:.3f}\n\n"
        else:
            info += "✅ **未检测到裂缝**\n\n"

        return annotated_img, info

    def detect_batch(self, images, conf_threshold, iou_threshold):
        """
        批量检测图片

        Args:
            images: 图片列表
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值

        Returns:
            结果列表
        """
        if not images:
            return []

        results = []
        for image in images:
            annotated_img, info = self.detect_image(image, conf_threshold, iou_threshold)
            results.append((annotated_img, info))

        return results

    def create_interface(self):
        """
        创建Gradio界面

        Returns:
            Gradio界面
        """
        with gr.Blocks(title="墙面裂缝检测系统", theme=gr.themes.Soft()) as demo:
            gr.Markdown(
                """
                # 🏗️ 墙面裂缝检测系统

                基于YOLOv8的智能墙面裂缝检测与分割系统

                上传图片即可自动检测墙面裂缝，系统会标注裂缝位置并提供详细分析报告。
                """
            )

            with gr.Tab("单张图片检测"):
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(
                            label="上传图片",
                            type="numpy",
                            height=400
                        )

                        with gr.Row():
                            conf_slider = gr.Slider(
                                minimum=0.1,
                                maximum=0.9,
                                value=0.25,
                                step=0.05,
                                label="置信度阈值",
                                info="降低阈值可检测更多裂缝，但可能增加误检"
                            )

                            iou_slider = gr.Slider(
                                minimum=0.1,
                                maximum=0.9,
                                value=0.7,
                                step=0.05,
                                label="IOU阈值",
                                info="用于非极大值抑制"
                            )

                        detect_btn = gr.Button("🔍 开始检测", variant="primary", size="lg")

                    with gr.Column():
                        output_image = gr.Image(
                            label="检测结果",
                            type="numpy",
                            height=400
                        )

                        output_info = gr.Markdown(label="检测信息")

                detect_btn.click(
                    fn=self.detect_image,
                    inputs=[input_image, conf_slider, iou_slider],
                    outputs=[output_image, output_info]
                )

                # 示例图片
                gr.Examples(
                    examples=[
                        ["crack-seg/test/images/" + f for f in os.listdir("crack-seg/test/images")[:3]]
                        if os.path.exists("crack-seg/test/images") else []
                    ],
                    inputs=input_image,
                    label="示例图片"
                )

            with gr.Tab("批量检测"):
                gr.Markdown("### 批量上传图片进行检测")

                with gr.Row():
                    batch_input = gr.File(
                        file_count="multiple",
                        label="上传多张图片",
                        file_types=["image"]
                    )

                with gr.Row():
                    batch_conf_slider = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.25,
                        step=0.05,
                        label="置信度阈值"
                    )

                    batch_iou_slider = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.7,
                        step=0.05,
                        label="IOU阈值"
                    )

                batch_detect_btn = gr.Button("🔍 批量检测", variant="primary")

                batch_output = gr.Gallery(
                    label="批量检测结果",
                    columns=3,
                    height="auto"
                )

            with gr.Tab("关于"):
                gr.Markdown(
                    f"""
                    ## 系统信息

                    - **模型**: {self.model_name}
                    - **计算设备**: {self.device.upper()}
                    - **框架**: YOLOv8 + Ultralytics

                    ## 使用说明

                    1. **单张图片检测**
                       - 上传一张墙面图片
                       - 调整置信度阈值（可选）
                       - 点击"开始检测"按钮
                       - 查看检测结果和详细信息

                    2. **批量检测**
                       - 上传多张图片
                       - 设置检测参数
                       - 点击"批量检测"
                       - 查看所有结果

                    ## 严重程度分类

                    - 🟢 **轻微**: 裂缝面积占比 ≤ 2%
                    - 🟡 **中等**: 2% < 裂缝面积占比 ≤ 5%
                    - 🔴 **严重**: 裂缝面积占比 > 5%

                    ## 技术支持

                    如有问题或建议，请查看项目文档或联系技术支持。

                    ---

                    © 2024 墙面裂缝检测系统 | Powered by YOLOv8
                    """
                )

            return demo

def main():
    print("启动墙面裂缝检测Web界面...")
    print("="*60)

    # 检查模型
    model_path = 'runs/segment/crack_seg/weights/best.pt'
    if not os.path.exists(model_path):
        print(f"未找到训练好的模型: {model_path}")
        print("将使用预训练模型进行演示")

    # 创建应用
    app = CrackDetectorApp(model_path)

    # 创建界面
    demo = app.create_interface()

    # 启动服务
    print("\n正在启动Web服务...")
    demo.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,        # 端口
        share=False,             # 不创建公共链接
        inbrowser=True,          # 自动打开浏览器
    )

if __name__ == '__main__':
    main()
