"""
批量处理脚本
批量处理图片并生成检测报告
"""
from ultralytics import YOLO
import cv2
import os
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm

class BatchProcessor:
    def __init__(self, model_path, conf_threshold=0.25):
        """
        初始化批量处理器

        Args:
            model_path: 模型路径
            conf_threshold: 置信度阈值
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.results_data = []

    def process_image(self, image_path):
        """
        处理单张图片

        Args:
            image_path: 图片路径

        Returns:
            检测结果字典
        """
        # 进行预测
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            verbose=False
        )

        result = results[0]

        # 提取检测信息
        num_cracks = len(result.boxes)
        confidences = result.boxes.conf.cpu().numpy().tolist() if num_cracks > 0 else []

        # 计算裂缝总面积（像素）
        total_area = 0
        if result.masks is not None:
            for mask in result.masks.data:
                total_area += mask.sum().item()

        # 获取图片信息
        img = cv2.imread(image_path)
        img_height, img_width = img.shape[:2]
        img_area = img_height * img_width

        # 计算裂缝面积占比
        crack_ratio = (total_area / img_area * 100) if img_area > 0 else 0

        result_dict = {
            'image_name': os.path.basename(image_path),
            'image_path': image_path,
            'image_width': img_width,
            'image_height': img_height,
            'num_cracks': num_cracks,
            'confidences': confidences,
            'avg_confidence': sum(confidences) / len(confidences) if confidences else 0,
            'max_confidence': max(confidences) if confidences else 0,
            'min_confidence': min(confidences) if confidences else 0,
            'total_crack_area_pixels': int(total_area),
            'crack_area_ratio': round(crack_ratio, 2),
            'has_crack': num_cracks > 0,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        return result_dict, result

    def process_directory(self, input_dir, output_dir='batch_results', save_images=True):
        """
        批量处理目录中的所有图片

        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            save_images: 是否保存标注后的图片
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        images_output_dir = os.path.join(output_dir, 'images')
        if save_images:
            os.makedirs(images_output_dir, exist_ok=True)

        # 获取所有图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(input_dir).rglob(f'*{ext}'))
            image_files.extend(Path(input_dir).rglob(f'*{ext.upper()}'))
        image_files = list(set(image_files))  # Windows大小写不敏感去重

        if not image_files:
            print(f"在 {input_dir} 中未找到图片文件")
            return

        print(f"找到 {len(image_files)} 张图片")
        print("开始批量处理...")

        # 处理每张图片
        self.results_data = []
        for image_path in tqdm(image_files, desc="处理进度"):
            try:
                result_dict, result = self.process_image(str(image_path))
                self.results_data.append(result_dict)

                # 保存标注后的图片
                if save_images:
                    annotated_img = result.plot()
                    output_path = os.path.join(images_output_dir, result_dict['image_name'])
                    cv2.imwrite(output_path, annotated_img)

            except Exception as e:
                print(f"\n处理 {image_path} 时出错: {e}")
                continue

        print("\n批量处理完成!")

        # 生成报告
        self.generate_report(output_dir)

    def generate_report(self, output_dir):
        """
        生成检测报告

        Args:
            output_dir: 输出目录
        """
        if not self.results_data:
            print("没有数据可生成报告")
            return

        print("\n生成检测报告...")

        # 1. 保存详细结果为JSON
        json_path = os.path.join(output_dir, 'detection_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results_data, f, indent=2, ensure_ascii=False)
        print(f"✓ JSON报告已保存: {json_path}")

        # 2. 保存为CSV
        df = pd.DataFrame(self.results_data)
        csv_path = os.path.join(output_dir, 'detection_results.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✓ CSV报告已保存: {csv_path}")

        # 3. 生成统计摘要
        summary = self.generate_summary(df)
        summary_path = os.path.join(output_dir, 'summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"✓ 统计摘要已保存: {summary_path}")

        # 4. 生成HTML报告
        html_report = self.generate_html_report(df)
        html_path = os.path.join(output_dir, 'report.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        print(f"✓ HTML报告已保存: {html_path}")

        print(f"\n所有报告已保存到: {output_dir}")

    def generate_summary(self, df):
        """
        生成统计摘要

        Args:
            df: 结果DataFrame

        Returns:
            摘要文本
        """
        total_images = len(df)
        images_with_cracks = df['has_crack'].sum()
        images_without_cracks = total_images - images_with_cracks

        summary = f"""
墙面裂缝检测批量处理报告
{'='*60}

处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

【总体统计】
- 处理图片总数: {total_images}
- 检测到裂缝的图片: {images_with_cracks} ({images_with_cracks/total_images*100:.1f}%)
- 未检测到裂缝的图片: {images_without_cracks} ({images_without_cracks/total_images*100:.1f}%)

【裂缝统计】
- 裂缝总数: {df['num_cracks'].sum()}
- 平均每张图片裂缝数: {df['num_cracks'].mean():.2f}
- 最多裂缝数: {df['num_cracks'].max()}
- 最少裂缝数: {df['num_cracks'].min()}

【置信度统计】
- 平均置信度: {df['avg_confidence'].mean():.3f}
- 最高置信度: {df['max_confidence'].max():.3f}
- 最低置信度: {df[df['min_confidence'] > 0]['min_confidence'].min():.3f}

【裂缝面积统计】
- 平均裂缝面积占比: {df['crack_area_ratio'].mean():.2f}%
- 最大裂缝面积占比: {df['crack_area_ratio'].max():.2f}%
- 最小裂缝面积占比: {df[df['crack_area_ratio'] > 0]['crack_area_ratio'].min():.2f}%

【严重程度分类】
"""

        # 按裂缝面积占比分类
        severe = df[df['crack_area_ratio'] > 5.0]
        moderate = df[(df['crack_area_ratio'] > 2.0) & (df['crack_area_ratio'] <= 5.0)]
        mild = df[(df['crack_area_ratio'] > 0) & (df['crack_area_ratio'] <= 2.0)]

        summary += f"- 严重 (面积占比 > 5%): {len(severe)} 张\n"
        summary += f"- 中等 (2% < 面积占比 <= 5%): {len(moderate)} 张\n"
        summary += f"- 轻微 (0% < 面积占比 <= 2%): {len(mild)} 张\n"

        if len(severe) > 0:
            summary += f"\n【需要重点关注的图片】\n"
            for idx, row in severe.nlargest(10, 'crack_area_ratio').iterrows():
                summary += f"- {row['image_name']}: 裂缝面积占比 {row['crack_area_ratio']:.2f}%, 裂缝数量 {row['num_cracks']}\n"

        summary += f"\n{'='*60}\n"

        return summary

    def generate_html_report(self, df):
        """
        生成HTML报告

        Args:
            df: 结果DataFrame

        Returns:
            HTML文本
        """
        total_images = len(df)
        images_with_cracks = df['has_crack'].sum()

        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>墙面裂缝检测报告</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            opacity: 0.9;
        }}
        .stat-card .value {{
            font-size: 32px;
            font-weight: bold;
            margin: 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .severity-severe {{
            color: #d32f2f;
            font-weight: bold;
        }}
        .severity-moderate {{
            color: #f57c00;
            font-weight: bold;
        }}
        .severity-mild {{
            color: #388e3c;
        }}
        .timestamp {{
            color: #999;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>墙面裂缝检测批量处理报告</h1>
        <p class="timestamp">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>总体统计</h2>
        <div class="stats">
            <div class="stat-card">
                <h3>处理图片总数</h3>
                <p class="value">{total_images}</p>
            </div>
            <div class="stat-card">
                <h3>检测到裂缝</h3>
                <p class="value">{images_with_cracks}</p>
            </div>
            <div class="stat-card">
                <h3>裂缝总数</h3>
                <p class="value">{df['num_cracks'].sum()}</p>
            </div>
            <div class="stat-card">
                <h3>平均置信度</h3>
                <p class="value">{df['avg_confidence'].mean():.2f}</p>
            </div>
        </div>

        <h2>详细检测结果</h2>
        <table>
            <thead>
                <tr>
                    <th>图片名称</th>
                    <th>裂缝数量</th>
                    <th>平均置信度</th>
                    <th>裂缝面积占比</th>
                    <th>严重程度</th>
                </tr>
            </thead>
            <tbody>
"""

        for idx, row in df.iterrows():
            severity = '严重' if row['crack_area_ratio'] > 5.0 else ('中等' if row['crack_area_ratio'] > 2.0 else '轻微')
            severity_class = 'severity-severe' if row['crack_area_ratio'] > 5.0 else ('severity-moderate' if row['crack_area_ratio'] > 2.0 else 'severity-mild')

            html += f"""
                <tr>
                    <td>{row['image_name']}</td>
                    <td>{row['num_cracks']}</td>
                    <td>{row['avg_confidence']:.3f}</td>
                    <td>{row['crack_area_ratio']:.2f}%</td>
                    <td class="{severity_class}">{severity if row['has_crack'] else '无裂缝'}</td>
                </tr>
"""

        html += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""

        return html

def main():
    print("批量处理工具")
    print("="*60)

    # 检查模型
    model_path = 'runs/segment/crack_seg/weights/best.pt'
    if not os.path.exists(model_path):
        print(f"未找到训练好的模型: {model_path}")
        model_path = input("请输入模型路径: ").strip()
        if not os.path.exists(model_path):
            print("模型不存在!")
            return

    # 获取输入目录
    input_dir = input("请输入要处理的图片目录: ").strip()
    if not os.path.exists(input_dir):
        print(f"目录不存在: {input_dir}")
        return

    # 获取输出目录
    output_dir = input("请输入输出目录 (默认: batch_results): ").strip()
    output_dir = output_dir if output_dir else 'batch_results'

    # 置信度阈值
    conf_threshold = input("请输入置信度阈值 (默认: 0.25): ").strip()
    conf_threshold = float(conf_threshold) if conf_threshold else 0.25

    # 是否保存图片
    save_images = input("是否保存标注后的图片? (y/n, 默认: y): ").strip().lower()
    save_images = save_images != 'n'

    # 创建处理器并开始处理
    processor = BatchProcessor(model_path, conf_threshold)
    processor.process_directory(input_dir, output_dir, save_images)

    print("\n处理完成!")
    print(f"结果已保存到: {output_dir}")

if __name__ == '__main__':
    main()
