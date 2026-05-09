"""
主启动脚本 - 统一入口
提供菜单选择不同功能
"""
import os
import sys

def print_menu():
    """打印主菜单"""
    print("\n" + "="*60)
    print("墙面裂缝检测系统 - 主菜单")
    print("="*60)
    print("\n【数据准备】")
    print("  1. 快速开始 - 环境检查和测试")
    print("  2. 数据集分析 - 统计和可视化")
    print("\n【模型训练】")
    print("  3. 训练模型 - 开始训练")
    print("  4. 可视化训练结果 - 查看训练曲线")
    print("\n【模型推理】")
    print("  5. 单张图片预测")
    print("  6. 批量处理 - 生成报告")

    print("  7. 实时检测 - 摄像头/视频")
    print("\n【模型优化】")
    print("  8. 模型导出 - ONNX/TensorRT等")
    print("  9. 模型评估 - 性能测试")
    print("\n【Web界面】")
    print(" 10. 启动Web界面 - Gradio")
    print("\n【高级功能】")
    print(" 11. 性能基准测试")
    print(" 12. 数据增强预览")
    print("\n【其他】")
    print("  0. 退出")
    print("="*60)

def run_script(script_name):
    """运行指定脚本"""
    if not os.path.exists(script_name):
        print(f"错误: 脚本不存在 - {script_name}")
        return False

    print(f"\n启动: {script_name}")
    print("-"*60)
    os.system(f"python {script_name}")
    return True

def main():
    """主函数"""
    while True:
        print_menu()

        try:
            choice = input("\n请选择功能 (0-10): ").strip()

            if choice == '0':
                print("\n感谢使用！再见！")
                break

            elif choice == '1':
                run_script("quick_start.py")

            elif choice == '2':
                run_script("analyze_dataset.py")

            elif choice == '3':
                print("\n开始训练模型...")
                print("注意: 训练可能需要较长时间，建议使用GPU")
                confirm = input("确认开始训练? (y/n): ").strip().lower()
                if confirm == 'y':
                    run_script("train_crack_seg.py")

            elif choice == '4':
                run_script("visualize_results.py")

            elif choice == '5':
                run_script("predict_crack_seg.py")

            elif choice == '6':
                run_script("batch_process.py")

            elif choice == '7':
                run_script("realtime_detect.py")

            elif choice == '8':
                run_script("export_model.py")

            elif choice == '9':
                print("\n模型评估")
                model_path = input("请输入模型路径 (默认: runs/segment/crack_seg/weights/best.pt): ").strip()
                if not model_path:
                    model_path = "runs/segment/crack_seg/weights/best.pt"

                if os.path.exists(model_path):
                    from ultralytics import YOLO
                    model = YOLO(model_path)
                    print("\n正在评估模型...")
                    metrics = model.val(data='crack-seg/data.yaml')
                    print("\n评估完成!")
                else:
                    print(f"模型不存在: {model_path}")

            elif choice == '10':
                run_script("web_app.py")

            elif choice == '11':
                run_script("benchmark.py")

            elif choice == '12':
                run_script("preview_augmentation.py")

            else:
                print("\n无效的选项，请重新选择")

            input("\n按回车键继续...")

        except KeyboardInterrupt:
            print("\n\n操作已取消")
            break
        except Exception as e:
            print(f"\n错误: {e}")
            input("\n按回车键继续...")

if __name__ == '__main__':
    main()
