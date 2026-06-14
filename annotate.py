"""
裂缝标注工具 — matplotlib 交互版，鼠标点击画多边形
==============================================================
操作:
  左键     →  添加多边形顶点
  右键     →  闭合当前裂缝（自动连回首点）
  N 键     →  下一张图（自动保存）
  D 键     →  删除最后一条裂缝
  C 键     →  清空当前图所有标注
  U 键     →  撤销最后一个顶点
  S 键     →  保存标注
  Q / ESC  →  退出

输出: labels/ 目录下同名 .txt (YOLO 分割格式: 0 x1 y1 x2 y2 ...)
"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os
import sys
from pathlib import Path

for fname in ['Microsoft YaHei', 'SimHei']:
    hits = [f for f in fm.fontManager.ttflist if fname in f.name]
    if hits:
        plt.rcParams['font.family'] = hits[0].name
        break
plt.rcParams['axes.unicode_minus'] = False

COLORS = ['#FF0000', '#00FF00', '#0066FF', '#FF6600', '#CC00CC', '#00CCCC']
ALPHA_FILL = 0.25


def save_label(txt_path, cracks, w, h):
    with open(txt_path, 'w', encoding='utf-8') as f:
        for crack in cracks:
            if len(crack) < 3:
                continue
            pts = ' '.join(f'{x/w:.6f} {y/h:.6f}' for x, y in crack)
            f.write(f'0 {pts}\n')


class Annotator:
    def __init__(self, img_dir, labels_dir='labels'):
        self.img_dir = Path(img_dir)
        self.labels_dir = Path(labels_dir)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        exts = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.PNG']
        self.images = sorted([f for f in self.img_dir.iterdir() if f.suffix in exts])
        if not self.images:
            print(f"未找到图片: {img_dir}")
            sys.exit(1)
        self.idx = 0
        self.cracks = []         # [[(x,y),...], ...]
        self.polygon = []        # 正在画的多边形顶点
        self.img = None
        self.img_path = None
        self.w, self.h = 0, 0
        self.ax = None
        self.fig = None
        self.artists = []

        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self._load()

    def _load(self):
        self.cracks = []
        self.polygon = []
        self.img_path = self.images[self.idx]
        self.img = plt.imread(str(self.img_path))
        self.h, self.w = self.img.shape[:2]

        # 读取已有标注
        txt_path = self.labels_dir / (self.img_path.stem + '.txt')
        if txt_path.exists():
            with open(txt_path, encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 7:
                        continue
                    nums = [float(x) for x in parts[1:]]
                    pts = [(int(nums[i]*self.w), int(nums[i+1]*self.h)) for i in range(0, len(nums), 2)]
                    if len(pts) >= 3:
                        self.cracks.append(pts)

        self._draw()
        self.fig.canvas.manager.set_window_title(
            f'{self.img_path.name}  ({self.idx+1}/{len(self.images)})'
        )

    def _draw(self):
        self.ax.clear()
        self.artists = []
        self.ax.imshow(self.img)
        self.ax.axis('off')

        # 画已完成的裂缝
        for ci, crack in enumerate(self.cracks):
            color = COLORS[ci % len(COLORS)]
            xs = [p[0] for p in crack] + [crack[0][0]]
            ys = [p[1] for p in crack] + [crack[0][1]]
            self.ax.fill(xs, ys, color=color, alpha=ALPHA_FILL)
            self.ax.plot(xs, ys, '-', color=color, linewidth=2)
            self.ax.plot(xs, ys, 'o', color=color, markersize=4)

        # 画正在画的多边形
        if len(self.polygon) >= 1:
            xs = [p[0] for p in self.polygon]
            ys = [p[1] for p in self.polygon]
            if len(self.polygon) >= 3:
                self.ax.plot(xs + [xs[0]], ys + [ys[0]], '--', color='white', linewidth=1)
                self.ax.fill(xs + [xs[0]], ys + [ys[0]], color='orange', alpha=0.1)
            self.ax.plot(xs, ys, 'o-', color='orange', markersize=6, linewidth=1.5)

        # 状态
        total_v = sum(len(c) for c in self.cracks) + len(self.polygon)
        title = f'Cracks: {len(self.cracks)}  顶点: {total_v}'
        self.ax.set_title(title, fontsize=13, fontweight='bold', color='#333')

        # 图例
        info = 'L-click=add  R-click=close  N=next  D=delete  U=undo  C=clear  S=save  Q=quit'
        self.fig.text(0.5, 0.01, info, ha='center', fontsize=10, color='#666')

        self.fig.canvas.draw_idle()

    def _save(self):
        if not self.cracks and not self.polygon:
            return
        if self.polygon and len(self.polygon) >= 3:
            self.cracks.append(self.polygon[:])
            self.polygon = []
        txt_path = self.labels_dir / (self.img_path.stem + '.txt')
        save_label(txt_path, self.cracks, self.w, self.h)
        print(f"Saved: {self.img_path.name} -> {txt_path.name}")

    def _next(self):
        self._save()
        self.idx = (self.idx + 1) % len(self.images)
        self._load()

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 1:  # 左键: 加点
            self.polygon.append((int(event.xdata), int(event.ydata)))
            self._draw()
        elif event.button == 3:  # 右键: 闭合
            if len(self.polygon) >= 3:
                self.cracks.append(self.polygon[:])
                self.polygon = []
                self._draw()

    def _on_key(self, event):
        if event.key == 'n':
            self._next()
        elif event.key == 'd':
            if self.cracks:
                self.cracks.pop()
            self._draw()
        elif event.key == 'c':
            self.cracks = []
            self.polygon = []
            self._draw()
        elif event.key == 'u':
            if self.polygon:
                self.polygon.pop()
            elif self.cracks:
                self.cracks.pop()
            self._draw()
        elif event.key == 's':
            self._save()
            self._draw()
        elif event.key in ('q', 'escape'):
            self._save()
            print("退出")
            plt.close()

    def run(self):
        print(f"\n裂缝标注工具 (matplotlib 版)")
        print(f"图片: {self.img_dir}  ({len(self.images)} 张)")
        print(f"标注: {self.labels_dir}")
        print(f"\n左键加点 | 右键闭合 | N下一张 | D删裂缝 | U撤点 | C清空 | S保存 | Q退出\n")
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    d = sys.argv[1] if len(sys.argv) > 1 else input("图片目录: ").strip()
    if not d or not os.path.isdir(d):
        print("目录不存在"); sys.exit(1)
    lbl = sys.argv[2] if len(sys.argv) > 2 else 'labels'
    app = Annotator(d, lbl)
    app.run()
