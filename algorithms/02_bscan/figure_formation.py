"""A 扫堆叠到 B 扫图像的形成示意图：3D 堆叠 → 投影 + 色带。"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from ultrasonic_ndt.processing.bscan import process_bscan
from ultrasonic_ndt.simulation.bscan import simulate_bscan
from ultrasonic_ndt.visualization import white_blue_yellow_red_colormap

OUTPUT_PATH = Path(__file__).resolve().parent / "bscan_formation.png"

TRACE_COUNT = 50              # 3D 侧抽稀显示的 A 扫数量
AMPLITUDE_SCALE_MM = 15.0     # 3D 中幅值轴的显示高度
DISPLAY_DEPTH_MAX_MM = 56.0


def main() -> None:
    bscan = simulate_bscan()
    result = process_bscan(bscan)
    depth_mm = bscan.depth_mm

    figure = plt.figure(figsize=(16, 9), dpi=100, facecolor="#F4F6F8")
    figure.suptitle("每条 A 扫经处理后按扫查位置排开，就得到一幅 B 扫", fontsize=18, y=0.97)

    # 左：3D 堆叠的 A 扫（深度自上而下，与 B 扫的声束方向一致）
    ax3d = figure.add_axes((0.02, 0.02, 0.48, 0.88), projection="3d")
    indices = np.linspace(0, bscan.rf_samples.shape[0] - 1, TRACE_COUNT).astype(int)
    ascan_color = "#E65100"  # 与第一篇最终包络 A 扫的橙色保持一致
    for index in indices:
        position = float(bscan.scan_positions_mm[index])
        amplitude = result.display_amplitude[index] * AMPLITUDE_SCALE_MM
        # 探头轴基线：从表面 (z=0) 沿声束方向向下
        ax3d.plot(
            [position, position], [0.0, 0.0], [0.0, DISPLAY_DEPTH_MAX_MM],
            color="#C9D2D9", linewidth=0.5,
        )
        # A 扫：x 为扫查位置固定不变，y 为幅值侧向偏移（朝向读者），z 为深度（向下）
        ax3d.plot(
            np.full(depth_mm.size, position), -amplitude, depth_mm,
            color=ascan_color, linewidth=0.9,
        )
    ax3d.set_title(f"① 每个位置一条处理后的 A 扫（示意图显示 {TRACE_COUNT} 条）", fontsize=13, pad=8)
    ax3d.set_xlabel("扫查位置 x (mm)", labelpad=6)
    ax3d.set_zlabel("深度 z (mm)", labelpad=6)
    ax3d.set_xlim(0.0, 120.0)
    ax3d.set_ylim(-AMPLITUDE_SCALE_MM, 0.0)
    ax3d.set_zlim(0.0, DISPLAY_DEPTH_MAX_MM)
    ax3d.invert_zaxis()  # 0 在顶部，越大越深
    ax3d.set_xticks([0, 30, 60, 90, 120])
    ax3d.set_yticks([])
    ax3d.set_box_aspect((2.0, 0.8, 1.4))
    ax3d.view_init(elev=22, azim=-72)
    # 把 z 轴（深度）刻度和标签挪回立方体左侧，避免遮挡中间的箭头文字
    try:
        ax3d.zaxis._axinfo["juggled"] = (1, 2, 0)
    except (AttributeError, KeyError):
        pass
    for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
        pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        pane.set_edgecolor("#D7DCE1")
    ax3d.grid(False)

    # 中间：投影说明箭头
    figure.text(0.545, 0.52, "→", fontsize=44, ha="center", va="center", color="#101820")
    figure.text(0.545, 0.44, "幅值映射为颜色", fontsize=11, ha="center", va="center", color="#101820")

    # 右：投影后的平面 B 扫
    ax2d = figure.add_axes((0.62, 0.14, 0.34, 0.68))
    extent = (0.0, 120.0, float(depth_mm[-1]), 0.0)
    image = ax2d.imshow(
        result.display_amplitude.T,
        cmap=white_blue_yellow_red_colormap(),
        vmin=0.0,
        vmax=1.0,
        interpolation="bilinear",
        aspect="auto",
        extent=extent,
    )
    ax2d.set_title("② 投影成平面 B 扫图像", fontsize=13)
    ax2d.set(xlabel="扫查位置 x (mm)", ylabel="深度 z (mm)")
    ax2d.set_xlim(0.0, 120.0)
    ax2d.set_ylim(DISPLAY_DEPTH_MAX_MM, 0.0)
    colorbar = figure.colorbar(image, ax=ax2d, orientation="horizontal", pad=0.10, fraction=0.05)
    colorbar.set_label("归一化幅值")

    figure.savefig(OUTPUT_PATH, dpi=100, facecolor=figure.get_facecolor())
    plt.close(figure)
    print(f"Saved figure: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
