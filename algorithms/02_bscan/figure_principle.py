"""B 扫原理示意图：探头沿四阶试块的平齐面扫查，阶梯在底面。"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from ultrasonic_ndt.simulation.bscan import STEP_THICKNESSES_MM

OUTPUT_PATH = Path(__file__).resolve().parent / "bscan_principle.png"

SCAN_WIDTH_MM = 120.0
BLOCK_WIDTH_MM = 40.0       # y 方向宽度
FIRE_INTERVAL_MM = 1.0      # 示意图中的激发间隔（实际扫查步进为 0.25 mm）
PROBE_X_MM = 105.0          # 探头当前所在位置
PROBE_RADIUS_MM = 2.0
PROBE_HEIGHT_MM = 12.0
BEAM_RADIUS_MM = 0.2
TRANSPARENT_STEP_INDEX = 3  # 探头下方台阶做成半透明，露出内部声束

# 平齐扫查面所在高度（等于最厚台阶的厚度）；阶梯在底面，底波深度随台阶变化。
FLAT_TOP_Z_MM = float(STEP_THICKNESSES_MM[-1])

BLOCK_COLOR = "#9AA7B0"
BLOCK_EDGE = "#5B6770"
PROBE_COLOR = "#0757B9"
BEAM_COLOR = "#FFC107"        # 声束与激发点用黄色
BEAM_TEXT_COLOR = "#F57F17"   # 黄色在浅底上不易读，相关标注用深琥珀
TEXT_COLOR = "#101820"

# 手动 zorder：禁用按深度自动排序，保证探头始终渲染在试块之上
BLOCK_ZORDER = 1
FIRE_ZORDER = 2
BEAM_ZORDER = 3
PROBE_ZORDER = 4


def _cylinder(ax, center_x, center_y, radius, z_bottom, z_top, color, alpha, shade, zorder):
    """绘制一个竖直圆柱侧面（及顶面轮廓）。"""
    theta = np.linspace(0.0, 2.0 * np.pi, 36)
    z = np.linspace(z_bottom, z_top, 2)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = center_x + radius * np.cos(theta_grid)
    y_grid = center_y + radius * np.sin(theta_grid)
    surface = ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha, shade=shade, linewidth=0)
    surface.set_zorder(zorder)
    if alpha >= 1.0:
        top_line, = ax.plot(
            center_x + radius * np.cos(theta),
            center_y + radius * np.sin(theta),
            np.full_like(theta, z_top),
            color=color,
        )
        top_line.set_zorder(zorder)


def _project(ax, x, y, z):
    """把 3D 坐标投影到 3D 坐标系的 2D 数据空间，用于不被遮挡的叠加标注。"""
    x2, y2, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
    return (x2, y2)


def main() -> None:
    figure = plt.figure(figsize=(14, 8), dpi=100, facecolor="#F4F6F8")
    figure.suptitle("探头沿试块平齐面扫查，底面阶梯决定回波深度", fontsize=18, y=0.97)
    ax = figure.add_axes((0.0, 0.0, 1.0, 0.93), projection="3d")

    step_width = SCAN_WIDTH_MM / STEP_THICKNESSES_MM.size
    center_y = BLOCK_WIDTH_MM / 2.0
    probe_thickness = float(STEP_THICKNESSES_MM[TRANSPARENT_STEP_INDEX])

    # 四阶试块：顶面平齐（扫查面），底面呈阶梯状（底波随厚度变化）
    for index, thickness in enumerate(STEP_THICKNESSES_MM):
        block = ax.bar3d(
            index * step_width,
            0.0,
            FLAT_TOP_Z_MM - float(thickness),  # 台阶底面（底波）高度
            step_width,
            BLOCK_WIDTH_MM,
            float(thickness),
            color=BLOCK_COLOR,
            edgecolor=BLOCK_EDGE,
            linewidth=0.5,
            shade=True,
            alpha=0.30 if index == TRANSPARENT_STEP_INDEX else 0.96,
        )
        block.set_zorder(BLOCK_ZORDER)

    # 扫查路径上的激发点（每 1 mm 一个，都在平齐面上）
    fire_x = np.arange(FIRE_INTERVAL_MM / 2.0, SCAN_WIDTH_MM, FIRE_INTERVAL_MM)
    fire_z = np.full_like(fire_x, FLAT_TOP_Z_MM + 0.25)
    fire_points = ax.scatter(
        fire_x,
        np.full_like(fire_x, center_y),
        fire_z,
        s=3,
        color=BEAM_COLOR,
        depthshade=False,
    )
    fire_points.set_zorder(FIRE_ZORDER)

    # 探头贴在平齐面上，声束向下打到当前台阶的底面
    _cylinder(
        ax, PROBE_X_MM, center_y, PROBE_RADIUS_MM,
        FLAT_TOP_Z_MM, FLAT_TOP_Z_MM + PROBE_HEIGHT_MM,
        PROBE_COLOR, alpha=1.0, shade=True, zorder=PROBE_ZORDER,
    )
    _cylinder(
        ax, PROBE_X_MM, center_y, BEAM_RADIUS_MM,
        FLAT_TOP_Z_MM - probe_thickness, FLAT_TOP_Z_MM,
        BEAM_COLOR, alpha=0.85, shade=False, zorder=BEAM_ZORDER,
    )

    ax.set_xlabel("扫查方向 x (mm)", labelpad=6)
    ax.set_ylabel("宽度方向 y (mm)", labelpad=6)
    ax.set_zlabel("深度 z (mm)", labelpad=4)
    ax.set_xlim(0.0, SCAN_WIDTH_MM)
    ax.set_ylim(-2.0, BLOCK_WIDTH_MM + 2.0)
    ax.set_zlim(0.0, FLAT_TOP_Z_MM + PROBE_HEIGHT_MM + 3.0)
    ax.set_xticks([0, 30, 60, 90, 120])
    ax.set_yticks([])
    # z 轴用真实深度标注：顶部（平齐面）为 0，向下递增到底波最深处的 25 mm。
    depth_values_mm = np.concatenate(([0.0], STEP_THICKNESSES_MM[::-1]))
    ax.set_zticks(FLAT_TOP_Z_MM - depth_values_mm)
    ax.set_zticklabels([f"{t:g}" for t in depth_values_mm])
    ax.set_box_aspect((3.0, 1.3, 1.05))
    ax.view_init(elev=20, azim=-60)
    ax.computed_zorder = False  # 用手动 zorder，确保探头压在试块之上
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        pane.set_edgecolor("#D7DCE1")
    ax.grid(False)

    # 2D 叠加层标注：投影到 2D 数据空间后绘制，并用高 zorder 压过 3D 表面
    ANNOTATION_ZORDER = 1000
    for index, thickness in enumerate(STEP_THICKNESSES_MM):
        label_x = index * step_width + step_width / 2.0
        if index == TRANSPARENT_STEP_INDEX:
            label_x -= 8.0  # 避开声束
        ax.annotate(
            f"{thickness:g} mm",
            xy=_project(ax, label_x, 0.0, FLAT_TOP_Z_MM - float(thickness) / 2.0),
            ha="center",
            va="center",
            fontsize=11,
            color=TEXT_COLOR,
            zorder=ANNOTATION_ZORDER,
        )
    ax.annotate(
        "探头",
        xy=_project(ax, PROBE_X_MM, center_y, FLAT_TOP_Z_MM + PROBE_HEIGHT_MM + 1.0),
        ha="center",
        va="bottom",
        fontsize=12,
        color=PROBE_COLOR,
        zorder=ANNOTATION_ZORDER,
    )
    ax.annotate(
        "声束",
        xy=_project(ax, PROBE_X_MM + 4.5, center_y, FLAT_TOP_Z_MM - probe_thickness / 2.0),
        ha="left",
        va="center",
        fontsize=10,
        color=BEAM_TEXT_COLOR,
        zorder=ANNOTATION_ZORDER,
    )
    ax.annotate(
        "",
        xy=_project(ax, 27.0, center_y, FLAT_TOP_Z_MM + 2.0),
        xytext=_project(ax, 5.0, center_y, FLAT_TOP_Z_MM + 2.0),
        arrowprops={"arrowstyle": "->", "color": TEXT_COLOR, "linewidth": 1.6},
        zorder=ANNOTATION_ZORDER,
    )
    ax.annotate(
        "扫查方向",
        xy=_project(ax, 16.0, center_y, FLAT_TOP_Z_MM + 4.0),
        ha="center",
        va="bottom",
        fontsize=12,
        color=TEXT_COLOR,
        zorder=ANNOTATION_ZORDER,
    )
    figure.savefig(OUTPUT_PATH, dpi=100, facecolor=figure.get_facecolor())
    plt.close(figure)
    print(f"Saved figure: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
