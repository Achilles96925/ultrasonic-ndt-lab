"""B 扫侧视图交互示例：包络 / RF 两种显示可切换。"""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from ultrasonic_ndt.processing.bscan import process_bscan
from ultrasonic_ndt.simulation.bscan import simulate_bscan, thickness_profile_mm
from ultrasonic_ndt.visualization import white_blue_yellow_red_colormap
from ultrasonic_ndt.widgets import make_circular_checkbox


OUTPUT_DIRECTORY = Path(__file__).resolve().parent
DISPLAY_DEPTH_MAX_MM = 56.0


class BScanWorkbench:
    """显示 B 扫，并联动查看指定扫查位置的 A 扫（包络 / RF 可切换）。"""

    def __init__(self) -> None:
        self.bscan = simulate_bscan()
        self.result = process_bscan(self.bscan)
        self.colormap = white_blue_yellow_red_colormap()
        self.rf_limit = float(np.percentile(np.abs(self.result.filtered_rf), 99.7))
        self.use_envelope = True
        self.selected_index = int(np.argmin(np.abs(self.bscan.scan_positions_mm - 105.0)))
        self._background = None
        self._create_figure()
        self._refresh_display()

    def _create_figure(self) -> None:
        self.figure = plt.figure(figsize=(16, 9), dpi=100, facecolor="#F4F6F8")
        manager = getattr(self.figure.canvas, "manager", None)
        if manager is not None:
            manager.set_window_title("超声无损检测实验室 - B 扫")

        self.figure.suptitle("B扫：四阶厚度校准试块", fontsize=18, y=0.965)
        self.bscan_axis = self.figure.add_axes((0.065, 0.18, 0.64, 0.70), facecolor="white")
        self.ascan_axis = self.figure.add_axes((0.77, 0.18, 0.19, 0.70), facecolor="white")

        extent = (
            float(self.bscan.scan_positions_mm[0]),
            float(self.bscan.scan_positions_mm[-1]),
            float(self.bscan.depth_mm[-1]),
            float(self.bscan.depth_mm[0]),
        )
        self.image = self.bscan_axis.imshow(
            self.result.display_amplitude.T,
            cmap=self.colormap,
            vmin=0.0,
            vmax=1.0,
            interpolation="bilinear",
            aspect="auto",
            extent=extent,
        )
        self.scan_line = self.bscan_axis.axvline(0.0, color="#101820", linewidth=1.4, linestyle="--")
        self.bscan_axis.set(xlabel="扫查位置 x (mm)", ylabel="深度 z (mm)")
        self.bscan_axis.set_ylim(DISPLAY_DEPTH_MAX_MM, 0.0)
        self.bscan_axis.grid(False)

        colorbar_axis = self.figure.add_axes((0.065, 0.105, 0.64, 0.025))
        self.colorbar = self.figure.colorbar(self.image, cax=colorbar_axis, orientation="horizontal")

        self.trace_line, = self.ascan_axis.plot([], [], linewidth=1.3)
        self.ascan_axis.set(xlabel="深度 z (mm)", xlim=(0.0, DISPLAY_DEPTH_MAX_MM))
        self.ascan_axis.grid(alpha=0.18)

        check_axis = self.figure.add_axes((0.77, 0.045, 0.13, 0.04))
        self.envelope_check = make_circular_checkbox(check_axis, ["包络显示"], [True])
        self.envelope_check.on_clicked(self._on_envelope_toggled)

        slider_axis = self.figure.add_axes((0.77, 0.10, 0.13, 0.035), facecolor="#E4E8EC")
        self.position_slider = Slider(
            slider_axis,
            "x (mm)",
            float(self.bscan.scan_positions_mm[0]),
            float(self.bscan.scan_positions_mm[-1]),
            valinit=float(self.bscan.scan_positions_mm[self.selected_index]),
            valstep=float(np.diff(self.bscan.scan_positions_mm)[0]),
        )
        self.position_slider.on_changed(self._on_slider_changed)

        save_axis = self.figure.add_axes((0.915, 0.095, 0.055, 0.045))
        self.save_button = Button(save_axis, "保存", color="#E3E8EC", hovercolor="#CFD8DC")
        self.save_button.on_clicked(self._on_save_clicked)
        self.figure.canvas.mpl_connect("button_press_event", self._on_image_clicked)
        self.figure.canvas.mpl_connect("resize_event", self._on_resize)

    def _dynamic_artists(self):
        """拖动时需要重绘的元素：扫描线、A 扫曲线、A 扫标题。"""
        return (self.scan_line, self.trace_line, self.ascan_axis.title)

    def _capture_background(self) -> None:
        """绘制一帧不含动态元素的干净背景并缓存，供拖动时 blit 复用。

        避免 481×1000 的 imshow 在每次滑条拖动时被全量重绘。
        """
        try:
            for artist in self._dynamic_artists():
                artist.set_visible(False)
            self.figure.canvas.draw()
            self._background = self.figure.canvas.copy_from_bbox(self.figure.bbox)
            for artist in self._dynamic_artists():
                artist.set_visible(True)
            self._blit_dynamic()
        except Exception:
            for artist in self._dynamic_artists():
                artist.set_visible(True)
            self._background = None
            self.figure.canvas.draw_idle()

    def _blit_dynamic(self) -> None:
        """恢复缓存背景，只重绘动态元素后刷到屏幕。"""
        if self._background is None:
            self.figure.canvas.draw_idle()
            return
        try:
            self.figure.canvas.restore_region(self._background)
            self.bscan_axis.draw_artist(self.scan_line)
            self.ascan_axis.draw_artist(self.trace_line)
            self.ascan_axis.draw_artist(self.ascan_axis.title)
            self.figure.canvas.blit(self.figure.bbox)
        except Exception:
            self._background = None
            self.figure.canvas.draw_idle()

    def _refresh_display(self) -> None:
        """按当前显示模式刷新 B 扫图像、色带和 A 扫坐标轴。"""
        if self.use_envelope:
            self.image.set_data(self.result.display_amplitude.T)
            self.image.set_cmap(self.colormap)
            self.image.set_clim(0.0, 1.0)
            self.image.set_interpolation("bilinear")
            self.bscan_axis.set_title("包络 B扫")
            self.colorbar.set_ticks((0.0, 0.25, 0.5, 0.75, 1.0))
            self.colorbar.set_ticklabels(("0", "25", "50", "75", "100"))
            self.colorbar.set_label("归一化幅值 (%)", labelpad=4)
            self.ascan_axis.set_ylim(0.0, 2047.0)
            self.ascan_axis.set_yticks((0, 512, 1024, 1536, 2047))
            self.ascan_axis.set_ylabel("包络幅值")
            self.trace_line.set_color("#D7191C")
        else:
            self.image.set_data(self.result.filtered_rf.T)
            self.image.set_cmap("gray")
            self.image.set_clim(-self.rf_limit, self.rf_limit)
            self.image.set_interpolation("nearest")
            self.bscan_axis.set_title("RF B扫")
            self.colorbar.set_ticks((-self.rf_limit, 0.0, self.rf_limit))
            self.colorbar.set_ticklabels(("-100%", "0", "+100%"))
            self.colorbar.set_label("RF 幅值", labelpad=4)
            self.ascan_axis.set_ylim(-2048.0, 2047.0)
            self.ascan_axis.set_yticks((-2048, -1024, 0, 1024, 2047))
            self.ascan_axis.set_ylabel("RF 幅值")
            self.trace_line.set_color("#0757B9")
        self._background = None
        self._update_ascan()

    def _update_ascan(self) -> None:
        if self.use_envelope:
            trace = self.result.envelope[self.selected_index]
        else:
            trace = self.result.filtered_rf[self.selected_index]
        self.trace_line.set_data(self.bscan.depth_mm, trace)
        position_mm = float(self.bscan.scan_positions_mm[self.selected_index])
        thickness_mm = float(thickness_profile_mm(np.array([position_mm]))[0])
        self.scan_line.set_xdata([position_mm, position_mm])
        self.ascan_axis.set_title(f"x = {position_mm:.2f} mm，厚度 {thickness_mm:g} mm")
        if self._background is None:
            self._capture_background()
        else:
            self._blit_dynamic()

    def _select_position(self, position_mm: float) -> None:
        self.selected_index = int(np.argmin(np.abs(self.bscan.scan_positions_mm - position_mm)))
        actual_position = float(self.bscan.scan_positions_mm[self.selected_index])
        if not np.isclose(self.position_slider.val, actual_position):
            self.position_slider.set_val(actual_position)
        else:
            self._update_ascan()

    def _on_slider_changed(self, value: float) -> None:
        self.selected_index = int(np.argmin(np.abs(self.bscan.scan_positions_mm - value)))
        self._update_ascan()

    def _on_image_clicked(self, event) -> None:
        if event.inaxes is self.bscan_axis and event.xdata is not None:
            self._select_position(float(event.xdata))

    def _on_envelope_toggled(self, _label: str) -> None:
        self.use_envelope = not self.use_envelope
        self._refresh_display()

    def _on_resize(self, _event) -> None:
        self._background = None

    def _on_save_clicked(self, _) -> None:
        mode = "envelope" if self.use_envelope else "rf"
        output_path = OUTPUT_DIRECTORY / f"bscan_workbench_{mode}.png"
        self.save(output_path)
        print(f"Saved figure: {output_path}")

    def save(self, output_path: Path) -> None:
        self.figure.savefig(output_path, dpi=100, facecolor=self.figure.get_facecolor())

    def run(self) -> None:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="B 扫侧视图演示")
    parser.add_argument(
        "--save-views",
        action="store_true",
        help="分别保存包络与 RF 两种视图后退出",
    )
    args = parser.parse_args()

    workbench = BScanWorkbench()
    if args.save_views:
        workbench.save(OUTPUT_DIRECTORY / "bscan_workbench_envelope.png")
        print(f"Saved figure: {OUTPUT_DIRECTORY / 'bscan_workbench_envelope.png'}")
        workbench.envelope_check.set_active(0)
        workbench.save(OUTPUT_DIRECTORY / "bscan_workbench_rf.png")
        print(f"Saved figure: {OUTPUT_DIRECTORY / 'bscan_workbench_rf.png'}")
        plt.close(workbench.figure)
        return
    workbench.run()


if __name__ == "__main__":
    main()
