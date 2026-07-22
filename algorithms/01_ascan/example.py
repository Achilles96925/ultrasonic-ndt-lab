"""A 扫交互工作台。"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RangeSlider, Slider
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from ultrasonic_ndt.processing.gate import find_gate_peak
from ultrasonic_ndt.processing.pipeline import process_ascan
from ultrasonic_ndt.processing.tgc import TgcCurve
from ultrasonic_ndt.simulation.ascan import simulate_ascan
from ultrasonic_ndt.widgets import checkbox_markers, make_circular_checkbox


class MovableRangeSlider(RangeSlider):
    """支持拖动中间选区并保持带宽不变的双端滑条。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._dragging_band = False
        self._band_start_value = 0.0
        self._band_initial_limits = (0.0, 0.0)

    def _update(self, event) -> None:
        if self.ignore(event) or event.button != 1:
            return
        if event.name == "button_press_event" and self.ax.contains(event)[0]:
            low, high = self.val
            if low < event.xdata < high:
                self._dragging_band = True
                self._band_start_value = event.xdata
                self._band_initial_limits = (low, high)
                event.canvas.grab_mouse(self.ax)
                return
        if self._dragging_band:
            if event.name == "button_release_event":
                self._dragging_band = False
                event.canvas.release_mouse(self.ax)
                return
            if event.xdata is None:
                return
            low, high = self._band_initial_limits
            width = high - low
            new_low = np.clip(low + event.xdata - self._band_start_value, self.valmin, self.valmax - width)
            self.set_val((new_low, new_low + width))
            return
        super()._update(event)


class AScanWorkbench:
    """负责参数绑定、绘图和鼠标交互的 A 扫界面。"""

    ADC_MIN = -2048
    ADC_MAX = 2047
    DISPLAY_LIMIT = 2048
    GATE_MIN_WIDTH_US = 0.05
    GATE_HANDLE_TOLERANCE_US = 0.25
    TGC_NODE_PICK_RADIUS_PX = 14
    FIGURE_SIZE = (16, 9)
    FIGURE_DPI = 100

    # Normalized figure coordinates for every layout element.
    LAYOUT = {
        "raw_axis": (0.06, 0.55, 0.40, 0.38),
        "comparison_axis": (0.54, 0.55, 0.40, 0.38),
        "final_axis": (0.54, 0.10, 0.40, 0.36),
        "parameter_title": (0.07, 0.48),
        "frequency_label": (0.07, 0.425),
        "frequency_slider": (0.07, 0.385, 0.3, 0.035),
        "filter_label": (0.07, 0.35),
        "filter_slider": (0.07, 0.31, 0.3, 0.035),
        "envelope_toggle": (0.07, 0.24, 0.13, 0.05),
        "envelope_marker": (0.25, 0.5),
        "envelope_label": (0.5, 0.5),
        "save_button": (0.07, 0.18, 0.13, 0.05),
        "instructions": (0.07, 0.15),
    }

    def __init__(self) -> None:
        self.probe_frequency_mhz = 5.0
        self.filter_limits_mhz = (2.0, 8.0)
        self.envelope_enabled = True
        self.gate_start_us, self.gate_end_us = 3.3, 5.2
        self.gate_drag_mode: str | None = None
        self.gate_drag_anchor_us = 0.0
        self.tgc_drag_index: int | None = None
        self.tgc = TgcCurve(
            times_us=np.array([0.0, 3.2, 6.4, 9.99]),
            gains_db=np.array([0.0, 0.0, 0.0, 0.0]),
        )
        self.frequency_preview_pending = False

        self._load_ascan()
        self._create_figure()
        self._recompute_pipeline()
        self._update_all_artists()

    def _load_ascan(self) -> None:
        self.ascan = simulate_ascan(center_frequency_hz=self.probe_frequency_mhz * 1e6)
        self.time_us = self.ascan.time_seconds * 1e6
        self.time_limit_us = float(self.time_us[-1])
        self.tgc.set_end_time(self.time_limit_us)

    def _create_figure(self) -> None:
        self.figure = plt.figure(figsize=self.FIGURE_SIZE, dpi=self.FIGURE_DPI)
        self.figure.canvas.manager.set_window_title("超声无损检测实验室 - A 扫")
        self._try_lock_window_size()

        # 左侧显示原始数据和参数，右侧显示处理对比与最终结果。
        self.raw_axis = self.figure.add_axes(self.LAYOUT["raw_axis"])
        self.comparison_axis = self.figure.add_axes(self.LAYOUT["comparison_axis"])
        self.final_axis = self.figure.add_axes(self.LAYOUT["final_axis"])

        self.raw_line, = self.raw_axis.plot([], [], color="#616161", linewidth=1.1, label="原始 A 扫")
        self.comparison_raw_line, = self.comparison_axis.plot(
            [], [], color="#9E9E9E", linewidth=0.9, label="原始 A 扫"
        )
        self.filtered_line, = self.comparison_axis.plot(
            [], [], color="#00897B", linewidth=1.1, label="滤波 A 扫"
        )
        self.envelope_line, = self.comparison_axis.plot(
            [], [], color="#E65100", linewidth=1.5, label="包络 A 扫"
        )
        self.final_line, = self.final_axis.plot([], [], color="#E65100", linewidth=1.5, label="最终 A 扫")
        self.peak_marker, = self.final_axis.plot([], [], "o", color="#D32F2F", markersize=6, label="门内峰值")
        self.peak_annotation = self.final_axis.annotate(
            "", xy=(0, 0), xytext=(0, 0), fontsize=9, color="#D32F2F",
            arrowprops={"arrowstyle": "->", "color": "#D32F2F", "linewidth": 1.2},
        )

        self.gate_patch = self.final_axis.axvspan(
            self.gate_start_us, self.gate_end_us, color="#43A047", alpha=0.18, zorder=0
        )
        self.gate_left = self.final_axis.axvline(self.gate_start_us, color="#2E7D32", linewidth=2)
        self.gate_right = self.final_axis.axvline(self.gate_end_us, color="#2E7D32", linewidth=2)

        self.gain_axis = self.final_axis.twinx()
        self.tgc_curve, = self.gain_axis.plot([], [], "--", color="#7B1FA2", linewidth=1.5, label="TGC 曲线")
        self.tgc_nodes, = self.gain_axis.plot([], [], "o", color="#7B1FA2", markersize=7)
        self.tgc_labels = [self._create_tgc_label() for _ in self.tgc.times_us]

        for axis, title in (
            (self.raw_axis, "原始A扫数据"),
            (self.comparison_axis, "波形对比"),
            (self.final_axis, "最终显示"),
        ):
            axis.set_title(title, fontsize=14, pad=8)
            axis.set_xlim(0, self.time_limit_us)
            axis.set_ylim(-self.DISPLAY_LIMIT, self.DISPLAY_LIMIT)
            axis.tick_params(labelsize=9, length=4, pad=3)
            axis.grid(alpha=0.2, linewidth=0.6)
        self.raw_axis.set_ylabel("波幅", fontsize=11, labelpad=3)
        self.comparison_axis.set_ylabel("波幅", fontsize=11, labelpad=3)
        self.final_axis.set(xlabel="时间 (μs)", ylabel="波幅")
        self.final_axis.xaxis.label.set_size(11)
        self.final_axis.yaxis.label.set_size(11)
        self.gain_axis.set_ylim(-40, 40)
        self.gain_axis.set_ylabel("TGC 增益 (dB)", fontsize=11, color="#7B1FA2", labelpad=4)
        self.gain_axis.tick_params(axis="y", colors="#7B1FA2", labelsize=9, length=4, pad=3)

        self.raw_axis.legend(fontsize=9, loc="upper right", framealpha=0.7)
        self.comparison_axis.legend(fontsize=9, loc="upper right", framealpha=0.7)
        self.final_axis.legend(fontsize=9, loc="upper left", framealpha=0.7)

        self._create_parameter_controls()
        self.figure.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.figure.canvas.mpl_connect("motion_notify_event", self._on_mouse_motion)
        self.figure.canvas.mpl_connect("button_release_event", self._on_mouse_release)

    def _try_lock_window_size(self) -> None:
        """尽量锁定常见桌面后端的窗口尺寸，同时保留兼容性回退。"""
        window = getattr(self.figure.canvas.manager, "window", None)
        try:
            if hasattr(window, "resizable"):  # Tk 后端
                window.resizable(False, False)
            elif hasattr(window, "setFixedSize"):  # Qt 后端
                width, height = (int(value * self.FIGURE_DPI) for value in self.FIGURE_SIZE)
                window.setFixedSize(width, height)
        except Exception:
            pass

    def _create_parameter_controls(self) -> None:
        self.figure.text(*self.LAYOUT["parameter_title"], "参数区域", fontsize=15, fontweight="bold")
        self.figure.text(*self.LAYOUT["frequency_label"], "探头中心频率", fontsize=11)
        frequency_axis = self.figure.add_axes(self.LAYOUT["frequency_slider"])
        self.frequency_slider = Slider(
            frequency_axis, "MHz", 0.5, 25.0, valinit=self.probe_frequency_mhz, valstep=0.5
        )
        self.frequency_slider.label.set_fontsize(9)
        self.frequency_slider.valtext.set_fontsize(9)
        self.frequency_slider.on_changed(self._on_frequency_changed)

        self.figure.text(*self.LAYOUT["filter_label"], "带通滤波范围", fontsize=11)
        filter_axis = self.figure.add_axes(self.LAYOUT["filter_slider"])
        self.filter_slider = MovableRangeSlider(
            filter_axis, "MHz", 0.1, 25.0, valinit=self.filter_limits_mhz, valstep=0.1
        )
        self.filter_slider.label.set_fontsize(9)
        self.filter_slider.valtext.set_fontsize(9)
        self.filter_slider.on_changed(self._on_filter_changed)

        checkbox_axis = self.figure.add_axes(self.LAYOUT["envelope_toggle"])
        self.envelope_checkbox = make_circular_checkbox(checkbox_axis, ["显示包络"], [True])

        # 参数面板里把勾选框和文字重新定位到统一位置
        markers = checkbox_markers(self.envelope_checkbox)
        self.envelope_checkbox._frames.set_offsets([self.LAYOUT["envelope_marker"]])
        markers.set_offsets([self.LAYOUT["envelope_marker"]])
        for label in self.envelope_checkbox.labels:
            label.set_fontsize(11)
            label.set_position(self.LAYOUT["envelope_label"])
            label.set_horizontalalignment("center")
        self.envelope_checkbox.on_clicked(self._on_envelope_toggled)
        save_axis = self.figure.add_axes(self.LAYOUT["save_button"])
        self.save_button = Button(save_axis, "保存图片", color="#E3F2FD", hovercolor="#BBDEFB")
        self.save_button.label.set_fontsize(10)
        self.save_button.label.set_horizontalalignment("center")
        self.save_button.on_clicked(self._on_save_clicked)
        self.figure.text(
            *self.LAYOUT["instructions"],
            "绿色区域：拖动闸门或边缘\n紫色节点：左键拖动；右键删除\n双击紫色曲线：添加节点\n滤波蓝色区域：整体平移范围\nTGC 范围：-40 至 +40 dB",
            fontsize=9, va="top",
        )

    def _recompute_pipeline(self) -> None:
        low_mhz, high_mhz = self.filter_limits_mhz
        self.result = process_ascan(
            self.ascan, low_mhz, high_mhz, self.tgc, adc_min=self.ADC_MIN, adc_max=self.ADC_MAX
        )

    def _create_tgc_label(self):
        return self.gain_axis.annotate(
            "", (0, 0), xytext=(0, 7), textcoords="offset points",
            ha="center", color="#7B1FA2", fontsize=9,
        )

    def _update_all_artists(self) -> None:
        self.raw_line.set_data(self.time_us, self.ascan.samples)
        self.comparison_raw_line.set_data(self.time_us, self.ascan.samples)
        self.filtered_line.set_data(self.time_us, self.result.filtered)
        self.envelope_line.set_data(self.time_us, self.result.filtered_envelope)
        self.envelope_line.set_visible(self.envelope_enabled)
        final_signal = self.result.compensated_envelope if self.envelope_enabled else self.result.compensated
        self.final_line.set_data(self.time_us, final_signal)
        self.final_line.set_color("#E65100" if self.envelope_enabled else "#1565C0")
        self.final_line.set_label("最终包络 A 扫" if self.envelope_enabled else "最终滤波 A 扫")
        self.tgc_curve.set_data(self.time_us, self.result.tgc_gain_db)
        self.tgc_nodes.set_data(self.tgc.times_us, self.tgc.gains_db)
        for label, time_us, gain_db in zip(self.tgc_labels, self.tgc.times_us, self.tgc.gains_db):
            label.xy = (time_us, gain_db)
            label.set_text(f"{gain_db:+.1f} dB")
        self._update_gate_artists()
        self._update_gate_peak()
        self.figure.canvas.draw_idle()

    def _update_gate_artists(self) -> None:
        self.gate_patch.set_x(self.gate_start_us)
        self.gate_patch.set_width(self.gate_end_us - self.gate_start_us)
        self.gate_left.set_xdata([self.gate_start_us, self.gate_start_us])
        self.gate_right.set_xdata([self.gate_end_us, self.gate_end_us])

    def _update_gate_peak(self) -> None:
        measurement = self.result.compensated_envelope if self.envelope_enabled else self.result.compensated
        peak = find_gate_peak(
            self.time_us, measurement, self.gate_start_us, self.gate_end_us,
            use_absolute_value=not self.envelope_enabled,
        )
        if peak is not None:
            peak_time_us = peak.time_us
            peak_amplitude = peak.amplitude
            self.peak_marker.set_data([peak_time_us], [peak_amplitude])
            annotation_x = min(self.gate_end_us + 0.8, self.time_limit_us - 3.6)
            annotation_y = peak_amplitude * 0.85 if peak_amplitude >= 0 else peak_amplitude * 1.15
            self.peak_annotation.xy = (peak_time_us, peak_amplitude)
            self.peak_annotation.set_position((annotation_x, annotation_y))
            self.peak_annotation.set_text(f"峰值：{peak_amplitude:.0f}\nTOF：{peak_time_us:.2f} μs")
            self.final_axis.legend(fontsize=9, loc="upper left", framealpha=0.7)

    def _on_frequency_changed(self, value: float) -> None:
        self.probe_frequency_mhz = float(value)
        self._load_ascan()
        # 滑条回调会连续触发；拖动时仅刷新原始 A 扫，松开后再完成完整处理。
        self.raw_line.set_data(self.time_us, self.ascan.samples)
        self.raw_axis.set_xlim(0, self.time_limit_us)
        self.frequency_preview_pending = True
        self.figure.canvas.draw_idle()

    def _on_filter_changed(self, values: tuple[float, float]) -> None:
        self.filter_limits_mhz = tuple(float(value) for value in values)
        self._recompute_pipeline()
        self._update_all_artists()

    def _on_envelope_toggled(self, _: str) -> None:
        self.envelope_enabled = not self.envelope_enabled
        self._update_all_artists()

    def _on_save_clicked(self, _) -> None:
        output_path = Path(__file__).with_name("ascan_pipeline.png")
        self.figure.savefig(output_path, dpi=100)
        print(f"Saved figure: {output_path}")

    def _find_tgc_node(self, event) -> int | None:
        points = np.column_stack((self.tgc.times_us, self.tgc.gains_db))
        pixels = self.gain_axis.transData.transform(points)
        distances = np.hypot(pixels[:, 0] - event.x, pixels[:, 1] - event.y)
        index = int(np.argmin(distances))
        return index if distances[index] <= self.TGC_NODE_PICK_RADIUS_PX else None

    def _is_near_tgc_curve(self, event) -> bool:
        """判断鼠标是否位于插值后的紫色 TGC 曲线附近。"""
        if event.xdata is None or event.ydata is None:
            return False
        gain_at_click = self.tgc.gain_at(np.array([event.xdata]))[0]
        curve_pixel = self.gain_axis.transData.transform((event.xdata, gain_at_click))
        return np.hypot(curve_pixel[0] - event.x, curve_pixel[1] - event.y) <= self.TGC_NODE_PICK_RADIUS_PX

    def _add_tgc_node(self, time_us: float) -> None:
        insert_index = int(np.searchsorted(self.tgc.times_us, time_us))
        if self.tgc.add_node(time_us):
            self.tgc_labels.insert(insert_index, self._create_tgc_label())
            self._recompute_pipeline()
            self._update_all_artists()

    def _delete_tgc_node(self, index: int) -> None:
        if self.tgc.delete_node(index):
            self.tgc_labels.pop(index).remove()
            self._recompute_pipeline()
            self._update_all_artists()

    def _on_mouse_press(self, event) -> None:
        if event.xdata is None or event.inaxes not in (self.final_axis, self.gain_axis):
            return
        if event.button == 3 and event.inaxes is self.gain_axis:
            node_index = self._find_tgc_node(event)
            if node_index is not None:
                self._delete_tgc_node(node_index)
            return
        if event.button != 1:
            return
        if getattr(event, "dblclick", False) and event.inaxes is self.gain_axis:
            if self._is_near_tgc_curve(event):
                self._add_tgc_node(float(event.xdata))
            return
        if event.inaxes is self.gain_axis:
            self.tgc_drag_index = self._find_tgc_node(event)
        if self.tgc_drag_index is not None:
            return
        if abs(event.xdata - self.gate_start_us) <= self.GATE_HANDLE_TOLERANCE_US:
            self.gate_drag_mode = "left"
        elif abs(event.xdata - self.gate_end_us) <= self.GATE_HANDLE_TOLERANCE_US:
            self.gate_drag_mode = "right"
        elif self.gate_start_us < event.xdata < self.gate_end_us:
            self.gate_drag_mode = "move"
            self.gate_drag_anchor_us = event.xdata

    def _on_mouse_motion(self, event) -> None:
        if event.xdata is None or event.inaxes not in (self.final_axis, self.gain_axis):
            return
        if self.tgc_drag_index is not None:
            index = self.tgc_drag_index
            if event.ydata is not None:
                self.tgc.move_node(index, event.xdata, event.ydata)
            self._recompute_pipeline()
            self._update_all_artists()
            return
        if self.gate_drag_mode is None:
            return
        x_us = float(np.clip(event.xdata, 0.0, self.time_limit_us))
        if self.gate_drag_mode == "left":
            self.gate_start_us = min(x_us, self.gate_end_us - self.GATE_MIN_WIDTH_US)
        elif self.gate_drag_mode == "right":
            self.gate_end_us = max(x_us, self.gate_start_us + self.GATE_MIN_WIDTH_US)
        else:
            width_us = self.gate_end_us - self.gate_start_us
            start_us = np.clip(self.gate_start_us + x_us - self.gate_drag_anchor_us, 0, self.time_limit_us - width_us)
            self.gate_start_us, self.gate_end_us = start_us, start_us + width_us
            self.gate_drag_anchor_us = x_us
        self._update_gate_artists()
        self._update_gate_peak()
        self.figure.canvas.draw_idle()

    def _on_mouse_release(self, _) -> None:
        self.gate_drag_mode = None
        self.tgc_drag_index = None
        if self.frequency_preview_pending:
            self._recompute_pipeline()
            self._update_all_artists()
            self.frequency_preview_pending = False

    def run(self) -> None:
        plt.show()


if __name__ == "__main__":
    AScanWorkbench().run()
