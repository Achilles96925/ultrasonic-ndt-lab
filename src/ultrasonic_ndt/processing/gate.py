"""软件闸门与门内特征提取。"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GatePeak:
    """门内峰值的时间和幅值。"""

    time_us: float
    amplitude: float


def find_gate_peak(time_us: np.ndarray, samples: np.ndarray, start_us: float, end_us: float,
                   use_absolute_value: bool = False) -> GatePeak | None:
    """在指定时间门内寻找最大峰值。"""
    if start_us >= end_us:
        raise ValueError("闸门起点必须早于终点。")
    in_gate = (time_us >= start_us) & (time_us <= end_us)
    if not np.any(in_gate):
        return None
    measurement = np.abs(samples) if use_absolute_value else samples
    peak_index = int(np.argmax(np.where(in_gate, measurement, -np.inf)))
    return GatePeak(time_us=float(time_us[peak_index]), amplitude=float(measurement[peak_index]))
