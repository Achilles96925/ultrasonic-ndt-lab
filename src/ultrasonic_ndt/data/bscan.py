"""B 扫数据模型。"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BScan:
    """沿一条直线采集的一组 RF A 扫。"""

    rf_samples: np.ndarray
    scan_positions_mm: np.ndarray
    sampling_frequency_hz: float
    sound_velocity_m_s: float

    def __post_init__(self) -> None:
        if self.rf_samples.ndim != 2:
            raise ValueError("B 扫 RF 数据必须是二维数组。")
        if self.scan_positions_mm.ndim != 1:
            raise ValueError("扫查位置必须是一维数组。")
        if self.rf_samples.shape[0] != self.scan_positions_mm.size:
            raise ValueError("A 扫数量必须与扫查位置数量一致。")
        if self.rf_samples.shape[1] < 2:
            raise ValueError("每条 A 扫至少需要两个采样点。")
        if self.sampling_frequency_hz <= 0 or self.sound_velocity_m_s <= 0:
            raise ValueError("采样率和声速必须大于零。")

    @property
    def time_seconds(self) -> np.ndarray:
        return np.arange(self.rf_samples.shape[1]) / self.sampling_frequency_hz

    @property
    def depth_mm(self) -> np.ndarray:
        """按直探头脉冲回波路径把飞行时间换算为深度。"""
        return self.time_seconds * self.sound_velocity_m_s * 1e3 / 2.0
