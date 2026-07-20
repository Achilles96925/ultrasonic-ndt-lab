"""A 扫数据模型。"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AScan:
    """一条包含采样率信息的超声回波数据。"""

    samples: np.ndarray
    sampling_frequency_hz: float

    @property
    def time_seconds(self) -> np.ndarray:
        return np.arange(self.samples.size) / self.sampling_frequency_hz
