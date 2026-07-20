"""时间增益补偿曲线。"""

from dataclasses import dataclass

import numpy as np

from ultrasonic_ndt.signal.processing import apply_gain_curve


@dataclass
class TgcCurve:
    """由时间—增益控制点组成的分段线性 TGC 曲线。"""

    times_us: np.ndarray
    gains_db: np.ndarray

    def __post_init__(self) -> None:
        self.times_us = np.asarray(self.times_us, dtype=float)
        self.gains_db = np.asarray(self.gains_db, dtype=float)
        self._validate()

    def _validate(self) -> None:
        if self.times_us.ndim != 1 or self.gains_db.ndim != 1:
            raise ValueError("TGC 控制点必须是一维数组。")
        if self.times_us.size < 2 or self.times_us.size != self.gains_db.size:
            raise ValueError("TGC 至少需要两个时间与增益一一对应的控制点。")
        if np.any(np.diff(self.times_us) <= 0):
            raise ValueError("TGC 控制点时间必须严格递增。")

    def gain_at(self, time_us: np.ndarray) -> np.ndarray:
        """对给定时间轴插值得到每个采样点的 dB 增益。"""
        return np.interp(time_us, self.times_us, self.gains_db)

    def apply(self, samples: np.ndarray, time_us: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回补偿后的信号，以及对应的逐点 dB 增益。"""
        gain_db = self.gain_at(time_us)
        return apply_gain_curve(samples, gain_db), gain_db

    def move_node(self, index: int, time_us: float, gain_db: float) -> None:
        """移动控制点；首尾节点的时间保持为记录时窗边界。"""
        if not 0 <= index < self.times_us.size:
            raise IndexError("TGC 控制点索引超出范围。")
        if 0 < index < self.times_us.size - 1:
            self.times_us[index] = np.clip(time_us, self.times_us[index - 1] + 0.1, self.times_us[index + 1] - 0.1)
        self.gains_db[index] = np.clip(gain_db, -40.0, 40.0)

    def add_node(self, time_us: float) -> bool:
        """在曲线内部添加一个采用当前插值增益的新节点。"""
        insert_index = int(np.searchsorted(self.times_us, time_us))
        if insert_index == 0 or insert_index == self.times_us.size:
            return False
        if min(abs(time_us - self.times_us[insert_index - 1]), abs(self.times_us[insert_index] - time_us)) < 0.1:
            return False
        gain_db = float(np.interp(time_us, self.times_us, self.gains_db))
        self.times_us = np.insert(self.times_us, insert_index, time_us)
        self.gains_db = np.insert(self.gains_db, insert_index, gain_db)
        return True

    def delete_node(self, index: int) -> bool:
        """删除中间节点；首尾节点保留为时窗边界。"""
        if index == 0 or index == self.times_us.size - 1:
            return False
        self.times_us = np.delete(self.times_us, index)
        self.gains_db = np.delete(self.gains_db, index)
        return True

    def set_end_time(self, time_us: float) -> None:
        """将最后一个节点固定到新的记录终点。"""
        if time_us <= self.times_us[-2]:
            raise ValueError("记录终点必须大于倒数第二个 TGC 控制点。")
        self.times_us[-1] = time_us
