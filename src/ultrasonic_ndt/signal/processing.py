"""A 扫处理可复用的基础信号处理函数。"""

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt


def bandpass_filter(samples: np.ndarray, sampling_frequency_hz: float,
                    low_cut_hz: float, high_cut_hz: float, order: int = 4) -> np.ndarray:
    """应用零相位 Butterworth 带通滤波。"""
    nyquist_hz = sampling_frequency_hz / 2.0
    if not 0.0 < low_cut_hz < high_cut_hz < nyquist_hz:
        raise ValueError("Cutoff frequencies must lie between 0 and the Nyquist frequency.")
    sos = butter(order, [low_cut_hz / nyquist_hz, high_cut_hz / nyquist_hz],
                 btype="bandpass", output="sos")
    return sosfiltfilt(sos, samples)


def envelope(samples: np.ndarray) -> np.ndarray:
    """计算解析信号的幅值包络。"""
    return np.abs(hilbert(samples))


def time_gain_compensation(samples: np.ndarray, time_seconds: np.ndarray,
                           gain_db_per_microsecond: float) -> np.ndarray:
    """应用以 dB/μs 表示的线性时间增益补偿曲线。"""
    gain_linear = 10.0 ** ((gain_db_per_microsecond * time_seconds * 1e6) / 20.0)
    return samples * gain_linear


def apply_gain_curve(samples: np.ndarray, gain_db: np.ndarray) -> np.ndarray:
    """应用逐采样点的 dB 增益曲线。"""
    if samples.shape != gain_db.shape:
        raise ValueError("Samples and gain curve must have the same shape.")
    return samples * (10.0 ** (gain_db / 20.0))
