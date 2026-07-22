"""B 扫处理流水线。"""

from dataclasses import dataclass

import numpy as np

from ultrasonic_ndt.data.bscan import BScan
from ultrasonic_ndt.signal.processing import bandpass_filter, envelope


@dataclass(frozen=True)
class BScanProcessingResult:
    filtered_rf: np.ndarray
    envelope: np.ndarray
    display_amplitude: np.ndarray
    amplitude_db: np.ndarray


def process_bscan(
    bscan: BScan,
    low_cut_mhz: float = 2.0,
    high_cut_mhz: float = 8.0,
    dynamic_range_db: float = 42.0,
    noise_floor_db: float = -32.0,
    adc_max: float = 2047.0,
) -> BScanProcessingResult:
    """对全部 A 扫进行带通滤波、包络和对数压缩。"""
    if dynamic_range_db <= 0:
        raise ValueError("动态范围必须大于零。")
    filtered_rf = bandpass_filter(
        bscan.rf_samples.astype(float),
        bscan.sampling_frequency_hz,
        low_cut_mhz * 1e6,
        high_cut_mhz * 1e6,
    )
    envelope_data = np.clip(envelope(filtered_rf), 0.0, adc_max)
    reference = max(float(np.max(envelope_data)), np.finfo(float).tiny)
    amplitude_db = 20.0 * np.log10(np.maximum(envelope_data / reference, np.finfo(float).tiny))
    display_amplitude = np.clip((amplitude_db + dynamic_range_db) / dynamic_range_db, 0.0, 1.0)
    display_amplitude[amplitude_db < noise_floor_db] = 0.0
    return BScanProcessingResult(
        filtered_rf=filtered_rf,
        envelope=envelope_data,
        display_amplitude=display_amplitude,
        amplitude_db=amplitude_db,
    )
