"""A 扫处理流水线。"""

from dataclasses import dataclass

import numpy as np

from ultrasonic_ndt.data.ascan import AScan
from ultrasonic_ndt.processing.tgc import TgcCurve
from ultrasonic_ndt.signal.processing import bandpass_filter, envelope


@dataclass(frozen=True)
class AScanProcessingResult:
    """A 扫滤波、包络和 TGC 处理后的中间结果。"""

    filtered: np.ndarray
    filtered_envelope: np.ndarray
    tgc_gain_db: np.ndarray
    compensated: np.ndarray
    compensated_envelope: np.ndarray


def process_ascan(ascan: AScan, low_cut_mhz: float, high_cut_mhz: float,
                  tgc_curve: TgcCurve, adc_min: float = -2048, adc_max: float = 2047) -> AScanProcessingResult:
    """按带通滤波、TGC 和包络的顺序处理一条 A 扫。"""
    filtered = bandpass_filter(
        ascan.samples.astype(float), ascan.sampling_frequency_hz, low_cut_mhz * 1e6, high_cut_mhz * 1e6
    )
    filtered_envelope = np.clip(envelope(filtered), 0, adc_max)
    time_us = ascan.time_seconds * 1e6
    compensated, tgc_gain_db = tgc_curve.apply(filtered, time_us)
    compensated = np.clip(compensated, adc_min, adc_max)
    compensated_envelope = np.clip(envelope(compensated), 0, adc_max)
    return AScanProcessingResult(
        filtered=filtered,
        filtered_envelope=filtered_envelope,
        tgc_gain_db=tgc_gain_db,
        compensated=compensated,
        compensated_envelope=compensated_envelope,
    )
