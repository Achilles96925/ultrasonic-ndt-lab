"""A 扫模拟数据生成。"""

from dataclasses import dataclass

import numpy as np

from ultrasonic_ndt.data.ascan import AScan


@dataclass(frozen=True)
class Echo:
    """一个模拟反射体的到达时间和回波幅值。"""

    time_us: float
    amplitude: float


def simulate_ascan(
    sampling_frequency_hz: float = 100e6,
    duration_us: float = 10.0,
    center_frequency_hz: float = 5e6,
    seed: int | None = None,
) -> AScan:
    """生成与探头中心频率相关的 12-bit 量化脉冲回波 A 扫。"""
    if not 0.5e6 <= center_frequency_hz <= 25e6:
        raise ValueError("探头中心频率必须位于 0.5 至 25 MHz 之间。")

    sample_count = int(round(duration_us * 1e-6 * sampling_frequency_hz))
    time_seconds = np.arange(sample_count) / sampling_frequency_hz
    frequency_mhz = center_frequency_hz / 1e6
    echoes = (
        Echo(time_us=1.4, amplitude=1760.0),
        Echo(time_us=4.2, amplitude=620.0),
        Echo(time_us=8.0, amplitude=1120.0),
    )

    waveform = np.zeros(sample_count, dtype=np.float64)
    # 在不同中心频率下保持近似相同的振荡周期数。
    pulse_width_us = np.clip(1.1 / frequency_mhz, 0.045, 1.8)
    for echo in echoes:
        offset_seconds = time_seconds - echo.time_us * 1e-6
        gaussian = np.exp(-0.5 * (offset_seconds / (pulse_width_us * 1e-6)) ** 2)
        carrier = np.sin(2.0 * np.pi * center_frequency_hz * offset_seconds)
        waveform += echo.amplitude * gaussian * carrier

    if seed is None:
        seed = 700 + int(round(frequency_mhz * 100))
    rng = np.random.default_rng(seed)
    noise_std_counts = 34.0 + 2.6 * frequency_mhz
    waveform += rng.normal(scale=noise_std_counts, size=sample_count)
    quantized = np.clip(np.rint(waveform), -2048, 2047).astype(np.int16)
    return AScan(samples=quantized, sampling_frequency_hz=sampling_frequency_hz)
