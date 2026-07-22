"""用于 B 扫演示的四阶厚度试块和 RF 数据生成。"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ultrasonic_ndt.data.bscan import BScan


STEP_THICKNESSES_MM = np.array([6.25, 12.5, 18.75, 25.0])


def thickness_profile_mm(scan_positions_mm: np.ndarray, scan_width_mm: float = 120.0) -> np.ndarray:
    """返回四个等宽区域对应的试块厚度。"""
    step_width_mm = scan_width_mm / STEP_THICKNESSES_MM.size
    indices = np.floor(np.asarray(scan_positions_mm) / step_width_mm).astype(int)
    return STEP_THICKNESSES_MM[np.clip(indices, 0, STEP_THICKNESSES_MM.size - 1)]


def _add_tone_burst(
    waveforms: np.ndarray,
    time_seconds: np.ndarray,
    arrival_seconds: np.ndarray,
    amplitude: np.ndarray,
    center_frequency_hz: float,
    pulse_cycles: float = 2.5,
) -> None:
    """向每条 A 扫加入一个有限时宽的高斯调制正弦回波。"""
    offset = time_seconds[None, :] - arrival_seconds[:, None]
    sigma_seconds = pulse_cycles / (2.355 * center_frequency_hz)
    pulse = np.exp(-0.5 * (offset / sigma_seconds) ** 2)
    pulse *= np.sin(2.0 * np.pi * center_frequency_hz * offset)
    waveforms += amplitude[:, None] * pulse


def simulate_bscan(
    scan_count: int = 481,
    scan_width_mm: float = 120.0,
    sampling_frequency_hz: float = 50e6,
    duration_us: float = 20.0,
    sound_velocity_m_s: float = 5900.0,
    center_frequency_hz: float = 5e6,
    seed: int = 2026,
) -> BScan:
    """模拟在四阶碳钢厚度校准试块上进行直线扫查。"""
    if scan_count < 2 or scan_width_mm <= 0 or duration_us <= 0:
        raise ValueError("扫查点数、宽度和记录时长必须大于零。")
    if not 0.5e6 <= center_frequency_hz < sampling_frequency_hz / 2.0:
        raise ValueError("中心频率必须位于 0.5 MHz 与奈奎斯特频率之间。")

    sample_count = int(round(duration_us * 1e-6 * sampling_frequency_hz))
    scan_positions_mm = np.linspace(0.0, scan_width_mm, scan_count)
    time_seconds = np.arange(sample_count) / sampling_frequency_hz
    thickness_mm = thickness_profile_mm(scan_positions_mm, scan_width_mm)
    rng = np.random.default_rng(seed)

    waveforms = np.zeros((scan_count, sample_count), dtype=np.float64)
    coupling = gaussian_filter1d(rng.normal(size=scan_count), sigma=18.0)
    coupling /= max(float(np.max(np.abs(coupling))), np.finfo(float).tiny)
    coupling = 1.0 + 0.08 * coupling

    # 台阶边缘附近声束部分脱离底面，底波幅值随距交界距离衰减，模拟边缘逐渐变弱。
    edge_fade = np.ones(scan_count)
    step_width_mm = scan_width_mm / STEP_THICKNESSES_MM.size
    for boundary_index in range(1, STEP_THICKNESSES_MM.size):
        boundary_mm = boundary_index * step_width_mm
        distance_mm = np.abs(scan_positions_mm - boundary_mm)
        edge_fade = np.minimum(edge_fade, 1.0 - np.exp(-0.5 * (distance_mm / 1.2) ** 2))

    # 始波保留在约 1.5 mm 位置，避开绘图边缘。
    entry_depth_mm = np.full(scan_count, 1.5)
    entry_arrival = 2.0 * entry_depth_mm * 1e-3 / sound_velocity_m_s
    _add_tone_burst(
        waveforms,
        time_seconds,
        entry_arrival,
        720.0 * coupling,
        center_frequency_hz,
        pulse_cycles=2.0,
    )

    backwall_arrival = 2.0 * thickness_mm * 1e-3 / sound_velocity_m_s
    backwall_amplitude = 1820.0 * np.exp(-0.012 * thickness_mm) * coupling * edge_fade
    _add_tone_burst(
        waveforms,
        time_seconds,
        backwall_arrival,
        backwall_amplitude,
        center_frequency_hz,
    )

    # 二次底面回波更弱，并随传播距离继续衰减。
    second_arrival = 4.0 * thickness_mm * 1e-3 / sound_velocity_m_s
    second_amplitude = 520.0 * np.exp(-0.024 * thickness_mm) * coupling * edge_fade
    _add_tone_burst(
        waveforms,
        time_seconds,
        second_arrival,
        second_amplitude,
        center_frequency_hz,
    )

    waveforms += rng.normal(scale=14.0, size=waveforms.shape)
    quantized = np.clip(np.rint(waveforms), -2048, 2047).astype(np.int16)
    return BScan(
        rf_samples=quantized,
        scan_positions_mm=scan_positions_mm,
        sampling_frequency_hz=sampling_frequency_hz,
        sound_velocity_m_s=sound_velocity_m_s,
    )
