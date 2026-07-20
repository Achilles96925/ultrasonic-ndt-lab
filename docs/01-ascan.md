# 超声成像算法实战——A扫信号处理链路

本系列主要讨论软件算法部分，所以暂时没有接入真实采集设备、探头和工件。示例先生成一条模拟A扫，再依次进行带通滤波、包络计算、时间增益补偿（TGC）和门控测量。

![A扫处理界面](https://cdn.jsdelivr.net/gh/Achilles96925/ultrasonic-ndt-lab@main/algorithms/01_ascan/ascan_pipeline.png)


## **A扫**

A扫是“幅值随时间变化”的一维信号。真实测量时，探头发出一束超声脉冲；声波遇到材料界面、缺陷或底面后会发生反射，探头接收到回波后转换为电信号。不同反射体的回波到达时间不同，因此会出现在A扫的不同位置。

当前示例不模拟声场传播、材料衰减和探头机电转换，而是把它们简化为几段在不同时间到达的回波包。每条回波由“高斯包络 × 正弦载波”表示：

* 正弦载波表示探头中心频率附近的RF振荡。真实探头通常发出持续数个周期的窄带脉冲，振荡频率集中在中心频率附近；
* 高斯包络使振荡在时间上逐渐衰减，起止过程较平滑。它近似描述了有限宽度的脉冲包络，也避免了突变波形带来的额外频谱成分；
* 回波的时间偏移对应传播延迟，幅值对应反射强弱。把多段回波相加，就得到包含多个反射体的简化A扫。

对应代码如下：

```
offset_seconds = time_seconds - echo.time_us * 1e-6
gaussian = np.exp(-0.5 * (offset_seconds / (pulse_width_us * 1e-6)) ** 2)
carrier = np.sin(2.0 * np.pi * center_frequency_hz * offset_seconds)
waveform += echo.amplitude * gaussian * carrier
```

本例的采样率为100MHz、记录时长为10μs，一条A扫包含1000个点。随后加入随机噪声，用来近似电子噪声和其他干扰：

```
sample_count = int(round(duration_us * 1e-6 * sampling_frequency_hz))
time_seconds = np.arange(sample_count) / sampling_frequency_hz
rng = np.random.default_rng(seed)
noise_std_counts = 34.0 + 2.6 * frequency_mhz
waveform += rng.normal(scale=noise_std_counts, size=sample_count)
```

最后，程序把浮点波形量化为12bit有符号整数：

```
quantized = np.clip(np.rint(waveform), -2048, 2047).astype(np.int16)
```

这里的12位是对采集卡ADC的简化模拟，表示ADC的量化分辨率，范围为 `[-2048,2047]`。RF波形会围绕零点正负摆动，符号反映的是电信号相位，不代表“负的回波能量”。

## **滤波**

滤波的目的是尽量保留与探头工作频带相符的成分，压低频带以外的能量，提高目标频带内的信噪比。

回波虽然围绕探头中心频率振荡，但短脉冲不可能只含一个频率；它会占据一段频带。实际检测时可能还会混入低频漂移、开关干扰和高频电子噪声。示例里面是模拟宽带随机噪声，所以带通滤波可以直观看到“保留回波频段、拒绝带外噪声”的取舍。

低截止频率主要排除较慢的变化和低频干扰，高截止频率排除高频噪声。截止频率通常应围绕探头中心频率，并结合其有效带宽设置：范围过宽，带外噪声会保留下来；范围过窄，则可能切掉回波本身的频谱，使脉冲在时间域变宽、波形失真，影响对幅值的判断。

带通滤波的核心实现如下：

```
def bandpass_filter(samples, sampling_frequency_hz, low_cut_hz, high_cut_hz, order=4):
    nyquist_hz = sampling_frequency_hz / 2.0
    sos = butter(
        order,
        [low_cut_hz / nyquist_hz, high_cut_hz / nyquist_hz],
        btype="bandpass",
        output="sos",
    )
    return sosfiltfilt(sos, samples)
```

## **包络**

带通滤波后的RF波形仍会围绕零点正负振荡。对于同一段回波，直接取某一个采样点的值会强烈依赖载波相位：采样点可能恰好落在正峰、负峰或过零点。因此，若目标是观察回波整体强弱或在时间窗内找峰值，先提取包络通常更稳定。

Hilbert变换把实信号构造成解析信号，包络就是该解析信号的幅值：

```
def envelope(samples):
    return np.abs(hilbert(samples))
```

包络去掉了RF振荡的正负方向，保留幅值随时间变化的外形。它不会创造新的回波，也不能分开本来就相互重叠的回波；当两个回波距离很近时，包络反而可能把它们合成一个较宽的峰。因此，当前界面把RF波形和包络同时保留：前者适合观察相位与波形细节，后者适合稳定显示和峰值测量。

```
filtered_envelope = np.clip(envelope(filtered), 0, adc_max)
```

## **TGC**

超声在材料中传播时会因吸收、散射和声束扩散逐渐衰减，较晚到达的回波往往比近表面的回波弱。TGC的作用是按时间提高增益，让不同深度的回波尽量落在相近、可观察的显示范围内。

这不等于TGC能提高信噪比或恢复真实反射系数：同一时间位置的回波和噪声会一起被放大。它主要用于动态范围管理和显示补偿，实际曲线需要结合材料、路径和标定结果设置。

紫色虚线表示各个时间点对应的TGC增益，单位是dB值。示例中没有根据衰减模型自动求曲线，也没有实际检测中的校准功能，直接用可拖动的点来直观展示TGC曲线的作用：

```
def gain_at(self, time_us):
    return np.interp(time_us, self.times_us, self.gains_db)

def apply(self, samples, time_us):
    gain_db = self.gain_at(time_us)
    return apply_gain_curve(samples, gain_db), gain_db
```

dB是对数单位，应用到波形前需要转换为线性幅值增益：

```
def apply_gain_curve(samples, gain_db):
    return samples * (10.0 ** (gain_db / 20.0))
```

对应的关系是：

```
线性幅值增益 = 10^(GdB / 20)
```

补偿结果会再次限制在ADC的可表示范围内：

```
compensated, tgc_gain_db = tgc_curve.apply(filtered, time_us)
compensated = np.clip(compensated, adc_min, adc_max)
```

如果出现大段贴近上下限的波形时，就说明当前参数已经产生了数值裁剪饱和。这时峰值幅值已经失真，不能再据此比较反射强弱。

## **闸门**

一条A扫常常包含初始脉冲、近表面回波、目标回波和底面回波。若在整条波形中直接寻找最大值，最强的无关回波或噪声可能会遮蔽目标。因此，门控先限定一个与目标深度区间对应的时间窗，再从这个时间窗内提取特征。

绿色区域表示时间闸门。拖动闸门中间部分可以移动整个时间范围，拖动两侧边缘可以调整起止位置。闸门的起止时间本身需要根据预期传播路径和材料声速选择；在当前界面中，它只是可交互的算法参数。

峰值搜索只在闸门内部进行：

* 显示包络时，寻找包络的最大值；
* 显示RF波形时，寻找绝对值最大的采样点。

门内搜索的核心代码如下：

```
in_gate = (time_us >= start_us) & (time_us <= end_us)
measurement = np.abs(samples) if use_absolute_value else samples
peak_index = int(np.argmax(np.where(in_gate, measurement, -np.inf)))

return GatePeak(
    time_us=float(time_us[peak_index]),
    amplitude=float(measurement[peak_index]),
)
```

界面调用时，根据是否显示包络决定是否取绝对值：

```
peak = find_gate_peak(
    time_us,
    measurement,
    gate_start_us,
    gate_end_us,
    use_absolute_value=not envelope_enabled,
)
```

显示包络时，峰值代表包络最大值；显示RF波形时，代码取绝对值最大的采样点，避免正负相位使负峰被忽略。这个峰值只是当前实现选用的一种特征，实际检测还可能使用阈值、能量、首达波或多峰规则。

TOF是回波从发射到接收所经历的时间。对于已知材料声速、探头延迟和传播路径的直探头模型，深度可以近似写成：

```
深度 = 声速 × TOF / 2
```

除以2是因为测到的是发射到反射体再返回探头的往返时间。当前代码没有真实工件几何、材料声速、探头延迟和标定参数，这里只显示TOF，不直接把它解释为深度。

## **代码**

示例代码已上传GitHub：https://github.com/Achilles96925/ultrasonic-ndt-lab

下一篇会继续介绍沿一个方向连续生成多条A扫，排列成一幅B扫图像的过程。
