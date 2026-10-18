"""不可变的采集与在线解码配置。

配置是边界对象：CLI、LSL 传输、预处理和 GUI 都通过它共享同一输入契约。
"""

from dataclasses import asdict, dataclass, fields
from math import isfinite
from numbers import Integral, Real
import math

from .constants import N_CLASSES


# ========================================================================
# 01. 参数配置与输入校验
# ========================================================================
@dataclass(frozen=True)
class Config:
    # 设备信息、滤波参数、窗口时序和质量门限集中保存在不可变配置中。
    sfreq: float  # 实际采样率，单位 Hz
    channels: tuple[str, ...]  # 模型使用的电极名称及固定顺序
    reference: str  # 实际参考电极说明；此字段不执行重参考
    unit: str = "V"  # 核心处理流程固定使用 V
    low_hz: float = 8.0  # 带通滤波下限
    high_hz: float = 30.0  # 带通滤波上限，必须低于奈奎斯特频率
    filter_order: int = 4
    warmup_seconds: float = 1.0  # 滤波状态重置后暂不用于推断的时长
    window_seconds: float = 3.0  # 250 Hz 下每窗 750 点
    step_seconds: float = 0.2  # 相邻滑动窗口的起点间隔
    cue_offset_seconds: float = 0.5  # 相对 MI_START 延后多久开始第一个窗口
    imagery_seconds: float = 5.0  # 给 3 秒窗、连续确认和 LSL 传输留出时间
    buffer_seconds: float = 10.0  # 环形缓冲保留的数据长度
    chunk_samples: int = 25  # 每次读取的数据包采样点数上限
    n_components: int = 4  # 仅用于加载旧 CSP 模型；当前三种模型入口不使用此参数
    threshold: float = 0.70  # 单个窗口的方向概率门限
    confirmations: int = 3  # 输出方向前所需的连续同方向窗口数
    max_abs_volts: float = 500e-6  # 单样本绝对幅度上限
    flat_std_volts: float = 1e-9  # 识别平直信号的标准差门限
    max_ac_rms_volts: float = 100e-6
    flat_window_seconds: float = 1.0
    line_frequency_hz: float = 50.0
    max_line_power_fraction: float = 0.35
    high_frequency_hz: float = 40.0
    max_high_frequency_power_fraction: float = 0.65
    saturation_run_seconds: float = 0.02
    max_age_seconds: float = 0.5  # 数据或结果允许的最大年龄
    timestamp_tolerance_seconds: float = 0.002  # 采样间隔及部分时钟检查允许的偏差
    lsl_channel_labels: tuple[str, ...] | None = None  # 按实际传输顺序声明的电极映射
    lsl_input_units: str | None = None  # LSL 输入单位，用于换算到 V
    lsl_marker_grace_seconds: float = 0.10  # 短暂保留 EEG，等待独立传输的阶段标记

    def __post_init__(self):
        # 构造时统一检查类型、范围与参数关系，防止错误配置进入采集或模型。
        for parameter in fields(self):
            if parameter.type in (int, float):
                value = getattr(self, parameter.name)
                if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
                    raise ValueError(f"{parameter.name} must be a finite number")
                if parameter.type is int and not isinstance(value, Integral):
                    raise ValueError(f"{parameter.name} must be an integer")
        if isinstance(self.channels, str):
            raise ValueError("Channels must be a sequence of electrode names")
        object.__setattr__(self, "channels", tuple(self.channels))
        if self.lsl_channel_labels is not None:
            if isinstance(self.lsl_channel_labels, str):
                raise ValueError("LSL channel mapping must be a sequence in transmitted channel order")
            object.__setattr__(self, "lsl_channel_labels", tuple(self.lsl_channel_labels))
            if (any(not isinstance(c, str) or not c.strip() for c in self.lsl_channel_labels)
                    or len(set(self.lsl_channel_labels)) != len(self.lsl_channel_labels)
                    or not set(self.channels).issubset(self.lsl_channel_labels)):
                raise ValueError("LSL mapping must contain unique names for every configured electrode")
        if any(not isinstance(channel, str) or not channel.strip() for channel in self.channels):
            raise ValueError("Channel names must be nonempty strings")
        if not isinstance(self.reference, str) or not self.reference.strip():
            raise ValueError("Reference must be a nonempty string")
        if not (0 < self.low_hz < self.high_hz < self.sfreq / 2):
            raise ValueError("Bandpass must be below Nyquist")
        if not self.channels or len(set(self.channels)) != len(self.channels):
            raise ValueError("Channel names must be unique")
        if self.unit != "V":
            raise ValueError("Core pipeline uses volts")
        if self.lsl_input_units not in (None, "V", "mV", "uV", "µV", "μV"):
            raise ValueError("LSL input units must be verified V, mV or uV")
        if not (1 / N_CLASSES <= self.threshold <= 1) or self.confirmations < 1:
            raise ValueError("Invalid decision policy")
        if self.n_components < 1 or self.n_components > len(self.channels):
            raise ValueError("Invalid CSP component count")
        if self.chunk_samples < 1 or self.filter_order < 1:
            raise ValueError("Chunk size and filter order must be positive")
        if self.window_seconds <= 0 or self.step_seconds <= 0 or self.warmup_seconds < 0:
            raise ValueError("Invalid timing")
        if self.cue_offset_seconds < 0 or self.cue_offset_seconds + self.window_seconds > self.imagery_seconds:
            raise ValueError("Window must fit in imagery period")
        available_windows = 1 + math.floor((self.imagery_seconds - self.cue_offset_seconds
                                           - self.window_seconds) / self.step_seconds + 1e-8)
        if self.confirmations > available_windows:
            raise ValueError("Confirmation count exceeds the number of windows in the imagery period")
        if self.buffer_seconds < self.window_seconds:
            raise ValueError("Buffer must hold a complete window")
        if self.max_abs_volts <= 0 or self.flat_std_volts < 0 or self.max_age_seconds <= 0 or self.timestamp_tolerance_seconds < 0:
            raise ValueError("Invalid quality limits")
        if min(self.max_ac_rms_volts, self.flat_window_seconds, self.line_frequency_hz,
               self.high_frequency_hz, self.saturation_run_seconds) <= 0:
            raise ValueError("Channel quality limits must be positive")
        if not (0 < self.max_line_power_fraction < 1 and 0 < self.max_high_frequency_power_fraction < 1):
            raise ValueError("Noise power fractions must be between zero and one")
        if not 0 <= self.lsl_marker_grace_seconds < self.max_age_seconds:
            raise ValueError("LSL marker grace must be nonnegative and below the maximum data age")
        # 时长必须精确对应整数采样点，保证离线和在线使用相同的窗口边界。
        for duration in (self.window_seconds, self.step_seconds, self.cue_offset_seconds, self.imagery_seconds, self.warmup_seconds, self.buffer_seconds):
            if abs(duration * self.sfreq - round(duration * self.sfreq)) > 1e-8:
                raise ValueError("Timing must map to integer samples")

    def samples(self, seconds: float) -> int:
        return round(seconds * self.sfreq)

    @property
    def window_samples(self):
        return self.samples(self.window_seconds)

    @property
    def step_samples(self):
        return self.samples(self.step_seconds)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        if not {"sfreq", "channels", "reference"}.issubset(values):
            raise ValueError("Configuration requires the actual device sample rate, channels and reference; use configure-lsl")
        if "channels" in values:
            values["channels"] = tuple(values["channels"])
        return cls(**values)
