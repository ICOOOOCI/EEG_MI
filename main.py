# 真实 EEG 运动想象命令行系统：采集、校准训练、连续回放与在线解码。
# 主要流程：inspect-lsl → configure-lsl → preflight → experiment + record
#           → import-session → train → experiment + live → evaluate-live。
# experiment 提供终端提示；阶段标记经 LSL 传输，左右手真值单独保存到标签日志。
# 程序内部统一使用伏特（V），EEG 数组按“通道 × 采样点”排列。
# T 会话用于校准训练，E 会话用于独立测试；在线解码器只接收不含类别的阶段标记。
# 运行 python main.py 查看帮助；各子命令的参数可用 --help 查看。
#
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version as package_version
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator, Protocol
from uuid import uuid4
import argparse
import json
import math
import os
import platform
import queue
import random
import re
import sys
import threading
import time
import uuid
import warnings

import joblib
import mne
import numpy as np
from mne.decoding import CSP
from scipy.signal import butter, sosfilt, welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline


__version__ = "0.3.0"



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
    window_seconds: float = 2.0  # 每次模型推断使用的 EEG 窗口长度
    step_seconds: float = 0.2  # 相邻滑动窗口的起点间隔
    cue_offset_seconds: float = 0.5  # 相对 MI_START 延后多久开始第一个窗口
    imagery_seconds: float = 4.0  # 每个试次的运动想象阶段长度
    buffer_seconds: float = 10.0  # 环形缓冲保留的数据长度
    chunk_samples: int = 25  # 每次读取的数据包采样点数上限
    n_components: int = 4  # CSP 输出特征的分量数
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
        if not (0.5 <= self.threshold <= 1) or self.confirmations < 1:
            raise ValueError("Invalid decision policy")
        if self.n_components < 1 or self.n_components > len(self.channels):
            raise ValueError("Invalid CSP component count")
        if self.chunk_samples < 1 or self.filter_order < 1:
            raise ValueError("Chunk size and filter order must be positive")
        if self.window_seconds <= 0 or self.step_seconds <= 0 or self.warmup_seconds < 0:
            raise ValueError("Invalid timing")
        if self.cue_offset_seconds < 0 or self.cue_offset_seconds + self.window_seconds > self.imagery_seconds:
            raise ValueError("Window must fit in imagery period")
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



# ========================================================================
# 02. 数据结构与模型接口
# ========================================================================
@dataclass(frozen=True)
class StreamInfo:
    # 描述进入核心处理流程后的数据约定，通道顺序必须与模型一致。
    sfreq: float
    channels: tuple[str, ...]
    reference: str
    unit: str = "V"


@dataclass(frozen=True)
class EEGChunk:
    # 一批连续 EEG 样本；每个采样点对应一个时间戳。
    data: np.ndarray  # 形状为 (通道数, 采样点数)，单位 V
    timestamps: np.ndarray
    start_sample: int  # 当前数据包首个样本的全局序号
    info: StreamInfo
    segment_id: int = 0  # 连续采集段标识，用于识别重启边界
    received_at: float | None = None  # 在线数据与 timestamps 使用同一接收端时钟


@dataclass(frozen=True)
class Marker:
    # 只描述实验阶段；类别真值通过独立日志按 trial_id 关联。
    kind: str
    sample: int  # 标记在接收端 EEG 采样网格上的位置
    trial_id: str | None = None
    timestamp: float | None = None
    segment_id: int = 0
    received_at: float | None = None
    alignment_seconds: float | None = None  # 标记后首个 EEG 样本与标记的时间差


@dataclass(frozen=True)
class Prediction:
    # 一个滑动窗口的预测记录；质量不合格时概率可为空。
    trial_id: str
    start_sample: int
    stop_sample: int  # 右开区间终点，窗口不包含此采样点
    probabilities: tuple[float, float] | None  # 列顺序固定为 LEFT、RIGHT
    timestamp: float  # 有效窗口最后一个已观测样本的采集时刻
    model_version: str
    quality: str = "OK"
    processing_seconds: float = 0.0
    bad_channels: tuple[str, ...] = ()
    channel_quality_reasons: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Command:
    # 判决结果及其时延信息；NO_COMMAND 表示不输出方向指令。
    trial_id: str | None
    direction: str  # LEFT、RIGHT 或 NO_COMMAND
    sample: int
    timestamp: float
    reason: str
    processing_seconds: float = 0.0
    emitted_at: float | None = None  # 在线发布时的 LSL 时刻；离线判决可为空
    cue_to_output_seconds: float | None = None
    result_age_seconds: float | None = None


class WindowModel(Protocol):
    # 模型输入为“窗口数 × 通道数 × 采样点数”，输出两列左右手概率。
    def predict_proba(self, windows: np.ndarray) -> np.ndarray: ...



# ========================================================================
# 03. 统一采集接口
# ========================================================================
class Source(Protocol):
    # 真实采集与离线回放均产生 EEGChunk 或 Marker，供同一 Runtime 处理。
    info: StreamInfo

    def __iter__(self) -> Iterator[EEGChunk | Marker]: ...

    def close(self) -> None: ...




# ========================================================================
# 04. 个人采集数据与存储
# ========================================================================
@dataclass(frozen=True)
class Trial:
    # 一个独立试次的真值与定位信息，用于训练和评分。
    subject_id: str
    session_id: str
    run_id: int
    trial_id: str
    cue_sample: int
    original_class: int  # 独立真值日志：1=LEFT，2=RIGHT；训练时减 1
    artifact_flag: bool
    start_sample: int | None = None


@dataclass
class Recording:
    # 完整录制及其试次表；只接受从个人 LSL 采集导入的 T/E 会话。
    data: np.ndarray
    timestamps: np.ndarray
    info: StreamInfo
    trials: list[Trial]
    run_starts: list[int]
    subject_id: str
    session_id: str  # T 为校准训练会话，E 为独立测试会话
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.session_id, str) or self.session_id not in {"T", "E"}:
            raise ValueError("Personal recording session_id must be T or E")
        if self.metadata.get("synthetic"):
            raise ValueError("Only recorded EEG is supported; legacy synthetic recordings are not accepted")
        if self.metadata.get("source") != "personal_lsl":
            raise ValueError("Only personal LSL recordings are supported; import a completed capture with import-session")
        self.data = np.asarray(self.data, dtype=np.float64)
        self.timestamps = np.asarray(self.timestamps, dtype=np.float64)
        if self.data.ndim != 2 or self.data.shape[0] != len(self.info.channels):
            raise ValueError("Recording must be channels x samples in declared channel order")
        if self.timestamps.shape != (self.data.shape[1],):
            raise ValueError("Timestamp count must equal sample count")
        if not np.isfinite(self.info.sfreq) or self.info.sfreq <= 0 or self.info.unit != "V":
            raise ValueError("Recording requires a positive sample rate and volts")
        if not np.all(np.isfinite(self.timestamps)) or np.any(np.diff(self.timestamps) <= 0):
            raise ValueError("Recording timestamps must be finite and strictly increasing")
        if len({t.trial_id for t in self.trials}) != len(self.trials):
            raise ValueError("Trial identifiers must be unique")
        if any(isinstance(t.original_class, bool) or not isinstance(t.original_class, Integral)
               or t.original_class not in (1, 2) for t in self.trials):
            raise ValueError("Personal truth labels must be 1=LEFT or 2=RIGHT")
        if any(t.subject_id != self.subject_id or t.session_id != self.session_id for t in self.trials):
            raise ValueError("Trial subject/session differs from recording")
        if any(not 0 <= t.cue_sample < self.data.shape[1] for t in self.trials):
            raise ValueError("Cue lies outside the recording")
        if any(t.start_sample is not None and not 0 <= t.start_sample <= t.cue_sample for t in self.trials):
            raise ValueError("Trial start must precede its cue")
        if any(a.cue_sample >= b.cue_sample for a, b in zip(self.trials, self.trials[1:])):
            raise ValueError("Trials must be in chronological order")
        if self.run_starts != sorted(set(self.run_starts)) or any(not 0 <= x < self.data.shape[1] for x in self.run_starts):
            raise ValueError("Run starts must be unique, sorted, and within the recording")


def save_recording(path, recording: Recording) -> None:
    # 将连续数据、时间戳和 JSON 元数据保存为便于训练及回放的 NPZ。
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1, "info": asdict(recording.info),
        "trials": [asdict(trial) for trial in recording.trials],
        "run_starts": recording.run_starts, "subject_id": recording.subject_id,
        "session_id": recording.session_id, "metadata": recording.metadata,
    }
    arrays = {"data": recording.data, "timestamps": recording.timestamps,
              "metadata_json": np.asarray(json.dumps(payload, ensure_ascii=False, allow_nan=False))}
    with path.open("wb") as file:
        np.savez_compressed(file, **arrays)


def load_recording(path) -> Recording:
    # 读取 NPZ 后重新构造 Recording，使导入数据再次经过结构校验。
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("Expected a personal recording NPZ created by import-session")
    with np.load(path, allow_pickle=False) as arrays:
        payload = json.loads(str(arrays["metadata_json"].item()))
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported recording format version")
        info = dict(payload["info"])
        info["channels"] = tuple(info["channels"])
        return Recording(
            data=arrays["data"], timestamps=arrays["timestamps"], info=StreamInfo(**info),
            trials=[Trial(**trial) for trial in payload["trials"]],
            run_starts=payload["run_starts"], subject_id=payload["subject_id"],
            session_id=payload["session_id"],
            metadata=payload.get("metadata", {}),
        )



# ========================================================================
# 05. 因果预处理：只使用当前及过去的样本
# ========================================================================
@dataclass(frozen=True)
class FilteredChunk:
    # 滤波结果同时保留有效性、连续段边界和逐样本拒绝原因。
    data: np.ndarray
    timestamps: np.ndarray
    sample_indices: np.ndarray
    valid: np.ndarray
    breaks: np.ndarray  # 当前位置是否开始新的滤波连续段
    reasons: np.ndarray


class CausalPreprocessor:
    # 跨数据包保留滤波状态；断点或坏样本会触发重置及预热。
    def __init__(self, config: Config, sos=None):
        self.config = config
        self.sos = np.array(sos if sos is not None else butter(
            config.filter_order, [config.low_hz, config.high_hz],
            btype="bandpass", fs=config.sfreq, output="sos"), dtype=float, copy=True)
        if self.sos.ndim != 2 or self.sos.shape[1] != 6 or not np.isfinite(self.sos).all():
            raise ValueError("SOS must have shape (sections, 6) and finite coefficients")
        if not np.all(self.sos[:, 3] == 1):
            raise ValueError("SOS denominator leading coefficients must equal 1")
        self.reset()

    def reset(self):
        # 二阶节状态形状为 (滤波节数, 通道数, 2)，每段从零状态开始。
        self._zi = np.zeros((len(self.sos), len(self.config.channels), 2))
        self._warmup = self.config.samples(self.config.warmup_seconds)
        self._next_sample = None
        self._last_timestamp = None
        self._segment_id = None
        self._restart = True

    def _reset_filter(self):
        self._zi.fill(0)
        self._warmup = self.config.samples(self.config.warmup_seconds)

    def process(self, chunk: EEGChunk) -> FilteredChunk:
        cfg = self.config
        info = chunk.info
        if (info.sfreq != cfg.sfreq or tuple(info.channels) != cfg.channels
                or info.unit != cfg.unit or info.reference != cfg.reference):
            raise ValueError("EEG stream metadata does not match frozen model configuration")
        data = np.asarray(chunk.data, dtype=float)
        timestamps = np.asarray(chunk.timestamps, dtype=float)
        if data.ndim != 2 or data.shape[0] != len(cfg.channels):
            raise ValueError("EEG data must have shape (configured channels, samples)")
        if timestamps.shape != (data.shape[1],):
            raise ValueError("One timestamp is required per EEG sample")
        if not isinstance(chunk.start_sample, (int, np.integer)) or chunk.start_sample < 0:
            raise ValueError("start_sample must be a nonnegative integer")
        n = data.shape[1]
        indices = np.arange(chunk.start_sample, chunk.start_sample + n, dtype=np.int64)
        valid = np.zeros(n, dtype=bool)
        breaks = np.zeros(n, dtype=bool)
        reasons = np.full(n, "OK", dtype="U20")
        filtered = np.full_like(data, np.nan)
        if not n:
            return FilteredChunk(filtered, timestamps.copy(), indices, valid, breaks, reasons)
        # 同时检查序号与时间间隔；丢包后即使序号连续，时间戳仍可能暴露断点。
        same_segment = self._segment_id == chunk.segment_id
        if same_segment and self._next_sample is not None and chunk.start_sample < self._next_sample:
            raise ValueError("Overlapping or out-of-order samples would be processed twice")
        breaks[0] = (self._restart or not same_segment or self._next_sample != chunk.start_sample)
        expected_dt = 1.0 / cfg.sfreq
        if self._last_timestamp is not None and same_segment:
            breaks[0] |= not np.isfinite(timestamps[0]) or timestamps[0] <= self._last_timestamp or abs(
                timestamps[0] - self._last_timestamp - expected_dt) > cfg.timestamp_tolerance_seconds
        if n > 1:
            delta = np.diff(timestamps)
            breaks[1:] |= ~np.isfinite(delta) | (delta <= 0) | (np.abs(delta - expected_dt) > cfg.timestamp_tolerance_seconds)
        good_signal = (np.isfinite(data).all(axis=0) & np.isfinite(timestamps)
                       & (np.max(np.abs(data), axis=0) <= cfg.max_abs_volts))
        breaks |= ~good_signal
        if n > 1:
            breaks[1:] |= ~good_signal[:-1]
        reasons[~good_signal] = "BAD_SIGNAL"
        stale = np.zeros(n, dtype=bool)
        if chunk.received_at is not None:
            if not np.isfinite(chunk.received_at):
                raise ValueError("received_at must be finite when supplied")
            stale = (chunk.received_at - timestamps) > cfg.max_age_seconds
            # 采集时间若明显晚于接收时间，说明时钟不一致，同样拒绝该样本。
            stale |= (timestamps - chunk.received_at) > cfg.timestamp_tolerance_seconds
            reasons[stale & good_signal] = "STALE_DATA"
        # 只在连续且有效的区间内延续滤波状态，坏样本不会污染后续滤波。
        starts = np.unique(np.r_[0, np.flatnonzero(breaks), np.flatnonzero(~good_signal), n])
        for left, right in zip(starts[:-1], starts[1:]):
            if breaks[left]:
                self._reset_filter()
            if not good_signal[left]:
                self._reset_filter()
                continue
            filtered[:, left:right], self._zi = sosfilt(
                self.sos, data[:, left:right], axis=-1, zi=self._zi)
            # 预热样本参与推进滤波状态，但不允许进入有效模型窗口。
            warm = min(self._warmup, right - left)
            reasons[left:left + warm] = np.where(stale[left:left + warm], "STALE_DATA", "WARMUP")
            self._warmup -= warm
            valid[left + warm:right] = ~stale[left + warm:right]
        self._segment_id = chunk.segment_id
        self._next_sample = chunk.start_sample + n
        self._last_timestamp = float(timestamps[-1]) if np.isfinite(timestamps[-1]) else None
        self._restart = not good_signal[-1]
        return FilteredChunk(filtered, timestamps.copy(), indices, valid, breaks, reasons)



# ========================================================================
# 06. 环形缓冲与连续窗口读取
# ========================================================================
@dataclass(frozen=True)
class BufferedWindow:
    # 窗口快照同时携带原始信号与滤波信号，供质量检查和模型推断使用。
    data: np.ndarray
    raw_data: np.ndarray
    timestamps: np.ndarray
    valid: np.ndarray
    epochs: np.ndarray  # 连续段编号；跨段窗口不得用于推断
    reasons: np.ndarray


class RingBuffer:
    # 固定容量缓存按采样序号寻址，避免运行时间增长导致缓存无限扩张。
    def __init__(self, channels: int, capacity: int):
        if channels < 1 or capacity < 1:
            raise ValueError("Ring dimensions must be positive")
        self.capacity = capacity
        self.data = np.empty((channels, capacity), dtype=float)
        self.raw_data = np.empty((channels, capacity), dtype=float)
        self.timestamps = np.empty(capacity, dtype=float)
        self.indices = np.empty(capacity, dtype=np.int64)
        self.valid = np.empty(capacity, dtype=bool)
        self.epochs = np.empty(capacity, dtype=np.int64)
        self.reasons = np.empty(capacity, dtype="U20")
        self.clear()

    def clear(self):
        self._position = 0
        self.size = 0

    def append_one(self, data, timestamp, sample, valid, epoch=0, reason="OK", raw_data=None):
        pos = self._position
        self.data[:, pos] = data
        self.raw_data[:, pos] = data if raw_data is None else raw_data
        self.timestamps[pos] = timestamp
        self.indices[pos] = sample
        self.valid[pos] = valid
        self.epochs[pos] = epoch
        self.reasons[pos] = reason
        self._position = (pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def read(self, start: int, stop: int) -> BufferedWindow | None:
        # 读取 [start, stop)；数据不足、已被覆盖或序号不连续时返回 None。
        count = stop - start
        if count <= 0 or count > self.size:
            return None
        latest_position = (self._position - 1) % self.capacity
        offset = int(self.indices[latest_position]) - (stop - 1)
        if offset < 0 or offset + count > self.size:
            return None
        positions = (np.arange(self._position - offset - count,
                               self._position - offset) % self.capacity)
        # 模运算只能找出缓存位置，还须核对采样序号，防止把断点两侧的数据拼接。
        if not np.array_equal(self.indices[positions], np.arange(start, stop)):
            return None
        return BufferedWindow(self.data[:, positions].copy(), self.raw_data[:, positions].copy(), self.timestamps[positions].copy(),
                              self.valid[positions].copy(), self.epochs[positions].copy(),
                              self.reasons[positions].copy())



# ========================================================================
# 07. 概率阈值、连续确认与单试次判决
# ========================================================================
class DecisionPolicy:
    # 同一方向连续达到置信度门限后确认；每个 trial_id 最多输出一次方向。
    def __init__(self, config: Config):
        self.config = config
        self._committed_trials = set()
        self.trial_id = None
        self.reset(None)

    @property
    def committed(self):
        return self.trial_id in self._committed_trials

    def reset(self, trial_id):
        self.trial_id = trial_id
        self._last_stop = None
        self.clear_streak()

    def clear_streak(self):
        # 清除尚未确认的连续证据，但不解除已经输出过指令的试次锁定。
        self._direction = None
        self._streak = 0

    def end(self):
        self.clear_streak()
        self.trial_id = None

    def decide(self, prediction: Prediction) -> Command:
        def output(direction, reason):
            return Command(prediction.trial_id, direction, prediction.stop_sample,
                           prediction.timestamp, reason, prediction.processing_seconds)

        if self.trial_id is None or prediction.trial_id != self.trial_id:
            self.clear_streak()
            return output("NO_COMMAND", "OUTSIDE_MI_PHASE")
        if self._last_stop is not None and prediction.stop_sample <= self._last_stop:
            self.clear_streak()
            return output("NO_COMMAND", "STALE_DATA")
        # 窗口缺失会打断连续确认，不能将时间上不相邻的证据累加。
        if self._last_stop is not None and prediction.stop_sample != self._last_stop + self.config.step_samples:
            self.clear_streak()
        self._last_stop = prediction.stop_sample
        if prediction.quality != "OK":
            self.clear_streak()
            return output("NO_COMMAND", prediction.quality)
        if self.committed:
            return output("NO_COMMAND", "ALREADY_COMMITTED")
        p = np.asarray(prediction.probabilities, dtype=float)
        if p.shape != (2,) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1) or not np.isclose(p.sum(), 1):
            self.clear_streak()
            return output("NO_COMMAND", "MODEL_ERROR")
        label = int(np.argmax(p))
        if p[label] < self.config.threshold or p[0] == p[1]:
            self.clear_streak()
            return output("NO_COMMAND", "LOW_CONFIDENCE")
        if self._direction == label:
            self._streak += 1
        else:
            self._direction, self._streak = label, 1
        if self._streak < self.config.confirmations:
            return output("NO_COMMAND", "AWAITING_CONFIRMATION")
        self._committed_trials.add(self.trial_id)
        return output(("LEFT", "RIGHT")[label], "CONFIRMED")

    process = decide



NEUTRAL_MARKERS = frozenset({"RUN_START", "TRIAL_START", "MI_START", "MI_END", "DISCONNECT", "SESSION_END"})


# ========================================================================
# 08. 流式解码与实验阶段状态机
# ========================================================================
def iter_window_starts(config: Config, cue_sample: int):
    # 只生成完整落在想象阶段内的窗口起点，终点采用右开区间。
    first = cue_sample + config.samples(config.cue_offset_seconds)
    last = cue_sample + config.samples(config.imagery_seconds) - config.window_samples
    return range(first, last + 1, config.step_samples)


class Runtime:
    # 将阶段标记、因果滤波、窗口质量检查、模型推断和指令判决串联起来。
    def __init__(self, config: Config, model, sos=None, model_version="unversioned", window_observer=None,
                 validity_guard=None):
        self.config, self.model, self.model_version = config, model, model_version
        self.preprocessor = CausalPreprocessor(config, sos)
        self.buffer = RingBuffer(len(config.channels), config.samples(config.buffer_seconds))
        self.decision = DecisionPolicy(config)
        self.window_observer = window_observer
        self.validity_guard = validity_guard
        self.quality_limits = QualityThresholds.for_window(config)
        self.trial_id = None
        self._active = False
        self._starts = []
        self._next_window = 0
        self._cue = None
        self._cue_timestamp = None
        self._epoch = 0
        self._last_sample = None
        self._last_timestamp = None
        self.stats = {"samples_processed": 0, "continuity_resets": 0,
                      "bad_signal_samples": 0, "stale_samples": 0,
                      "invalid_windows": 0, "model_errors": 0}
        self.stats["channel_rejected_windows"] = {name: 0 for name in config.channels}

    def process(self, event: EEGChunk | Marker) -> list[Prediction | Command]:
        if isinstance(event, Marker):
            return self._marker(event)
        if not isinstance(event, EEGChunk):
            raise TypeError("Runtime accepts EEGChunk or neutral Marker only")
        filtered = self.preprocessor.process(event)
        self.stats["samples_processed"] += len(filtered.sample_indices)
        self.stats["continuity_resets"] += int(np.count_nonzero(filtered.breaks))
        self.stats["bad_signal_samples"] += int(np.count_nonzero(filtered.reasons == "BAD_SIGNAL"))
        self.stats["stale_samples"] += int(np.count_nonzero(filtered.reasons == "STALE_DATA"))
        outputs = []
        # 逐样本写入并及时推断，避免大数据包先覆盖环形缓冲中尚未使用的窗口。
        for i, sample in enumerate(filtered.sample_indices):
            if filtered.breaks[i]:
                self._epoch += 1
                self.decision.clear_streak()
            sample = int(sample)
            if (self._active and self._cue_timestamp is None and sample >= self._cue
                    and np.isfinite(filtered.timestamps[i])):
                self._cue_timestamp = float(filtered.timestamps[i]) - (sample - self._cue) / self.config.sfreq
            self.buffer.append_one(filtered.data[:, i], filtered.timestamps[i], sample,
                                   filtered.valid[i], self._epoch, filtered.reasons[i], event.data[:, i])
            self._last_sample = sample
            self._last_timestamp = float(filtered.timestamps[i])
            while self._active and self._next_window < len(self._starts):
                start = self._starts[self._next_window]
                stop = start + self.config.window_samples
                if stop > sample + 1:
                    break
                outputs.extend(self._predict(start, stop, self._safe_timestamp(stop)))
                self._next_window += 1
        return outputs

    def _safe_timestamp(self, sample):
        if self._last_timestamp is not None and np.isfinite(self._last_timestamp):
            return self._last_timestamp + 1.0 / self.config.sfreq
        return sample / self.config.sfreq

    def _predict(self, start, stop, timestamp):
        began = perf_counter()
        window = self.buffer.read(start, stop)
        quality = "OK"
        probabilities = None
        channel_reasons = {}
        if window is not None:
            channel_reasons = {row["channel"]: row["reasons"] for row in channel_quality_report(
                window.raw_data, self.config.sfreq, self.config.channels, self.quality_limits) if row["reasons"]}
            for name in channel_reasons:
                self.stats["channel_rejected_windows"][name] += 1
        if window is None:
            quality = "INSUFFICIENT_DATA"
        elif np.any(window.reasons == "STALE_DATA"):
            quality = "STALE_DATA"
        elif not window.valid.all() or np.any(window.epochs != window.epochs[0]):
            quality = "BAD_SIGNAL"
        elif channel_reasons:
            quality = "BAD_SIGNAL"
        elif (not np.isfinite(window.data).all()
              or np.any(np.std(window.data, axis=1) <= self.config.flat_std_volts)
              or np.any(np.std(window.raw_data, axis=1) <= self.config.flat_std_volts)):
            quality = "BAD_SIGNAL"
        elif (np.any(np.diff(window.timestamps) <= 0)
              or np.any(np.abs(np.diff(window.timestamps) - 1.0 / self.config.sfreq)
                        > self.config.timestamp_tolerance_seconds)):
            quality = "BAD_SIGNAL"
        # 用提示的真实采集时刻固定窗口位置，防止丢包后的连续编号掩盖时间漂移。
        elif self._cue_timestamp is not None:
            tolerance = max(1.0 / self.config.sfreq, self.config.timestamp_tolerance_seconds) + 1e-9
            expected_start = self._cue_timestamp + (start - self._cue) / self.config.sfreq
            expected_last = self._cue_timestamp + (stop - 1 - self._cue) / self.config.sfreq
            if (abs(window.timestamps[0] - expected_start) > tolerance
                    or abs(window.timestamps[-1] - expected_last) > tolerance):
                quality = "BAD_SIGNAL"
        if quality == "OK":
            # 时效性以最后一个已观测样本为准，右开终点对应的下一个样本尚未到达。
            timestamp = float(window.timestamps[-1])
            if self.validity_guard is not None:
                quality = self.validity_guard(self.trial_id, timestamp) or "OK"
        if quality == "OK":
            try:
                p = np.asarray(self.model.predict_proba(window.data[np.newaxis]), dtype=float)
                if p.shape != (1, 2) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1) or not np.isclose(p.sum(), 1):
                    raise ValueError("Model probabilities must be a normalized 1 x 2 array")
                probabilities = tuple(float(x) for x in p[0])
            except Exception:
                quality = "MODEL_ERROR"
        # 模型运行期间可能发生阶段结束或断流，因此推断后再检查采集线程的最新状态。
        if quality == "OK" and self.validity_guard is not None:
            quality = self.validity_guard(self.trial_id, timestamp) or "OK"
            if quality != "OK":
                probabilities = None
        prediction = Prediction(self.trial_id, start, stop, probabilities, timestamp,
                                self.model_version, quality, perf_counter() - began,
                                tuple(channel_reasons), channel_reasons)
        if quality != "OK":
            self.stats["invalid_windows"] += 1
        if quality == "MODEL_ERROR":
            self.stats["model_errors"] += 1
        if quality == "OK" and self.window_observer is not None:
            self.window_observer(window.data.copy(), prediction)
        return [prediction, self.decision.decide(prediction)]

    def _flush(self, timestamp=None):
        # 阶段结束时将未凑齐的窗口记为无效，不追溯补做推断。
        outputs = []
        while self._active and self._next_window < len(self._starts):
            start = self._starts[self._next_window]
            stop = start + self.config.window_samples
            prediction = Prediction(self.trial_id, start, stop, None,
                                    timestamp if timestamp is not None else self._safe_timestamp(stop),
                                    self.model_version, "INSUFFICIENT_DATA")
            self.stats["invalid_windows"] += 1
            outputs.extend([prediction, self.decision.decide(prediction)])
            self._next_window += 1
        return outputs

    def _marker(self, marker: Marker):
        if marker.kind not in NEUTRAL_MARKERS:
            raise ValueError("Runtime accepts only neutral markers; class-bearing events are forbidden")
        if not isinstance(marker.sample, (int, np.integer)) or marker.sample < 0:
            raise ValueError("Marker sample must be a nonnegative integer")
        if marker.timestamp is not None and not np.isfinite(marker.timestamp):
            raise ValueError("Marker timestamp must be finite")
        if (marker.kind in {"TRIAL_START", "MI_START", "MI_END"}
                and self._last_sample is not None and marker.sample <= self._last_sample):
            raise ValueError("Phase markers must arrive before EEG at their sample; late markers cannot be decoded retrospectively")
        if marker.kind == "RUN_START":
            outputs = self._flush(marker.timestamp)
            self.preprocessor.reset()
            self.buffer.clear()
            self._active = False
            self.trial_id = None
            self.decision.end()
            self._last_sample = self._last_timestamp = None
            self._cue_timestamp = None
            return outputs
        if marker.kind == "DISCONNECT":
            outputs = self._flush(marker.timestamp)
            self.preprocessor.reset()
            self.buffer.clear()
            self._epoch += 1
            self._active = False
            self.decision.end()
            return outputs + [Command(self.trial_id, "NO_COMMAND", marker.sample,
                            marker.timestamp if marker.timestamp is not None else self._safe_timestamp(marker.sample),
                            "SIGNAL_LOST")]
        if marker.kind == "SESSION_END":
            return self.finish()
        if marker.kind == "TRIAL_START":
            if marker.trial_id is None:
                raise ValueError("TRIAL_START requires a trial_id")
            if self.trial_id == marker.trial_id:
                return []
            outputs = self._flush(marker.timestamp)
            self.trial_id = marker.trial_id
            self._active = False
            self._starts = []
            self._cue = None
            self._cue_timestamp = None
            self.decision.reset(marker.trial_id)
            return outputs
        if marker.kind == "MI_START":
            trial_id = marker.trial_id if marker.trial_id is not None else self.trial_id
            if trial_id is None:
                raise ValueError("MI_START requires a current trial_id")
            if self._active and self.trial_id == trial_id:
                if self._cue != marker.sample:
                    raise ValueError("Cannot change cue time of an active trial")
                return []
            outputs = self._flush(marker.timestamp)
            self.trial_id = trial_id
            self._cue = marker.sample
            self._cue_timestamp = marker.timestamp
            self._starts = list(iter_window_starts(self.config, marker.sample))
            self._next_window = 0
            self._active = True
            self.decision.reset(trial_id)
            return outputs
        if marker.trial_id is not None and marker.trial_id != self.trial_id:
            return []
        outputs = self._flush(marker.timestamp)
        self._active = False
        self.decision.end()
        return outputs + [Command(self.trial_id, "NO_COMMAND", marker.sample,
                                 marker.timestamp if marker.timestamp is not None else self._safe_timestamp(marker.sample),
                                 "MI_ENDED")]

    def finish(self):
        outputs = self._flush()
        self._active = False
        self.decision.end()
        return outputs



# ========================================================================
# 09. 按原始时间线连续回放
# ========================================================================
class ReplaySource:
    # 按阶段边界拆分数据包，确保标记先于该位置的 EEG 到达解码器。
    def __init__(self, recording, chunk_samples=25, realtime=False, speed=1.0, imagery_seconds=4.0):
        if isinstance(chunk_samples, bool) or not isinstance(chunk_samples, Integral) or chunk_samples < 1:
            raise ValueError("Chunk size must be a positive integer")
        if isinstance(speed, bool) or not isinstance(speed, Real) or not isfinite(speed) or speed <= 0:
            raise ValueError("Replay speed must be finite and positive")
        if isinstance(imagery_seconds, bool) or not isinstance(imagery_seconds, Real) or not isfinite(imagery_seconds) or imagery_seconds <= 0:
            raise ValueError("Imagery duration must be finite and positive")
        if abs(imagery_seconds * recording.info.sfreq - round(imagery_seconds * recording.info.sfreq)) > 1e-8:
            raise ValueError("Imagery duration must map to integer samples")
        self.recording = recording
        self.info = recording.info
        self.chunk_samples = int(chunk_samples)
        self.realtime = realtime
        self.speed = speed
        self.imagery_seconds = imagery_seconds
        self._stop = threading.Event()

    def close(self):
        self._stop.set()

    def __iter__(self):
        rec = self.recording
        n_samples = rec.data.shape[1]
        markers = []
        runs = sorted(set([0, *rec.run_starts]))
        for segment, sample in enumerate(runs):
            markers.append((sample, 0, "RUN_START", None))
        for trial in rec.trials:
            start = trial.start_sample if trial.start_sample is not None else trial.cue_sample - round(2 * self.info.sfreq)
            markers.extend([
                (max(0, start), 2, "TRIAL_START", trial.trial_id),
                (trial.cue_sample, 3, "MI_START", trial.trial_id),
                (trial.cue_sample + round(self.imagery_seconds * self.info.sfreq), 1, "MI_END", trial.trial_id),
            ])
        # 回放标记只使用阶段定位，不读取 original_class 或人工伪迹标签。
        markers.sort(key=lambda item: (item[0], item[1]))
        if any(m[0] > n_samples or m[0] < 0 for m in markers):
            raise ValueError("Marker outside recording")
        position = 0
        segment = 0
        clock_start = time.perf_counter()
        for sample, _, kind, trial_id in markers:
            while position < sample and not self._stop.is_set():
                end = min(position + self.chunk_samples, sample)
                if self.realtime:
                    self._stop.wait(max(0.0, (end / self.info.sfreq) / self.speed - (time.perf_counter() - clock_start)))
                if self._stop.is_set():
                    return
                yield EEGChunk(rec.data[:, position:end], rec.timestamps[position:end], position, self.info, segment)
                position = end
            if self._stop.is_set():
                return
            if kind == "RUN_START":
                segment = runs.index(sample)
            stamp = float(rec.timestamps[sample]) if sample < n_samples else float(rec.timestamps[-1] + 1 / self.info.sfreq)
            yield Marker(kind, sample, trial_id, stamp, segment)
        while position < n_samples and not self._stop.is_set():
            end = min(position + self.chunk_samples, n_samples)
            if self.realtime:
                self._stop.wait(max(0.0, (end / self.info.sfreq) / self.speed - (time.perf_counter() - clock_start)))
            if self._stop.is_set():
                return
            yield EEGChunk(rec.data[:, position:end], rec.timestamps[position:end], position, self.info, segment)
            position = end



UNIT_SCALE = {"V": 1.0, "mV": 1e-3, "uV": 1e-6, "µV": 1e-6, "μV": 1e-6}
NEUTRAL_KINDS = set(NEUTRAL_MARKERS)


# ========================================================================
# 10. LSL 设备采集与阶段标记对齐
# ========================================================================
def parse_neutral_marker(payload):
    # 只允许 kind 和 trial_id，防止左右手真值通过阶段流进入解码器。
    value = json.loads(payload)
    if not isinstance(value, dict) or set(value) - {"kind", "trial_id"}:
        raise ValueError("Marker schema is strictly {kind, trial_id}; class labels are forbidden")
    if value.get("kind") not in NEUTRAL_KINDS:
        raise ValueError("Only neutral phase markers are accepted")
    trial_id = value.get("trial_id")
    if trial_id is not None and (not isinstance(trial_id, str) or not trial_id):
        raise ValueError("trial_id must be a nonempty string or absent")
    if value["kind"] in {"TRIAL_START", "MI_START", "MI_END"} and (not isinstance(trial_id, str) or not trial_id):
        raise ValueError("Trial marker needs a nonempty trial_id")
    return value


class LiveSource:
    # 核对真实 LSL 流的采样率、通道、单位和参考，再转换成统一事件。
    is_live = True

    def __init__(self, config, eeg_name, marker_name, reference, units=None, resolve_timeout=5.0):
        try:
            import pylsl
        except ImportError as exc:
            raise RuntimeError("Install pylsl: python -m pip install pylsl; also ensure liblsl is available") from exc
        if reference != config.reference:
            raise ValueError("Device reference must match the frozen model; recalibrate after changing reference")
        if units and config.lsl_input_units and UNIT_SCALE.get(units) != UNIT_SCALE[config.lsl_input_units]:
            raise ValueError("Input units differ from the frozen device configuration")
        units = units or config.lsl_input_units
        self.config = config
        self.clock = pylsl.local_clock
        self.info = StreamInfo(config.sfreq, config.channels, reference=reference)
        self._stop = threading.Event()
        self.safety = LiveSafety(config, self.clock, self._stop)
        self.timing_records = []
        self.pending_raw = []
        self._state_lock = threading.Lock()
        self._iterator_active = False
        self.eeg = None
        self.markers = None
        try:
            eeg_matches = pylsl.resolve_byprop("name", eeg_name, timeout=resolve_timeout)
            marker_matches = pylsl.resolve_byprop("name", marker_name, timeout=resolve_timeout)
            if len(eeg_matches) != 1 or len(marker_matches) != 1:
                raise ValueError("Exactly one EEG stream and one marker stream must match the given names")
            self.eeg = pylsl.StreamInlet(eeg_matches[0], max_buflen=10, recover=False, processing_flags=pylsl.proc_clocksync)
            self.markers = pylsl.StreamInlet(marker_matches[0], max_buflen=10, recover=False, processing_flags=pylsl.proc_clocksync)
            header = self.eeg.info(timeout=resolve_timeout)
            if header.channel_format() == pylsl.cf_string:
                raise ValueError("EEG stream must have numeric channels")
            if abs(header.nominal_srate() - config.sfreq) > 1e-8:
                raise ValueError("Device sampling rate differs from model; implicit resampling is disabled")
            marker_header = self.markers.info(timeout=resolve_timeout)
            if marker_header.channel_count() != 1 or marker_header.channel_format() != pylsl.cf_string:
                raise ValueError("Marker stream must have one JSON string channel")
            xml_channel = header.desc().child("channels").child("channel")
            labels, scales = [], []
            for _ in range(header.channel_count()):
                labels.append(xml_channel.child_value("label"))
                unit = xml_channel.child_value("unit")
                if unit and units and UNIT_SCALE.get(unit) != UNIT_SCALE.get(units):
                    raise ValueError("Explicit units conflict with stream metadata")
                unit = unit or units
                scales.append(UNIT_SCALE.get(unit))
                xml_channel = xml_channel.next_sibling()
            self.reported_labels = tuple(labels)
            if config.lsl_channel_labels is not None:
                if len(config.lsl_channel_labels) != header.channel_count():
                    raise ValueError("Explicit LSL channel mapping length differs from the stream")
                # 显式映射代表已核实的接线顺序，不能仅凭通道数量推测电极位置。
                labels = list(config.lsl_channel_labels)
            if len(set(labels)) != len(labels) or any(c not in labels for c in config.channels):
                raise ValueError("LSL metadata must name every required EEG electrode exactly once")
            self.order = [labels.index(c) for c in config.channels]
            if any(scales[i] is None for i in self.order):
                raise ValueError("Missing/unknown EEG units; specify verified V, mV or uV explicitly")
            self.scales = np.array([scales[i] for i in self.order])[:, None]
            reported_reference = header.desc().child_value("reference")
            if reported_reference and reported_reference != reference:
                raise ValueError("Stream reference metadata conflicts with specified reference")
            # 两个入口均由 proc_clocksync 校正到接收端时钟域，后续不手工重复加偏移。
            self.eeg.open_stream(timeout=resolve_timeout)
            self.markers.open_stream(timeout=resolve_timeout)
        except BaseException:
            self.close()
            raise

    def close(self):
        self._stop.set()
        # 运行中的迭代器负责最终关闭入口，避免与正在执行的 pull_chunk 竞争。
        with self._state_lock:
            if not self._iterator_active:
                self._close_inlets()

    def _close_inlets(self):
        failures = []
        for name in ("eeg", "markers"):
            inlet = getattr(self, name, None)
            if inlet is not None:
                try:
                    inlet.close_stream()
                except BaseException as exc:
                    failures.append((name, exc))
                finally:
                    setattr(self, name, None)
        if failures:
            details = "; ".join(f"{name}: {type(exc).__name__}: {exc}" for name, exc in failures)
            raise RuntimeError(f"LSL inlet cleanup failed: {details}") from failures[0][1]

    def _released_eeg(self, values, times, index, segment):
        # 从待保留原始数据中扣除已交付的样本，避免退出恢复时重复记录。
        count = len(times)
        while count:
            packet = self.pending_raw[0]
            used = min(count, len(packet.timestamps))
            count -= used
            if used == len(packet.timestamps):
                self.pending_raw.pop(0)
            else:
                self.pending_raw[0] = replace(packet, data=packet.data[:, used:],
                                              timestamps=packet.timestamps[used:],
                                              start_sample=packet.start_sample + used)
        return EEGChunk(values, times, index, self.info, segment, self.clock())

    def __iter__(self):
        with self._state_lock:
            if self._iterator_active:
                raise RuntimeError("LiveSource supports only one acquisition iterator")
            if self._stop.is_set():
                return
            self._iterator_active = True
        pending = []
        buffered = []
        index = 0
        segment = 0
        last_data = self.clock()
        previous_end = None
        read_index = 0
        last_received_stamp = None
        try:
            while not self._stop.is_set():
                samples, stamps = self.eeg.pull_chunk(timeout=0.1, max_samples=self.config.chunk_samples)
                # 先保留收到的原始样本；即使随后遇到错误标记，也可在收尾阶段恢复记录。
                if stamps:
                    last_data = self.clock()
                    times = np.asarray(stamps, dtype=float)
                    values = np.asarray(samples, dtype=float).T[self.order] * self.scales
                    self.pending_raw.append(EEGChunk(values, times, read_index, self.info, segment, last_data))
                    read_index += len(times)
                    if (not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0)
                            or (last_received_stamp is not None and times[0] <= last_received_stamp)):
                        raise RuntimeError("EEG timestamps must be finite and strictly increasing across chunks")
                    last_received_stamp = float(times[-1])
                    self.safety.eeg(times[-1])
                    buffered.append((values, times, last_data))
                # 先收完当前可用标记，再推进 EEG 时间线。
                while True:
                    payload, stamp = self.markers.pull_sample(timeout=0.0)
                    if payload is None:
                        break
                    if not np.isfinite(stamp):
                        raise ValueError("Marker timestamp must be finite")
                    event = parse_neutral_marker(payload[0])
                    received = self.clock()
                    if stamp - received > self.config.timestamp_tolerance_seconds:
                        raise RuntimeError("Marker clock is ahead of the receiver clock")
                    self.safety.marker(event["kind"], event.get("trial_id"), float(stamp))
                    self.timing_records.append({"kind": event["kind"], "trial_id": event.get("trial_id"),
                                                "timestamp": float(stamp), "received_at": received,
                                                "transport_age_seconds": received - float(stamp)})
                    pending.append((float(stamp), event, received))
                # 短暂延后交付 EEG，让独立传输的标记追上；原时间戳不变，等待仍计入数据年龄。
                pending.sort(key=lambda pair: pair[0])
                while buffered and self.clock() - buffered[0][2] >= self.config.lsl_marker_grace_seconds:
                    values, times, _ = buffered.pop(0)
                    position = 0
                    while pending and pending[0][0] <= times[-1] + 1 / self.config.sfreq:
                        stamp, event, received = pending.pop(0)
                        if previous_end is not None and stamp <= previous_end:
                            raise RuntimeError("Late marker arrived after its EEG was decoded; stopping to preserve causal alignment")
                        # 标记对齐到时间戳不早于它的第一个 EEG 样本，并在该样本之前发出。
                        boundary = int(np.searchsorted(times, stamp, side="left"))
                        if boundary > position:
                            yield self._released_eeg(values[:, position:boundary], times[position:boundary], index + position, segment)
                        if event["kind"] in {"RUN_START", "DISCONNECT"}:
                            segment += 1
                        alignment = float(times[boundary] - stamp) if boundary < len(times) else None
                        yield Marker(event["kind"], index + boundary, event.get("trial_id"), stamp, segment,
                                     received, alignment)
                        if event["kind"] == "SESSION_END":
                            return
                        if event["kind"] == "DISCONNECT":
                            raise RuntimeError("External experiment aborted; this session is incomplete")
                        position = boundary
                    if position < len(times):
                        yield self._released_eeg(values[:, position:], times[position:], index + position, segment)
                    index += len(times)
                    previous_end = float(times[-1])
                if not stamps and self.clock() - last_data > 1.0:
                    self.safety.fail("SIGNAL_LOST")
                    yield Marker("DISCONNECT", index, timestamp=self.clock(), segment_id=segment)
                    raise RuntimeError("No EEG for one second; restart source after reconnecting")
        except BaseException:
            self.safety.fail("ACQUISITION_FAILED")
            raise
        finally:
            with self._state_lock:
                self._iterator_active = False
                self._close_inlets()



# ========================================================================
# 11. 异步记录与原始文件完整性检查
# ========================================================================
def json_value(value):
    if is_dataclass(value):
        return {"type": type(value).__name__, **asdict(value)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path, value):
    # 先完成序列化和临时文件写入，再原子替换目标，避免留下半份 JSON。
    path = Path(path)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, default=json_value, allow_nan=False)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class AsyncRecorder:
    # 采集线程入队，写盘线程保存原始 EEG 与事件；队列满时明确报错。
    def __init__(self, directory, info, capacity=256):
        if capacity < 1:
            raise ValueError("Recorder capacity must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.info = info
        self.queue = queue.Queue(maxsize=capacity)
        self.error = None
        self.max_backlog = 0
        self.raw_samples = 0
        self.processed_raw_samples = 0
        self.accepted_raw_samples = 0
        self.closed = False
        self.thread = threading.Thread(target=self._worker, name="mi-recorder", daemon=True)
        write_json(self.directory / "manifest.json", {"format": "mi-bci-session-v1", "status": "recording", "stream": info, "raw_layout": "sample-major little-endian float64, raw_eeg.f64 and timestamps.f64; chunk offsets in chunks.jsonl"})
        self.thread.start()

    def record(self, item, timeout=0):
        if self.closed:
            raise RuntimeError("Recorder already closed")
        if self.error:
            raise RuntimeError("Recorder write failed") from self.error
        if isinstance(item, EEGChunk):
            # 采集源可能复用数组；入队前复制，防止后台写盘读到后来覆盖的数据。
            item = EEGChunk(item.data.copy(), item.timestamps.copy(), item.start_sample, item.info, item.segment_id, item.received_at)
        try:
            self.queue.put(item, block=bool(timeout), timeout=timeout if timeout else None)
        except queue.Full as exc:
            raise RuntimeError("Recorder backlog full; session stopped to prevent silent loss") from exc
        self.max_backlog = max(self.max_backlog, self.queue.qsize())
        if isinstance(item, EEGChunk):
            self.accepted_raw_samples += item.data.shape[1]

    def _worker(self):
        try:
            with (self.directory / "raw_eeg.f64").open("wb") as raw, (self.directory / "timestamps.f64").open("wb") as stamps, (self.directory / "chunks.jsonl").open("w", encoding="utf-8") as chunks, (self.directory / "events.jsonl").open("w", encoding="utf-8") as events:
                while True:
                    item = self.queue.get()
                    try:
                        if item is None:
                            return
                        if isinstance(item, EEGChunk):
                            n = item.data.shape[1]
                            raw.write(np.asarray(item.data.T, dtype="<f8").tobytes())
                            stamps.write(np.asarray(item.timestamps, dtype="<f8").tobytes())
                            chunks.write(json.dumps({"offset_samples": self.processed_raw_samples, "count": n, "start_sample": item.start_sample, "segment_id": item.segment_id, "received_at": item.received_at}) + "\n")
                            # 写入缓冲的样本数不等于已确认保存数，最终以关闭文件后的核验为准。
                            self.processed_raw_samples += n
                        else:
                            events.write(json.dumps(item, ensure_ascii=False, default=json_value, allow_nan=False) + "\n")
                    finally:
                        self.queue.task_done()
        except BaseException as exc:
            self.error = exc

    def _confirm_raw_storage(self):
        # 核对已关闭文件与块索引共同确认的样本前缀；这不保证断电后的持久性。
        result = {"status": "unconfirmed", "confirmed_prefix_samples": 0,
                  "raw_file_samples": None, "timestamp_file_samples": None, "issues": [],
                  "scope": "Closed-file byte lengths plus complete contiguous chunk-index rows; no fsync/power-loss durability claim."}
        if self.thread.is_alive():
            result["issues"].append("RECORDER_THREAD_STILL_RUNNING")
            return result
        try:
            raw_bytes = (self.directory / "raw_eeg.f64").stat().st_size
            stamp_bytes = (self.directory / "timestamps.f64").stat().st_size
            stride = len(self.info.channels) * 8
            raw_count, stamp_count = raw_bytes // stride, stamp_bytes // 8
            result.update({"raw_file_samples": raw_count, "timestamp_file_samples": stamp_count,
                           "raw_trailing_bytes": raw_bytes % stride, "timestamp_trailing_bytes": stamp_bytes % 8})
            if raw_bytes % stride or stamp_bytes % 8:
                result["issues"].append("PARTIAL_SAMPLE_BYTES")
            if raw_count != stamp_count:
                result["issues"].append("RAW_TIMESTAMP_LENGTH_MISMATCH")
            available = min(raw_count, stamp_count)
            with (self.directory / "chunks.jsonl").open("rb") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.endswith(b"\n"):
                        result["issues"].append(f"INCOMPLETE_CHUNK_INDEX_LINE_{line_number}")
                        break
                    try:
                        row = json.loads(line)
                        offset, count = row["offset_samples"], row["count"]
                        if (isinstance(offset, bool) or not isinstance(offset, int)
                                or isinstance(count, bool) or not isinstance(count, int) or count <= 0
                                or offset != result["confirmed_prefix_samples"]):
                            raise ValueError("Noncontiguous or invalid chunk indices")
                        if offset + count > available:
                            result["issues"].append(f"CHUNK_INDEX_EXCEEDS_RETAINED_BYTES_AT_LINE_{line_number}")
                            break
                        result["confirmed_prefix_samples"] += count
                    except (ValueError, KeyError, TypeError) as exc:
                        result["issues"].append(f"INVALID_CHUNK_INDEX_LINE_{line_number}: {exc}")
                        break
            confirmed = result["confirmed_prefix_samples"]
            result["raw_samples_without_confirmed_index"] = max(0, raw_count - confirmed)
            result["timestamp_samples_without_confirmed_index"] = max(0, stamp_count - confirmed)
            if confirmed != raw_count or confirmed != stamp_count:
                result["issues"].append("SAMPLES_WITHOUT_CONFIRMED_CHUNK_INDEX")
            if confirmed != self.accepted_raw_samples:
                result["issues"].append("ACCEPTED_SAMPLES_NOT_FULLY_CONFIRMED")
            result["status"] = "confirmed" if not result["issues"] else "partial_or_unconfirmed"
        except BaseException as exc:
            result["issues"].append(f"{type(exc).__name__}: {exc}")
        return result

    def close(self, status="complete", error=None, received_raw_samples=None):
        if self.closed:
            return
        self.closed = True
        deadline = time.monotonic() + 10
        while self.thread.is_alive():
            if time.monotonic() >= deadline:
                if self.error is None:
                    self.error = RuntimeError("Recorder did not accept shutdown within 10 seconds")
                break
            try:
                self.queue.put(None, timeout=0.1)
                break
            except queue.Full:
                continue
        self.thread.join(timeout=max(0, deadline - time.monotonic()))
        if self.thread.is_alive():
            if self.error is None:
                self.error = RuntimeError("Recorder did not finish within 10 seconds")
        storage = self._confirm_raw_storage()
        self.raw_samples = storage["confirmed_prefix_samples"]
        if storage["status"] != "confirmed" and self.error is None:
            self.error = RuntimeError("Recorded EEG storage could not be fully confirmed: " + "; ".join(storage["issues"]))
        write_json(self.directory / "manifest.json", {
            "format": "mi-bci-session-v1", "status": "failed" if self.error or error else status,
            "error": str(self.error or error) if self.error or error else None,
            "stream": self.info, "raw_samples": self.raw_samples, "max_recorder_backlog": self.max_backlog,
            "processed_raw_samples": self.processed_raw_samples,
            "unconfirmed_processed_samples": max(0, self.processed_raw_samples - self.raw_samples),
            "storage_verification": storage,
            "received_raw_samples": received_raw_samples,
            "accepted_raw_samples": self.accepted_raw_samples,
            "unwritten_accepted_samples": max(0, self.accepted_raw_samples - self.raw_samples),
            "unconfirmed_accepted_samples": max(0, self.accepted_raw_samples - self.raw_samples),
            "failed_to_retain_samples": max(0, received_raw_samples - self.accepted_raw_samples)
                                        if received_raw_samples is not None else None,
            "raw_layout": "sample-major little-endian float64, raw_eeg.f64 and timestamps.f64; chunk offsets in chunks.jsonl",
        })
        if self.error:
            raise RuntimeError("Recorder failed") from self.error



# ========================================================================
# 11a. 合并真实采集记录与独立标签
# ========================================================================
def load_personal_session(directory, labels_path, *, subject_id=None, session_id=None) -> Recording:
    # 只导入完整采集，阶段位置以接收端 Marker.sample 为准，保持 EEG 时间线不变。
    # 独立真值按 trial_id 合并，并核对完整阶段序列；不以发送端时间戳重新定位 EEG。
    directory, labels_path = Path(directory), Path(labels_path)

    def read_object(path):
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object: {path.name}")
        return value

    def read_rows(path):
        rows = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                raise ValueError(f"Empty JSONL row at {path.name}:{number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path.name}:{number}")
            rows.append(row)
        return rows

    def integer(value, name, minimum=0):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
        return value

    def finite_number(value, name):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    manifest = read_object(directory / "manifest.json")
    if manifest.get("format") != "mi-bci-session-v1" or manifest.get("status") != "complete" or manifest.get("error"):
        raise ValueError("Personal import requires a complete, error-free recorder session")
    context = read_object(directory / "capture_context.json")
    if context.get("mode") not in {"calibration", "live"}:
        raise ValueError("Capture context must identify calibration or live mode")
    captured_subject, captured_session = context.get("subject_id"), context.get("session_id")
    if not isinstance(captured_subject, str) or not captured_subject.strip() or captured_session not in {"T", "E"}:
        raise ValueError("Capture context requires subject_id and session_id T or E")
    if (subject_id is not None and subject_id != captured_subject) or (session_id is not None and session_id != captured_session):
        raise ValueError("Subject/session override conflicts with the original capture context")
    config_values = context.get("config")
    if (not isinstance(config_values, dict) or not {"sfreq", "channels", "reference"}.issubset(config_values)
            or set(config_values) - {f.name for f in fields(Config)}):
        raise ValueError("Capture context must contain the complete acquisition configuration")
    config = Config.from_dict(config_values)
    stream = manifest.get("stream")
    if not isinstance(stream, dict):
        raise ValueError("Recorder manifest has no stream metadata")
    stream = dict(stream)
    stream_type = stream.pop("type", "StreamInfo")
    if stream_type != "StreamInfo":
        raise ValueError("Unexpected recorder stream metadata type")
    info = StreamInfo(**stream)
    info = replace(info, channels=tuple(info.channels))
    if (info.sfreq, info.channels, info.unit, info.reference) != (config.sfreq, config.channels, config.unit, config.reference):
        raise ValueError("Capture configuration and recorded stream metadata disagree")

    count = integer(manifest.get("raw_samples"), "raw_samples", 1)
    for name in ("received_raw_samples", "accepted_raw_samples"):
        if integer(manifest.get(name), name) != count:
            raise ValueError("Received, accepted and persisted sample counts must agree for a complete capture")
    for name in ("unwritten_accepted_samples", "failed_to_retain_samples"):
        if integer(manifest.get(name), name) != 0:
            raise ValueError("Complete capture contains unretained or unwritten samples")
    raw_path, timestamp_path = directory / "raw_eeg.f64", directory / "timestamps.f64"
    if raw_path.stat().st_size != count * len(info.channels) * 8 or timestamp_path.stat().st_size != count * 8:
        raise ValueError("Raw EEG/timestamp file size disagrees with the recorder manifest")
    data = np.fromfile(raw_path, dtype="<f8").reshape(count, len(info.channels)).T.copy()
    timestamps = np.fromfile(timestamp_path, dtype="<f8")
    chunks = read_rows(directory / "chunks.jsonl")
    offset, previous_segment = 0, -1
    for chunk in chunks:
        chunk_count = integer(chunk.get("count"), "chunk count", 1)
        chunk_offset = integer(chunk.get("offset_samples"), "chunk offset")
        chunk_start = integer(chunk.get("start_sample"), "chunk start")
        segment = integer(chunk.get("segment_id"), "chunk segment")
        if chunk_offset != offset or chunk_start != offset or segment < previous_segment:
            raise ValueError("Chunk indices are discontinuous, overlapping, or out of order")
        finite_number(chunk.get("received_at"), "chunk receipt timestamp")
        offset += chunk_count
        previous_segment = segment
    if offset != count:
        raise ValueError("Chunk log does not cover the complete recorded timeline")

    allowed_phases = {"RUN_START", "TRIAL_START", "MI_START", "MI_END", "SESSION_END"}
    markers = []
    recorded_events = read_rows(directory / "events.jsonl")
    if (sum(event.get("type") == "SessionEnd" for event in recorded_events) != 1
            or not recorded_events or recorded_events[-1].get("type") != "SessionEnd"):
        raise ValueError("Capture requires exactly one final successful SessionEnd record")
    for event in recorded_events:
        event_type = event.get("type")
        if event_type == "Marker":
            if event.get("kind") not in allowed_phases:
                raise ValueError("Disconnected, aborted, or unrecognized marker in personal capture")
            if any(key in event for key in ("original_class", "class_label", "label")):
                raise ValueError("Recorded neutral markers must not contain class labels")
            markers.append(event)
        elif event_type == "SessionEnd":
            if (event.get("error") or event.get("cancelled") or event.get("pending_acquisition_packets", 0)
                    or event.get("failed_to_retain_samples", 0) or event.get("cleanup_errors")):
                raise ValueError("Session ended with an error, cancellation, or unrecorded packets")
        elif event_type not in {"Prediction", "Command"}:
            raise ValueError("Unrecognized event in personal capture")
    truth_rows = read_rows(labels_path)
    truth = {}
    truth_signature = []
    previous_truth_stamp = -float("inf")
    for event in truth_rows:
        kind, trial_id = event.get("kind"), event.get("trial_id")
        if kind not in allowed_phases:
            raise ValueError("Unrecognized or aborted phase in independent truth log")
        stamp = finite_number(event.get("timestamp"), "truth timestamp")
        if stamp < previous_truth_stamp:
            raise ValueError("Truth events must be chronological within their own clock")
        previous_truth_stamp = stamp
        original = event.get("original_class")
        if kind == "MI_START":
            if not isinstance(trial_id, str) or not trial_id.strip() or trial_id in truth:
                raise ValueError("MI_START truth labels require unique, nonempty trial IDs")
            if isinstance(original, bool) or not isinstance(original, int) or original not in (1, 2):
                raise ValueError("Personal truth labels must be 1=LEFT or 2=RIGHT")
            truth[trial_id] = original
        elif original is not None:
            raise ValueError("Class labels are allowed only on independent MI_START events")
        truth_signature.append((kind, trial_id))
    if [(event.get("kind"), event.get("trial_id")) for event in markers] != truth_signature:
        raise ValueError("Captured neutral phases and independent truth log do not match exactly")

    trials, run_starts = [], []
    active, ended = None, False
    previous_sample, previous_stamp, previous_segment = -1, -float("inf"), -1
    run_trial_count = 0
    seen_trials = set()
    imagery_samples = config.samples(config.imagery_seconds)
    for event in markers:
        kind, trial_id = event["kind"], event.get("trial_id")
        sample = integer(event.get("sample"), "marker sample")
        stamp = finite_number(event.get("timestamp"), "received marker timestamp")
        segment = integer(event.get("segment_id"), "marker segment")
        if ended or sample > count or sample < previous_sample or stamp < previous_stamp or segment < previous_segment:
            raise ValueError("Marker timeline is out of order or outside the complete recording")
        previous_sample, previous_stamp, previous_segment = sample, stamp, segment
        if kind == "RUN_START":
            if active is not None or (run_starts and run_trial_count == 0) or sample == count or (run_starts and sample <= run_starts[-1]):
                raise ValueError("RUN_START interrupts a trial or creates an empty/duplicate run")
            run_starts.append(sample)
            run_trial_count = 0
        elif kind == "TRIAL_START":
            if not run_starts or active is not None or not isinstance(trial_id, str) or not trial_id.strip() or trial_id in seen_trials:
                raise ValueError("TRIAL_START requires an active run and a unique trial ID")
            active = {"id": trial_id, "start": sample, "cue": None}
            seen_trials.add(trial_id)
        elif kind == "MI_START":
            if active is None or active["id"] != trial_id or active["cue"] is not None:
                raise ValueError("MI_START must follow its matching TRIAL_START exactly once")
            active["cue"] = sample
        elif kind == "MI_END":
            if active is None or active["id"] != trial_id or active["cue"] is None:
                raise ValueError("MI_END must follow its matching MI_START exactly once")
            if sample - active["cue"] < imagery_samples - 2 or active["cue"] + imagery_samples > count:
                raise ValueError("Imagery phase or recorded EEG is truncated before its configured duration")
            trials.append(Trial(captured_subject, captured_session, len(run_starts), trial_id,
                                active["cue"], truth[trial_id], False, active["start"]))
            active = None
            run_trial_count += 1
        elif kind == "SESSION_END":
            if active is not None or not trials or run_trial_count == 0:
                raise ValueError("SESSION_END requires complete trials and a nonempty final run")
            ended = True
    if not ended or active is not None or set(truth) != seen_trials:
        raise ValueError("Capture requires SESSION_END and exactly one truth label per complete trial")

    return Recording(data=data, timestamps=timestamps, info=info, trials=trials,
                     run_starts=run_starts, subject_id=captured_subject, session_id=captured_session,
                     metadata={"source": "personal_lsl",
                               "capture_directory": directory.name, "labels_file": labels_path.name,
                               "capture_context": context, "config": config.to_dict(),
                               "label_alignment": "Exact trial IDs and phase sequence; cue samples from received neutral markers",
                               "artifact_annotation": "Not manually reviewed; runtime signal-quality checks remain active"})



CLASS_MAPPING = {"0": "LEFT", "1": "RIGHT"}


# ========================================================================
# 12. 冻结模型包与一致性校验
# ========================================================================
def canonical_json(value):
    # 固定键顺序和分隔符，使同一份元数据生成稳定的摘要。
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def dependency_versions():
    return {"python": platform.python_version(), **{
        name: package_version(name) for name in ("numpy", "scipy", "scikit-learn", "mne", "joblib")}}


@dataclass
class ModelBundle:
    # 将模型、配置、滤波系数、版本及训练报告作为同一套部署资料保存。
    config: Config
    model: object
    sos: np.ndarray
    version: str
    metadata: dict = field(default_factory=dict)
    report: dict = field(default_factory=dict)

    def validate(self):
        if self.metadata.get("synthetic") or self.metadata.get("recording_metadata", {}).get("synthetic"):
            raise ValueError("Only recorded-EEG models are supported; legacy synthetic models are not accepted")
        if self.metadata.get("recording_metadata", {}).get("source") != "personal_lsl":
            raise ValueError("Model must be trained from a personal LSL calibration recording")
        # 重新构造配置，确保模型包也通过所有参数约束。
        Config.from_dict(self.config.to_dict())
        sos = np.asarray(self.sos, dtype=float)
        if sos.shape != (self.config.filter_order, 6) or not np.isfinite(sos).all() or not np.allclose(sos[:, 3], 1):
            raise ValueError("Bundle SOS coefficients do not match the filter contract")
        expected_sos = butter(self.config.filter_order, [self.config.low_hz, self.config.high_hz],
                              btype="bandpass", fs=self.config.sfreq, output="sos")
        if not np.allclose(sos, expected_sos, rtol=1e-9, atol=1e-12):
            raise ValueError("Bundle SOS coefficients disagree with the configured Butterworth filter")
        if not self.version:
            raise ValueError("Model version is required")
        classes = np.asarray(getattr(self.model, "classes_", []))
        if not np.array_equal(classes, [0, 1]):
            raise ValueError("Model probability columns must be [0=LEFT, 1=RIGHT]")
        steps = getattr(self.model, "named_steps", {})
        csp, lda = steps.get("csp"), steps.get("lda")
        if csp is None or lda is None:
            raise ValueError("Expected a fitted CSP/LDA pipeline")
        filters = np.asarray(getattr(csp, "filters_", []))
        if filters.ndim != 2 or filters.shape[1] != len(self.config.channels) or csp.n_components != self.config.n_components:
            raise ValueError("CSP channel/component dimensions differ from the frozen configuration")
        if getattr(lda, "n_features_in_", None) != self.config.n_components:
            raise ValueError("LDA feature count differs from the frozen CSP configuration")
        if not self.metadata.get("training_subject") or self.metadata.get("training_session") != "T":
            raise ValueError("Bundle must identify its training subject and T session")
        if self.metadata.get("class_mapping") != CLASS_MAPPING:
            raise ValueError("Unsupported class mapping")
        if self.metadata.get("filter_initialization") != "zero_state_per_continuous_segment":
            raise ValueError("Unsupported filter initialization policy")
        if self.metadata.get("reference") != self.config.reference or self.metadata.get("warmup_seconds") != self.config.warmup_seconds:
            raise ValueError("Training metadata differs from the frozen reference/warmup configuration")
        if self.metadata.get("window_quality_limits") != asdict(QualityThresholds.for_window(self.config)):
            raise ValueError("Frozen channel-quality policy is missing or differs; retrain from calibration data")

    def save(self, path):
        # 模型文件、配置及清单均带摘要，用于加载时核对文件是否一致。
        self.validate()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        model_path = path / "model.joblib"
        temporary_model = path / "model.joblib.tmp"
        joblib.dump(self.model, temporary_model, compress=3)
        model_hash = sha256(temporary_model.read_bytes()).hexdigest()
        manifest = {
            "schema_version": 1, "model_version": self.version,
            "config": self.config.to_dict(), "config_sha256": digest(self.config.to_dict()),
            "sos": np.asarray(self.sos, dtype=float).tolist(), "class_mapping": CLASS_MAPPING,
            "filter_initialization": "zero_state_per_continuous_segment",
            "model_file": "model.joblib", "model_sha256": model_hash,
            "dependencies": self.metadata.get("dependencies", dependency_versions()),
            "metadata": self.metadata, "training_report": self.report,
        }
        manifest["manifest_sha256"] = digest(manifest)
        temporary_manifest = path / "manifest.json.tmp"
        temporary_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary_model.replace(model_path)
        temporary_manifest.replace(path / "manifest.json")
        return path

    @classmethod
    def load(cls, path):
        # joblib 反序列化只适用于可信来源的模型；摘要校验不能证明来源可信。
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        recorded_digest = manifest.pop("manifest_sha256", None)
        if recorded_digest != digest(manifest):
            raise ValueError("Bundle manifest hash mismatch")
        if manifest.get("schema_version") != 1 or manifest.get("model_file") != "model.joblib":
            raise ValueError("Unsupported model bundle format")
        if manifest.get("class_mapping") != CLASS_MAPPING:
            raise ValueError("Bundle class mapping must be [LEFT, RIGHT]")
        if manifest.get("filter_initialization") != "zero_state_per_continuous_segment":
            raise ValueError("Unsupported filter initialization policy")
        if manifest.get("config_sha256") != digest(manifest["config"]):
            raise ValueError("Bundle configuration hash mismatch")
        model_path = path / "model.joblib"
        if manifest.get("model_sha256") != sha256(model_path.read_bytes()).hexdigest():
            raise ValueError("Bundle classifier hash mismatch")
        current_versions = dependency_versions()
        mismatches = {name: (old, current_versions.get(name)) for name, old in manifest.get("dependencies", {}).items()
                      if current_versions.get(name) != old}
        if mismatches:
            warnings.warn(f"Bundle dependency versions differ (saved, current): {mismatches}", RuntimeWarning)
        model = joblib.load(model_path)
        bundle = cls(Config.from_dict(manifest["config"]), model, np.asarray(manifest["sos"], dtype=float),
                     manifest["model_version"], manifest.get("metadata", {}), manifest.get("training_report", {}))
        bundle.validate()
        return bundle


def load_bundle(path):
    return ModelBundle.load(path)



# ========================================================================
# 13. 按试次独立评估分类与指令
# ========================================================================
def _value(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {"count": len(values), "mean": float(values.mean()),
            "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
            "max": float(values.max())}


def _score(rows):
    # 分类准确率只使用可分类试次；覆盖率和正确指令比例以全部目标试次为分母。
    classified = [r for r in rows if r["predicted_class"] is not None]
    emitted = [r for r in rows if r["command_class"] is not None]
    correct = sum(r["command_correct"] for r in emitted)
    truth = [r["target_class"] for r in classified]
    pred = [r["predicted_class"] for r in classified]
    # 可分类试次缺少任一目标类别时，不报告可能误导的平衡准确率。
    balanced = float(balanced_accuracy_score(truth, pred)) if len(set(truth)) == 2 else None
    return {
        "target_trials": len(rows), "classified_trials": len(classified),
        "missing_classifications": len(rows) - len(classified),
        "accuracy": float(accuracy_score(truth, pred)) if truth else None,
        "balanced_accuracy": balanced,
        "confusion_matrix": confusion_matrix(truth, pred, labels=[0, 1]).tolist() if truth else [[0, 0], [0, 0]],
        "confusion_labels": ["LEFT", "RIGHT"],
        "command_trials": len(emitted), "correct_command_trials": correct,
        "wrong_command_trials": len(emitted) - correct,
        "command_accuracy": correct / len(emitted) if emitted else None,
        "coverage": len(emitted) / len(rows) if rows else None,
        "correct_command_fraction": correct / len(rows) if rows else None,
        "cue_to_command_seconds": _distribution([r["cue_to_command_seconds"] for r in emitted if r["cue_to_command_seconds"] is not None]),
        "invalid_command_timing_trials": sum(not r["command_in_phase"] for r in emitted),
        "command_processing_seconds": _distribution([r["command_processing_seconds"] for r in emitted]),
        "cue_to_output_seconds": _distribution([r["cue_to_output_seconds"] for r in emitted
                                                if r["cue_to_output_seconds"] is not None]),
        "result_age_at_output_seconds": _distribution([r["result_age_seconds"] for r in emitted
                                                       if r["result_age_seconds"] is not None]),
        "commands_within_4_seconds": sum(r["cue_to_output_seconds"] is not None
                                          and 0 <= r["cue_to_output_seconds"] <= 4 for r in emitted),
        "valid_window_fraction": (sum(r["valid_windows"] for r in rows) /
                                  sum(r["expected_windows"] for r in rows))
                                 if any(r["expected_windows"] for r in rows) else None,
        "expected_windows": sum(r["expected_windows"] for r in rows),
        "missing_windows": sum(r["missing_windows"] for r in rows),
        "output_timing_measurements_complete": bool(emitted) and all(r["output_timing_measured"] for r in emitted),
        "invalid_windows": sum(r["invalid_windows"] for r in rows),
        "invalid_window_reasons": dict(sum((Counter(r["invalid_window_reasons"]) for r in rows), Counter())),
    }


def _evaluation_parameters(config):
    return {name: getattr(config, name) for name in (
        "sfreq", "imagery_seconds", "window_seconds", "step_seconds", "cue_offset_seconds", "max_age_seconds")}


def evaluate(trials, predictions, commands, *, sfreq, imagery_seconds=4.0,
             window_seconds=2.0, step_seconds=0.2, cue_offset_seconds=0.5, max_age_seconds=0.5):
    # 按试次平均有效窗口概率得到分类结果，方向指令单独评分。
    # 缺失或被拒绝的试次仍计入覆盖率等指标；NO_COMMAND 不算方向指令。
    # 采样序号用于计算提示到指令的时间，软件处理耗时与实际发布时延分别统计。
    if any(isinstance(v, bool) or not isinstance(v, Real) or not isfinite(v) or v <= 0
           for v in (sfreq, imagery_seconds, window_seconds, step_seconds, max_age_seconds)):
        raise ValueError("Evaluation rates and durations must be finite and positive")
    if (isinstance(cue_offset_seconds, bool) or not isinstance(cue_offset_seconds, Real)
            or not isfinite(cue_offset_seconds) or cue_offset_seconds < 0
            or cue_offset_seconds + window_seconds > imagery_seconds):
        raise ValueError("Evaluation window must fit in the imagery phase")
    durations = (imagery_seconds, window_seconds, step_seconds, cue_offset_seconds)
    if any(abs(v * sfreq - round(v * sfreq)) > 1e-8 for v in durations):
        raise ValueError("Evaluation timing must map to integer samples")
    window_samples, step_samples = round(window_seconds * sfreq), round(step_seconds * sfreq)
    trials, predictions, commands = list(trials), list(predictions), list(commands)
    for item in [*predictions, *commands]:
        for name in ("timestamp", "processing_seconds", "emitted_at", "cue_to_output_seconds", "result_age_seconds"):
            value = _value(item, name)
            if name in {"timestamp", "processing_seconds"} and value is None:
                raise ValueError(f"Evaluation event requires {name}")
            if value is not None and (isinstance(value, (bool, np.bool_))
                                      or not isinstance(value, Real) or not isfinite(value)):
                raise ValueError(f"Evaluation {name} must be a finite number or an explicitly missing measurement")
        if _value(item, "processing_seconds", 0) < 0:
            raise ValueError("Processing time cannot be negative")
    if any(isinstance(_value(t, "original_class"), bool)
           or not isinstance(_value(t, "original_class"), Integral)
           or _value(t, "original_class") not in (1, 2) for t in trials):
        raise ValueError("Scoring requires personal truth labels 1=LEFT or 2=RIGHT")
    ids = [_value(t, "trial_id") for t in trials]
    if len(ids) != len(set(ids)):
        raise ValueError("Trial identifiers must be unique for label alignment")
    trial_ids = set(ids)
    pred_by, cmd_by = defaultdict(list), defaultdict(list)
    for p in predictions:
        pred_by[_value(p, "trial_id")].append(p)
    for command in commands:
        cmd_by[_value(command, "trial_id")].append(command)
    target_rows = []
    duplicate_commands = 0
    out_of_phase = 0
    for trial in trials:
        trial_id = _value(trial, "trial_id")
        cue_sample = int(_value(trial, "cue_sample"))
        phase_end = cue_sample + round(imagery_seconds * sfreq)
        expected_starts = set(range(cue_sample + round(cue_offset_seconds * sfreq),
                                    phase_end - window_samples + 1, step_samples))
        expected_stops = {start + window_samples for start in expected_starts}
        observed_starts = set()
        trial_predictions = sorted(pred_by[trial_id], key=lambda p: _value(p, "stop_sample", 0))
        valid = []
        invalid = Counter()
        for p in trial_predictions:
            quality = _value(p, "quality", "OK")
            probs = _value(p, "probabilities")
            start, stop = _value(p, "start_sample"), _value(p, "stop_sample")
            if start not in expected_starts or stop != start + window_samples:
                raise ValueError("Prediction window is outside the frozen window schedule")
            if start in observed_starts:
                raise ValueError("Duplicate prediction window in trial")
            observed_starts.add(start)
            if quality != "OK" or probs is None:
                invalid[quality if quality != "OK" else "MISSING_PROBABILITIES"] += 1
                continue
            probs = np.asarray(probs, dtype=float)
            if probs.shape != (2,) or not np.isfinite(probs).all() or np.any(probs < 0) or not np.isclose(probs.sum(), 1):
                raise ValueError("Invalid probability vector in evaluation")
            valid.append(probs)
        missing_windows = len(expected_starts - observed_starts)
        if missing_windows:
            invalid["MISSING_WINDOW"] += missing_windows
        mean = np.mean(valid, axis=0) if valid else None
        direction_commands = sorted(
            [c for c in cmd_by[trial_id] if _value(c, "direction") in ("LEFT", "RIGHT")],
            key=lambda c: _value(c, "sample", 0))
        duplicate_commands += max(0, len(direction_commands) - 1)
        def command_in_phase(command):
            if _value(command, "sample") not in expected_stops:
                return False
            latency = _value(command, "cue_to_output_seconds")
            age = _value(command, "result_age_seconds")
            if latency is not None and (not isfinite(latency) or not 0 <= latency < imagery_seconds):
                return False
            if age is not None and (not isfinite(age) or not 0 <= age <= max_age_seconds):
                return False
            return True
        for command in direction_commands:
            if not command_in_phase(command):
                out_of_phase += 1
        first = direction_commands[0] if direction_commands else None
        in_phase = bool(first and command_in_phase(first))
        original = _value(trial, "original_class")
        command_class = (0 if _value(first, "direction") == "LEFT" else 1) if first else None
        row = {
            "trial_id": trial_id, "run_id": _value(trial, "run_id"),
            "original_class": original, "target_class": original - 1,
            "artifact_flag": bool(_value(trial, "artifact_flag", False)),
            "valid_windows": len(valid), "invalid_windows": sum(invalid.values()),
            "expected_windows": len(expected_starts), "missing_windows": missing_windows,
            "invalid_window_reasons": dict(invalid),
            "mean_probabilities": mean.tolist() if mean is not None else None,
            "predicted_class": int(np.argmax(mean)) if mean is not None else None,
            "command_class": command_class,
            "command_in_phase": in_phase,
            "command_correct": bool(in_phase and command_class == original - 1),
            "direction_commands": len(direction_commands),
            "cue_to_command_seconds": (_value(first, "sample") - cue_sample) / sfreq if in_phase else None,
            "command_processing_seconds": float(_value(first, "processing_seconds", 0.0)) if first else None,
            "cue_to_output_seconds": _value(first, "cue_to_output_seconds") if first else None,
            "result_age_seconds": _value(first, "result_age_seconds") if first else None,
            "output_timing_measured": bool(first and _value(first, "cue_to_output_seconds") is not None
                                            and _value(first, "result_age_seconds") is not None),
            "no_command_reasons": dict(Counter(_value(c, "reason") for c in cmd_by[trial_id] if _value(c, "direction") == "NO_COMMAND")),
        }
        target_rows.append(row)
    all_scores = _score(target_rows)
    return {
        "scope": "Personal left/right cue-synchronous trial scoring.",
        "aggregation": "Mean valid window probabilities per trial; first directional command per trial.",
        "probability_note": "LDA scores are not calibrated correctness guarantees.",
        "latency_note": "cue_to_command_seconds uses sample indices. cue_to_output_seconds ends at the final pre-send safety check (includes queue/filter/inference/prior disk work), not transport completion. Physical display onset and actuator acknowledgement are not measured.",
        "physical_cue_alignment": "not_measured",
        "actuator_latency": "not_measured",
        "window_schedule": {"window_seconds": window_seconds, "step_seconds": step_seconds,
                            "cue_offset_seconds": cue_offset_seconds, "imagery_seconds": imagery_seconds},
        "all": all_scores,
        "artifact_clean": _score([r for r in target_rows if not r["artifact_flag"]]),
        "artifact_flagged": _score([r for r in target_rows if r["artifact_flag"]]),
        "stability": {
            "duplicate_direction_commands": duplicate_commands, "out_of_phase_direction_commands": out_of_phase,
            "unknown_trial_predictions": sum(_value(p, "trial_id") not in trial_ids for p in predictions),
            "unknown_trial_direction_commands": sum(_value(c, "trial_id") not in trial_ids and _value(c, "direction") in ("LEFT", "RIGHT") for c in commands),
            "prediction_processing_seconds": _distribution([float(_value(p, "processing_seconds", 0.0)) for p in predictions]),
        },
        "per_trial": target_rows,
    }



# ========================================================================
# 14. CSP 特征提取与 LDA 分类器训练
# ========================================================================
class _WindowProbe:
    # 窗口提取用的占位模型返回等概率；有效窗口由观察回调收集。
    def predict_proba(self, windows):
        return np.full((len(windows), 2), 0.5)


@dataclass
class ExtractedWindows:
    # 有效输入窗口及其对应预测位置；完整预测列表也保留被拒绝的窗口。
    windows: np.ndarray
    predictions: list
    valid_prediction_indices: list


def _recording_config(recording):
    values = recording.metadata.get("config")
    if not isinstance(values, dict):
        raise ValueError("Personal recording is missing its acquisition configuration; import the original capture again")
    return Config.from_dict(values)


def extract_windows(recording, config=None):
    # 复用部署时的 Runtime 获取经过因果滤波和质量检查的确切模型输入。
    # 真值不传给占位模型；无效预测仍保留，供验证时重现连续确认的中断。
    config = config if config is not None else _recording_config(recording)
    windows, observed_keys, predictions = [], [], []

    def observe(window, prediction):
        windows.append(window.copy())
        observed_keys.append((prediction.trial_id, prediction.start_sample, prediction.stop_sample))

    runtime = Runtime(config, _WindowProbe(), model_version="training-window-probe", window_observer=observe)
    source = ReplaySource(recording, chunk_samples=config.chunk_samples, realtime=False, imagery_seconds=config.imagery_seconds)
    for packet in source:
        predictions.extend(p for p in runtime.process(packet) if isinstance(p, Prediction))
    positions = {(p.trial_id, p.start_sample, p.stop_sample): i for i, p in enumerate(predictions)}
    if len(positions) != len(predictions):
        raise ValueError("Runtime emitted duplicate scheduled windows")
    indices = [positions[key] for key in observed_keys]
    x = np.asarray(windows, dtype=float) if windows else np.empty((0, len(config.channels), config.window_samples))
    return ExtractedWindows(x, predictions, indices)


def make_pipeline(config):
    # CSP 学习空间滤波并提取对数平均功率；LDA 使用自动收缩估计完成二分类。
    return Pipeline([
        ("csp", CSP(n_components=config.n_components, log=True, norm_trace=False,
                    transform_into="average_power")),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])


def _fit_model(config, x, y):
    # 每折都新建并拟合 CSP/LDA，避免沿用其他折学习到的参数。
    if set(np.asarray(y).tolist()) != {0, 1}:
        raise ValueError("Every training fold needs valid LEFT and RIGHT windows")
    pipeline = make_pipeline(config)
    with mne.use_log_level("ERROR"):
        pipeline.fit(x, y)
    return pipeline


def _validation_commands(config, trials, predictions):
    # 按试次和窗口时间顺序重放判决规则，重现拒绝窗口引起的连续证据清零。
    by_trial = {}
    for prediction in predictions:
        by_trial.setdefault(prediction.trial_id, []).append(prediction)
    commands = []
    policy = DecisionPolicy(config)
    for trial in sorted(trials, key=lambda t: t.cue_sample):
        policy.reset(trial.trial_id)
        for prediction in sorted(by_trial.get(trial.trial_id, []), key=lambda p: p.stop_sample):
            commands.append(policy.decide(prediction))
        policy.end()
    return commands


def _recording_digest(recording):
    hasher = sha256()
    hasher.update(np.ascontiguousarray(recording.data).view(np.uint8))
    hasher.update(np.ascontiguousarray(recording.timestamps).view(np.uint8))
    hasher.update(canonical_json([asdict(t) for t in recording.trials]).encode("utf-8"))
    hasher.update(canonical_json(asdict(recording.info)).encode("utf-8"))
    hasher.update(canonical_json(list(recording.run_starts)).encode("utf-8"))
    return hasher.hexdigest()


class CalibrationDataError(ValueError):

    # 校准数据不足时保留详细审计报告，便于定位需要补采的项目。
    def __init__(self, report):
        self.report = report
        super().__init__("Calibration needs additional collection / 需要补采或调整采集：" + "; ".join(report["issues"]))


def audit_calibration(recording, extraction, config, *, exclude_artifacts=True,
                      min_trials_per_class_per_run=10, max_class_ratio=2.0,
                      max_rejected_trial_fraction=0.5, max_rejected_window_fraction=0.5):
    # 按独立试次审计采集数量和拒绝比例，重叠窗口不增加独立试次数。
    # 这些门限是工程最低要求，并不保证真人实验的识别效果。
    if (isinstance(min_trials_per_class_per_run, bool)
            or not isinstance(min_trials_per_class_per_run, Integral)
            or min_trials_per_class_per_run < 2):
        raise ValueError("min_trials_per_class_per_run must be an integer >= 2")
    if isinstance(max_class_ratio, bool) or not isinstance(max_class_ratio, Real) or not isfinite(max_class_ratio) or max_class_ratio < 1:
        raise ValueError("max_class_ratio must be finite and >= 1")
    for name, value in (("max_rejected_trial_fraction", max_rejected_trial_fraction),
                        ("max_rejected_window_fraction", max_rejected_window_fraction)):
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or not 0 <= value < 1:
            raise ValueError(f"{name} must be in [0, 1)")
    predictions_by_trial = defaultdict(list)
    for prediction in extraction.predictions:
        predictions_by_trial[prediction.trial_id].append(prediction)
    valid_counts = Counter(extraction.predictions[index].trial_id for index in extraction.valid_prediction_indices)
    expected_per_trial = ((config.samples(config.imagery_seconds - config.cue_offset_seconds)
                           - config.window_samples) // config.step_samples + 1)
    rows, issues = [], []
    for run_id in sorted({t.run_id for t in recording.trials}, key=str):
        trials = [t for t in recording.trials if t.run_id == run_id]
        target = {"LEFT": 0, "RIGHT": 0}
        valid = dict(target)
        fit = dict(target)
        reasons = Counter()
        valid_windows = 0
        missing_windows = 0
        for trial in trials:
            label = "LEFT" if trial.original_class == 1 else "RIGHT"
            target[label] += 1
            if valid_counts[trial.trial_id]:
                valid[label] += 1
                if not exclude_artifacts or not trial.artifact_flag:
                    fit[label] += 1
            valid_windows += valid_counts[trial.trial_id]
            trial_predictions = predictions_by_trial[trial.trial_id]
            reasons.update(p.quality for p in trial_predictions if p.quality != "OK")
            missing_windows += max(0, expected_per_trial - len(trial_predictions))
        if missing_windows:
            reasons["MISSING_SCHEDULED_WINDOW"] += missing_windows
        expected_windows = len(trials) * expected_per_trial
        rejected_trial_fraction = 1 - sum(fit.values()) / len(trials)
        rejected_window_fraction = 1 - valid_windows / expected_windows
        class_ratio = max(fit.values()) / min(fit.values()) if min(fit.values()) else None
        row = {
            "run_id": run_id, "target_trials_per_class": target,
            "quality_valid_trials_per_class": valid, "fit_trials_per_class": fit,
            "expert_flagged_trials": sum(bool(t.artifact_flag) for t in trials),
            "rejected_trial_fraction": rejected_trial_fraction,
            "expected_windows": expected_windows, "quality_valid_windows": valid_windows,
            "rejected_window_fraction": rejected_window_fraction,
            "missing_windows": missing_windows, "invalid_window_reasons": dict(reasons),
            "fit_class_ratio": class_ratio,
        }
        rows.append(row)
        for label, count in fit.items():
            if count < min_trials_per_class_per_run:
                issues.append(f"Run {run_id} {label}: {count} valid fit trials; collect at least {min_trials_per_class_per_run - count} more")
        if class_ratio is None or class_ratio > max_class_ratio:
            issues.append(f"Run {run_id}: valid LEFT/RIGHT trial imbalance exceeds {max_class_ratio:g}")
        if rejected_trial_fraction > max_rejected_trial_fraction:
            issues.append(f"Run {run_id}: rejected fit trials {rejected_trial_fraction:.1%} exceed {max_rejected_trial_fraction:.1%}")
        if rejected_window_fraction > max_rejected_window_fraction:
            issues.append(f"Run {run_id}: rejected/missing windows {rejected_window_fraction:.1%} exceed {max_rejected_window_fraction:.1%}")
    if len(rows) < 2:
        issues.append("At least two complete acquisition runs are required; collect another balanced run")
    return {
        "status": "READY_TO_TRAIN" if not issues else "NEEDS_MORE_COLLECTION",
        "limits": {"min_trials_per_class_per_run": int(min_trials_per_class_per_run),
                   "max_class_ratio": float(max_class_ratio),
                   "max_rejected_trial_fraction": float(max_rejected_trial_fraction),
                   "max_rejected_window_fraction": float(max_rejected_window_fraction)},
        "trial_count_note": "Independent trials, not overlapping windows. Minimum counts do not establish online readiness.",
        "window_rejection_note": "Quality-rejected and missing scheduled windows; expert flags separately exclude trials from fitting.",
        "runs": rows, "issues": issues,
    }


def train_recording(recording, config=None, *, thresholds=(0.6, 0.7, 0.8), confirmations=(2, 3), exclude_artifacts=True,
                    min_trials_per_class_per_run=10, max_class_ratio=2.0,
                    max_rejected_trial_fraction=0.5, max_rejected_window_fraction=0.5):
    # 仅用 T 会话拟合和选参；按整组 run 留出验证，避免重叠窗口跨折泄漏。
    # 人工伪迹标签只影响拟合样本选择，验证报告仍包含带伪迹标记的试次。
    acquisition_config = _recording_config(recording)
    config = config if config is not None else acquisition_config
    if recording.metadata.get("synthetic"):
        raise ValueError("Training requires recorded EEG; legacy synthetic recordings are not accepted")
    if recording.metadata.get("source") != "personal_lsl":
        raise ValueError("Training requires a personal LSL calibration recording")
    if recording.session_id != "T":
        raise ValueError("Training requires a T session; E data must never select or fit parameters")
    # 可以在校准阶段选择处理参数，但不能改写设备身份或已采集的想象时长。
    for key in ("sfreq", "channels", "reference", "unit", "imagery_seconds"):
        if getattr(config, key) != getattr(acquisition_config, key):
            raise ValueError(f"Training configuration must preserve recorded {key}")
    thresholds, confirmations = tuple(thresholds), tuple(confirmations)
    if not thresholds or not confirmations:
        raise ValueError("At least one threshold and confirmation count are required")
    policies = [replace(config, threshold=threshold, confirmations=count)
                for threshold in thresholds for count in confirmations]
    trials = list(recording.trials)
    if any(t.subject_id != recording.subject_id or t.session_id != recording.session_id for t in trials):
        raise ValueError("Training recording must contain exactly its declared subject and session")
    if any(isinstance(t.original_class, bool) or not isinstance(t.original_class, Integral)
           or t.original_class not in (1, 2) for t in trials):
        raise ValueError("Training requires personal truth labels 1=LEFT or 2=RIGHT")
    truth = {t.trial_id: t for t in trials}
    if len(truth) != len(trials):
        raise ValueError("Trial IDs must be unique")
    run_ids = sorted({t.run_id for t in trials}, key=str)
    for trial in trials:
        acquired_run = int(np.searchsorted(recording.run_starts, trial.cue_sample, side="right"))
        if trial.run_id != acquired_run or acquired_run == 0:
            raise ValueError("Trial run_id differs from acquisition RUN_START boundaries; re-import the original capture")
        run_end = recording.run_starts[acquired_run] if acquired_run < len(recording.run_starts) else recording.data.shape[1]
        if trial.cue_sample + config.samples(config.imagery_seconds) > run_end:
            raise ValueError("Trial imagery crosses an acquisition run boundary or has incomplete EEG")
    extraction = extract_windows(recording, config)
    audit = audit_calibration(recording, extraction, config, exclude_artifacts=exclude_artifacts,
                              min_trials_per_class_per_run=min_trials_per_class_per_run,
                              max_class_ratio=max_class_ratio,
                              max_rejected_trial_fraction=max_rejected_trial_fraction,
                              max_rejected_window_fraction=max_rejected_window_fraction)
    if audit["issues"]:
        raise CalibrationDataError(audit)
    valid_predictions = [extraction.predictions[i] for i in extraction.valid_prediction_indices]
    window_trials = [truth[p.trial_id] for p in valid_predictions]
    eligible = np.asarray([not exclude_artifacts or not t.artifact_flag
                           for t in window_trials], dtype=bool)
    if not eligible.any():
        raise ValueError("No quality-approved left/right training windows")
    # 原始标签 1/2 转为模型类别 0/1；同一试次的所有窗口保留所属 run。
    labels = np.asarray([t.original_class - 1 for t in window_trials])
    groups = np.asarray([t.run_id for t in window_trials])
    oof_predictions = list(extraction.predictions)
    fold_records = []
    # 每次留出一整组；CSP 和 LDA 只拟合其余组，再预测留出组的有效窗口。
    for run_id in run_ids:
        fit_indices = np.flatnonzero(eligible & (groups != run_id))
        held_indices = np.flatnonzero(groups == run_id)
        model = _fit_model(config, extraction.windows[fit_indices], labels[fit_indices])
        if len(held_indices):
            for index in held_indices:
                began = perf_counter()
                probs = model.predict_proba(extraction.windows[index:index + 1])[0]
                inference_seconds = perf_counter() - began
                position = extraction.valid_prediction_indices[index]
                oof_predictions[position] = replace(oof_predictions[position], probabilities=tuple(float(p) for p in probs),
                                                    model_version=f"T-held-out-run-{run_id}",
                                                    processing_seconds=inference_seconds + oof_predictions[position].processing_seconds)
        fit_trial_ids = sorted({window_trials[index].trial_id for index in fit_indices})
        held_trial_ids = [t.trial_id for t in trials if t.run_id == run_id]
        assert not set(fit_trial_ids).intersection(held_trial_ids)
        fold_records.append({
            "held_out_run": run_id, "training_runs": [r for r in run_ids if r != run_id],
            "training_trial_ids": fit_trial_ids, "validation_trial_ids": held_trial_ids,
            "training_windows": len(fit_indices), "validation_valid_windows": len(held_indices),
        })
    candidates = []
    for candidate in policies:
        commands = _validation_commands(candidate, trials, oof_predictions)
        metrics = evaluate(trials, oof_predictions, commands, **_evaluation_parameters(config))
        candidates.append({"threshold": candidate.threshold, "confirmations": candidate.confirmations,
                           "metrics": metrics["all"], "artifact_clean": metrics["artifact_clean"],
                           "artifact_flagged": metrics["artifact_flagged"]})

    def selection_key(candidate):
        metrics = candidate["metrics"]
        return (metrics["correct_command_fraction"] or 0, metrics["command_accuracy"] or 0,
                candidate["threshold"], candidate["confirmations"])

    # 使用校准集的折外预测选择阈值与确认次数；该分数不是独立测试成绩。
    selected = max(candidates, key=selection_key)
    frozen = replace(config, threshold=selected["threshold"], confirmations=selected["confirmations"])
    selected_commands = _validation_commands(frozen, trials, oof_predictions)
    selected_metrics = evaluate(trials, oof_predictions, selected_commands, **_evaluation_parameters(config))
    for fold in fold_records:
        fold_trials = [t for t in trials if t.run_id == fold["held_out_run"]]
        fold_ids = {t.trial_id for t in fold_trials}
        fold["metrics"] = evaluate(fold_trials, [p for p in oof_predictions if p.trial_id in fold_ids],
                                   [c for c in selected_commands if c.trial_id in fold_ids], **_evaluation_parameters(config))
    # 判决策略选定后冻结配置，再用全部合格校准窗口训练最终部署模型。
    final_model = _fit_model(frozen, extraction.windows[eligible], labels[eligible])
    source_hash = _recording_digest(recording)
    versions = dependency_versions()
    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_subject": recording.subject_id, "training_session": recording.session_id,
        "recording_metadata": dict(recording.metadata),
        "recording_sha256": source_hash, "dependencies": versions,
        "class_mapping": CLASS_MAPPING, "original_class_mapping": {"1": 0, "2": 1},
        "filter_initialization": "zero_state_per_continuous_segment",
        "warmup_seconds": frozen.warmup_seconds, "reference": frozen.reference,
        "training_artifact_policy": "exclude_expert_flagged" if exclude_artifacts else "include_expert_flagged",
        "training_windows": int(eligible.sum()),
        "calibration_data_limits": audit["limits"],
        "window_quality_limits": asdict(QualityThresholds.for_window(frozen)),
        "training_trial_ids": sorted({t.trial_id for t, good in zip(window_trials, eligible) if good}),
        "validation": "leave-one-run-out; all overlapping trial windows grouped by run",
        "independent_test_used": False,
    }
    report = {
        "training_subject": recording.subject_id, "training_session": recording.session_id,
        "data_scope": "Personal LSL calibration EEG, subject-specific training.",
        "validation_scope": "Internal T-session selection; the selected CV score is not an independent test estimate.",
        "window_extraction": "Exact deployed Runtime window observer, causal filtering, neutral replay markers.",
        "processing_time_scope": "CV measures individual model calls plus observed extraction-gate overhead; independent replay measures the actual deployed decoder.",
        "artifact_policy": metadata["training_artifact_policy"],
        "calibration_audit": audit,
        "target_trials": len(trials),
        "expert_flagged_target_trials": sum(t.artifact_flag for t in trials),
        "fit_windows": int(eligible.sum()), "all_valid_windows": len(extraction.windows),
        "invalid_window_count": sum(p.quality != "OK" for p in extraction.predictions),
        "folds": fold_records, "candidate_policies": candidates,
        "selection_rule": "Maximize correct commands / ALL target trials; tie-break by command accuracy, then higher threshold, then more confirmations.",
        "selected_policy": {"threshold": frozen.threshold, "confirmations": frozen.confirmations},
        "selected_validation_metrics": selected_metrics,
        "independent_test_used": False,
    }
    sos = CausalPreprocessor(frozen).sos.copy()
    model_version = "csp-lda-" + digest({"recording": source_hash, "config": frozen.to_dict(), "dependencies": versions,
                                        "window_quality_limits": metadata["window_quality_limits"],
                                        "artifact_policy": metadata["training_artifact_policy"]})[:16]
    bundle = ModelBundle(frozen, final_model, sos, model_version, metadata, report)
    bundle.validate()
    return bundle



# ========================================================================
# 15. 采集、解码、发布与会话收尾
# ========================================================================
def _finalize_capture(recorder, config, source, summary, summary_name, target_status,
                      primary_error, cleanup_errors, received_raw_samples):
    # 原始文件关闭、完整性检查及必要报告写入后，才提交会话完成状态。
    directory = recorder.directory
    try:
        recorder.close(status="finalizing", error=primary_error or (cleanup_errors[0] if cleanup_errors else None),
                       received_raw_samples=received_raw_samples)
    except BaseException as exc:
        cleanup_errors.append(exc)
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except BaseException as exc:
        cleanup_errors.append(exc)
        manifest = {"format": "mi-bci-session-v1", "stream": recorder.info,
                    "raw_samples": recorder.raw_samples, "accepted_raw_samples": recorder.accepted_raw_samples,
                    "processed_raw_samples": recorder.processed_raw_samples,
                    "received_raw_samples": received_raw_samples}

    def current_status():
        return "failed" if primary_error is not None or recorder.error is not None or cleanup_errors else target_status

    def error_message():
        original = primary_error or recorder.error or (cleanup_errors[0] if cleanup_errors else None)
        return str(original) if original is not None else None

    def update_summary():
        summary.update({"status": current_status(), "error": error_message(),
                        "cleanup_errors": [str(exc) for exc in cleanup_errors],
                        "raw_samples": recorder.raw_samples, "received_raw_samples": received_raw_samples,
                        "accepted_raw_samples": recorder.accepted_raw_samples,
                        "persisted_raw_samples": recorder.raw_samples,
                        "processed_raw_samples": recorder.processed_raw_samples,
                        "unconfirmed_processed_samples": max(0, recorder.processed_raw_samples - recorder.raw_samples),
                        "failed_to_retain_samples": max(0, received_raw_samples - recorder.accepted_raw_samples),
                        "unwritten_accepted_samples": max(0, recorder.accepted_raw_samples - recorder.raw_samples),
                        "unconfirmed_accepted_samples": max(0, recorder.accepted_raw_samples - recorder.raw_samples)})

    def write_final_manifest():
        manifest.update({"status": current_status(),
                         "error": error_message(),
                         "cleanup_errors": [str(exc) for exc in cleanup_errors],
                         "finalization_complete": not cleanup_errors})
        write_json(directory / "manifest.json", manifest)

    manifest.update({"status": "finalizing" if current_status() != "failed" else "failed",
                     "finalization_target_status": current_status(), "finalization_complete": False,
                     "error": error_message()})
    try:
        write_json(directory / "manifest.json", manifest)
    except BaseException as exc:
        cleanup_errors.append(exc)

    diagnostics = {}
    for filename, create in (
            ("channel_quality.json", lambda: write_capture_quality(directory, config)),
            ("alignment_report.json", lambda: write_capture_timing(directory, config, getattr(source, "timing_records", None)))):
        try:
            report = create()
            diagnostics[filename] = report.get("status")
            if report.get("report_write_error"):
                details = report["report_write_error"]
                raise RuntimeError(f"Failed to save {filename}: {details.get('type')}: {details.get('message')}")
        except BaseException as exc:
            cleanup_errors.append(exc)
            diagnostics[filename] = "write_failed"
    summary["diagnostic_reports"] = diagnostics
    try:
        write_json(directory / "marker_transport.json", getattr(source, "timing_records", []))
    except BaseException as exc:
        cleanup_errors.append(exc)
    update_summary()
    try:
        write_json(directory / summary_name, summary)
    except BaseException as exc:
        cleanup_errors.append(exc)
        update_summary()
        try:
            write_json(directory / summary_name, summary)
        except BaseException as retry_exc:
            cleanup_errors.append(retry_exc)
            update_summary()
    try:
        write_final_manifest()
    except BaseException as exc:
        cleanup_errors.append(exc)
        update_summary()
        try:
            write_json(directory / summary_name, summary)
        except BaseException as repair_exc:
            cleanup_errors.append(repair_exc)
            update_summary()
        try:
            write_final_manifest()
        except BaseException as repair_exc:
            cleanup_errors.append(repair_exc)
            update_summary()
    return summary


def record_session(source, config, output, observer=None, stop_event=None, context=None):
    # 校准采集不加载分类器；原始 EEG 即使被质量检查拒绝也仍需保留。
    stop = stop_event if stop_event is not None else threading.Event()
    recorder = None
    iterator = None
    caught = None
    cancelled = stop.is_set()
    quality_reasons = Counter()
    counts = {"chunks": 0, "samples": 0, "valid_samples": 0, "markers": 0, "runs": 0,
              "received_raw_samples": 0}
    began = time.perf_counter()
    summary = None
    protocol = MarkerProtocol()
    saw_session_end = False
    try:
        preprocessor = CausalPreprocessor(config)
        recorder = AsyncRecorder(output, source.info)
        capture_context = {
            "format": "mi-bci-capture-context-v1", "mode": "calibration",
            **dict(context or {}), "config": config.to_dict(),
        }
        write_json(recorder.directory / "capture_context.json", capture_context)
        iterator = iter(source)
        while not stop.is_set():
            try:
                packet = next(iterator)
            except StopIteration:
                if not saw_session_end and not stop.is_set():
                    raise RuntimeError("Acquisition ended without SESSION_END; capture is incomplete")
                break
            except Exception:
                if stop.is_set():
                    cancelled = True
                    break
                raise
            if isinstance(packet, EEGChunk):
                counts["received_raw_samples"] += len(packet.timestamps)
            try:
                recorder.record(packet)
            except RuntimeError:
                if recorder.error is None:
                    recorder.record(packet, timeout=5)
                raise
            if stop.is_set():
                break
            if isinstance(packet, EEGChunk):
                filtered = preprocessor.process(packet)
                counts["chunks"] += 1
                counts["samples"] += packet.data.shape[1]
                counts["valid_samples"] += int(filtered.valid.sum())
                quality_reasons.update(filtered.reasons.tolist())
            elif isinstance(packet, Marker):
                if packet.kind not in NEUTRAL_KINDS:
                    raise ValueError("Calibration recording accepts only neutral phase markers")
                protocol.observe(packet.kind, packet.trial_id, packet.timestamp)
                if packet.kind == "DISCONNECT":
                    raise RuntimeError("Experiment or EEG disconnected before completing the session")
                if packet.kind == "SESSION_END":
                    saw_session_end = True
                counts["markers"] += 1
                if packet.kind in {"RUN_START", "DISCONNECT"}:
                    preprocessor.reset()
                if packet.kind == "RUN_START":
                    counts["runs"] += 1
            else:
                raise TypeError("Calibration source must emit EEGChunk or neutral Marker")
            if observer:
                observer(packet)
        cancelled = stop.is_set()
    except KeyboardInterrupt:
        cancelled = True
        raise
    except BaseException as exc:
        caught = exc
        raise
    finally:
        cleanup_errors = []
        try:
            source.close()
        except BaseException as exc:
            cleanup_errors.append(exc)
        try:
            if iterator is not None and hasattr(iterator, "close"):
                iterator.close()
        except BaseException as exc:
            cleanup_errors.append(exc)
        if recorder is not None:
            counts["received_raw_samples"] += sum(len(p.timestamps) for p in getattr(source, "pending_raw", []))
            try:
                for packet in getattr(source, "pending_raw", []):
                    recorder.record(packet, timeout=5)
                    counts["samples"] += len(packet.timestamps)
                if hasattr(source, "pending_raw"):
                    source.pending_raw.clear()
            except BaseException as exc:
                cleanup_errors.append(exc)
            status = "failed" if caught or cleanup_errors else "cancelled" if cancelled else "complete"
            summary = {
                "mode": "calibration", "status": status, **counts,
                "invalid_samples": counts["samples"] - counts["valid_samples"],
                "sample_quality_reasons": dict(quality_reasons),
                "quality_scope": "Causal sample checks only; no classifier or trial-level flat-signal check.",
                "elapsed_wall_seconds": time.perf_counter() - began,
                "cleanup_errors": [str(exc) for exc in cleanup_errors],
            }
            try:
                recorder.record({"type": "SessionEnd", "error": str(caught) if caught else None,
                                 "cancelled": cancelled,
                                 "cleanup_errors": [str(exc) for exc in cleanup_errors]})
            except BaseException as exc:
                cleanup_errors.append(exc)
            _finalize_capture(recorder, config, source, summary, "acquisition_summary.json", status,
                              caught, cleanup_errors, counts["received_raw_samples"])
        if (caught is None and cleanup_errors
                and not (cancelled and isinstance(sys.exc_info()[1], KeyboardInterrupt))):
            raise RuntimeError("Calibration recording cleanup failed") from cleanup_errors[0]
    return summary


def run_session(source, bundle, output, observer=None, stop_event=None, queue_capacity=64, context=None,
                result_stream_name="MI_Results"):
    # 采集、写盘与解码通过有界队列协作，在线结果在发布前再次检查时效。
    if queue_capacity < 1:
        raise ValueError("Acquisition queue capacity must be positive")
    stop = stop_event if stop_event is not None else threading.Event()
    live = getattr(source, "is_live", False)
    safety = getattr(source, "safety", None)
    if live and safety is None:
        safety = LiveSafety(bundle.config, source.clock, stop)
    publisher = None
    try:
        if context is not None:
            expected_mode = "live" if live else "replay"
            if context.get("mode") != expected_mode:
                raise ValueError("Capture context mode differs from the actual source mode")
        def guard(trial_id, timestamp):
            if stop.is_set():
                return "STOPPED"
            if recorder.error:
                return "RECORDER_FAILED"
            return safety.reason(trial_id, timestamp)
        decoder = Runtime(bundle.config, bundle.model, sos=bundle.sos, model_version=bundle.version,
                          validity_guard=guard if live else None)
        recorder = AsyncRecorder(output, source.info)
        if context is not None:
            write_json(recorder.directory / "capture_context.json", context)
        publisher = ResultPublisher(recorder.directory, bundle.version, live=live,
                                    clock=source.clock if live else time.monotonic,
                                    stream_name=result_stream_name, ttl=bundle.config.max_age_seconds)
        publisher.publish({"direction": "NO_COMMAND", "reason": "SESSION_STARTED"})
    except BaseException as exc:
        source.close()
        if publisher is not None:
            publisher.close()
        if "recorder" in locals():
            recorder.close(status="failed", error=exc)
        raise
    incoming = queue.Queue(maxsize=queue_capacity)
    done = threading.Event()
    failure = []
    predictions, commands = [], []
    max_backlog = 0
    retained_after_stop = 0
    received_raw_samples = 0
    saw_session_end = False

    def retain(packet):
        try:
            recorder.record(packet, timeout=0 if live else 5)
        except RuntimeError:
            if safety is not None:
                safety.fail("RECORDER_FAILED")
            stop.set()
            if recorder.error is None:
                recorder.record(packet, timeout=5)
            raise

    def acquire():
        nonlocal received_raw_samples, saw_session_end
        iterator = None
        try:
            iterator = iter(source)
            for packet in iterator:
                if isinstance(packet, EEGChunk):
                    received_raw_samples += len(packet.timestamps)
                # 原始记录先于推断入队，解码队列溢出时已接收数据仍有记录。
                retain(packet)
                if isinstance(packet, Marker) and packet.kind == "SESSION_END":
                    saw_session_end = True
                if live and not hasattr(source, "safety"):
                    if isinstance(packet, EEGChunk) and len(packet.timestamps):
                        safety.eeg(packet.timestamps[-1])
                    elif isinstance(packet, Marker):
                        safety.marker(packet.kind, packet.trial_id, packet.timestamp)
                if stop.is_set():
                    break
                if live:
                    incoming.put_nowait(packet)
                else:
                    while not stop.is_set():
                        try:
                            incoming.put(packet, timeout=0.1)
                            break
                        except queue.Full:
                            continue
            if live and not stop.is_set() and not saw_session_end:
                raise RuntimeError("Live acquisition ended without SESSION_END")
        except BaseException as exc:
            failure.append(exc)
            if safety is not None:
                safety.fail("ACQUISITION_FAILED")
            stop.set()
        finally:
            try:
                if iterator is not None and hasattr(iterator, "close"):
                    iterator.close()
                source.close()
            except BaseException as exc:
                failure.append(exc)
                stop.set()
            try:
                # 补存标记等待区中尚未交付的样本，退出恢复过程中不再进行模型推断。
                received_raw_samples += sum(len(p.timestamps) for p in getattr(source, "pending_raw", []))
                for packet in getattr(source, "pending_raw", []):
                    recorder.record(packet, timeout=5)
                if hasattr(source, "pending_raw"):
                    source.pending_raw.clear()
            except BaseException as exc:
                failure.append(exc)
                stop.set()
            done.set()

    def emit(result):
        def prepare(command):
            now = source.clock()
            deadline = now + bundle.config.max_age_seconds
            if command.direction != "NO_COMMAND":
                reason = guard(command.trial_id, command.timestamp)
                if reason:
                    command = replace(command, direction="NO_COMMAND", reason=reason)
                    decoder.decision.clear_streak()
                else:
                    deadline = min(command.timestamp + bundle.config.max_age_seconds,
                                   safety.cue_timestamp + bundle.config.imagery_seconds)
            command = replace(command, emitted_at=now,
                              result_age_seconds=now - command.timestamp,
                              cue_to_output_seconds=(now - safety.cue_timestamp)
                              if safety.cue_timestamp is not None and command.direction != "NO_COMMAND" else None)
            return command, deadline

        if isinstance(result, Command) and live:
            # 磁盘准备工作在安全锁外完成；重新检查状态与 LSL 发送在同一锁内执行。
            row = publisher.publish(result, prepare=prepare, dispatch_lock=safety.lock)
            result = Command(**row["payload"])
        else:
            publisher.publish(result)
        recorder.record(result)
        if isinstance(result, Prediction):
            predictions.append(result)
        elif isinstance(result, Command):
            commands.append(result)
        if observer:
            observer(result)

    worker = threading.Thread(target=acquire, name="mi-acquisition", daemon=True)
    worker.start()
    caught = None
    start_clock = time.perf_counter()
    try:
        while not stop.is_set() and (not done.is_set() or not incoming.empty()):
            max_backlog = max(max_backlog, incoming.qsize())
            try:
                packet = incoming.get(timeout=0.1)
            except queue.Empty:
                continue
            if observer:
                observer(packet)
            if live and isinstance(packet, EEGChunk):
                packet = replace(packet, received_at=source.clock())
            for result in decoder.process(packet):
                emit(result)
        if failure:
            raise RuntimeError("Acquisition failed or queue overflowed") from failure[0]
        if not stop.is_set():
            for result in decoder.finish():
                emit(result)
    except BaseException as exc:
        caught = exc
        raise
    finally:
        was_stopped = stop.is_set() or isinstance(caught, KeyboardInterrupt)
        stop.set()
        cleanup_errors = []
        try:
            source.close()
        except BaseException as exc:
            cleanup_errors.append(exc)
        worker.join(timeout=6)
        if worker.is_alive():
            cleanup_errors.append(RuntimeError("Acquisition thread did not stop within six seconds"))
        # 这些待推断包已交给原始记录器，停止后清空解码队列即可。
        retained_after_stop = incoming.qsize()
        while not incoming.empty():
            incoming.get_nowait()
        terminal_reason = "SESSION_FAILED" if caught or failure or cleanup_errors else "STOPPED" if was_stopped else "SESSION_ENDED"
        try:
            terminal = Command(None, "NO_COMMAND", (decoder._last_sample or 0) + 1,
                               source.clock() if live else time.monotonic(), terminal_reason)
            emit(terminal)
        except BaseException as exc:
            cleanup_errors.append(exc)
        try:
            publisher.close()
        except BaseException as exc:
            cleanup_errors.append(exc)
        try:
            recorder.record({"type": "SessionEnd", "error": str(caught or (failure[0] if failure else "")) or None,
                             "cause": str(caught.__cause__) if caught and caught.__cause__ else None,
                             "cancelled": was_stopped, "pending_acquisition_packets": 0,
                             "retained_without_inference_packets": retained_after_stop,
                             "received_raw_samples": received_raw_samples,
                             "failed_to_retain_samples": max(0, received_raw_samples - recorder.accepted_raw_samples),
                             "cleanup_errors": [str(e) for e in cleanup_errors]}, timeout=5)
        except BaseException as exc:
            cleanup_errors.append(exc)
        summary = {
            "predictions": len(predictions), "commands": len([c for c in commands if c.direction != "NO_COMMAND"]),
            "max_acquisition_backlog": max_backlog, "elapsed_wall_seconds": time.perf_counter() - start_clock,
            "mode": "live" if live else "replay", "model_version": bundle.version,
            "validation_scope": "Live hardware requires separate calibration and validation" if live else "Continuous replay only; not a live closed-loop experiment",
            "pending_acquisition_packets": 0, "retained_without_inference_packets": retained_after_stop,
            "received_raw_samples": received_raw_samples, "persisted_raw_samples": recorder.raw_samples,
            "failed_to_retain_samples": max(0, received_raw_samples - recorder.accepted_raw_samples),
            "unwritten_accepted_samples": max(0, recorder.accepted_raw_samples - recorder.raw_samples),
            "acquisition_thread_still_running": worker.is_alive(),
            "session_end_observed": saw_session_end,
            "cleanup_errors": [str(e) for e in cleanup_errors], "decoder_statistics": decoder.stats,
            "latency_budget": {"first_window_seconds": bundle.config.cue_offset_seconds + bundle.config.window_seconds,
                               "earliest_confirmation_seconds": bundle.config.cue_offset_seconds + bundle.config.window_seconds
                               + (bundle.config.confirmations - 1) * bundle.config.step_seconds,
                               "phase_deadline_seconds": bundle.config.imagery_seconds,
                               "max_result_age_seconds": bundle.config.max_age_seconds},
        }
        _finalize_capture(recorder, bundle.config, source, summary, "runtime_summary.json",
                          "cancelled" if was_stopped else "complete", caught or (failure[0] if failure else None),
                          cleanup_errors, received_raw_samples)
        if caught is None and (cleanup_errors or failure):
            raise RuntimeError("Session cleanup failed") from (cleanup_errors or failure)[0]
    return predictions, commands


# ========================================================================
# 16. 采集线程的实时状态与标记协议
# ========================================================================
class MarkerProtocol:

    # 验证组与试次阶段的顺序，拒绝缺失、重复、乱序和复用的试次 ID。
    def __init__(self):
        self.run = False
        self.trial = None
        self.phase = "WAITING"
        self.seen = set()
        self.last_stamp = None

    def observe(self, kind, trial_id, timestamp):
        if (isinstance(timestamp, bool) or not isinstance(timestamp, Real) or not math.isfinite(timestamp)
                or (self.last_stamp is not None and timestamp < self.last_stamp)):
            raise ValueError("Marker timestamps are nonfinite or out of order")
        if self.phase == "ENDED":
            raise ValueError("Marker arrived after session end")
        if kind == "DISCONNECT":
            self.phase, self.trial = "ENDED", None
        elif kind == "RUN_START":
            if self.trial is not None or self.phase == "RUN_START":
                raise ValueError("RUN_START interrupts a trial or repeats an empty run")
            self.run, self.phase = True, "RUN_START"
        elif kind == "TRIAL_START":
            if not self.run or self.trial is not None or not trial_id or trial_id in self.seen:
                raise ValueError("Missing phase or duplicate trial ID before TRIAL_START")
            self.seen.add(trial_id)
            self.trial, self.phase = trial_id, kind
        elif kind == "MI_START":
            if self.phase != "TRIAL_START" or trial_id != self.trial:
                raise ValueError("MI_START requires its unique matching TRIAL_START")
            self.phase = kind
        elif kind == "MI_END":
            if self.phase != "MI_START" or trial_id != self.trial:
                raise ValueError("MI_END requires its unique matching MI_START")
            self.trial, self.phase = None, kind
        elif kind == "SESSION_END":
            if self.trial is not None or self.phase != "MI_END":
                raise ValueError("SESSION_END requires complete trials and a nonempty final run")
            self.phase = "ENDED"
        else:
            raise ValueError("Unrecognized neutral phase")
        self.last_stamp = timestamp


class LiveSafety:
    # 保存采集线程直接更新的状态；解码及发布阶段在锁内检查其有效性。
    def __init__(self, config, clock, stop_event):
        self.config, self.clock, self.stop = config, clock, stop_event
        self.lock = threading.RLock()
        self.protocol = MarkerProtocol()
        self.latest_eeg = None
        self.cue_timestamp = None
        self.failure = None

    def eeg(self, last_timestamp):
        with self.lock:
            self.latest_eeg = float(last_timestamp)

    def marker(self, kind, trial_id, timestamp):
        with self.lock:
            try:
                self.protocol.observe(kind, trial_id, timestamp)
            except ValueError:
                self.failure = "MARKER_PROTOCOL_ERROR"
                raise
            if kind == "MI_START":
                self.cue_timestamp = timestamp

    def fail(self, reason):
        with self.lock:
            self.failure = reason

    def reason(self, trial_id, window_timestamp):
        with self.lock:
            now = self.clock()
            if self.failure:
                return self.failure
            if self.stop.is_set():
                return "STOPPED"
            if self.protocol.phase == "ENDED":
                return "SESSION_ENDED"
            if self.latest_eeg is None or now - self.latest_eeg > self.config.max_age_seconds:
                return "SIGNAL_LOST"
            if now - window_timestamp > self.config.max_age_seconds:
                return "STALE_RESULT"
            if window_timestamp - now > self.config.timestamp_tolerance_seconds:
                return "CLOCK_MISMATCH"
            if self.protocol.phase != "MI_START" or self.protocol.trial != trial_id:
                return "OUTSIDE_MI_PHASE"
            if self.cue_timestamp is None or now >= self.cue_timestamp + self.config.imagery_seconds:
                return "MI_DEADLINE_EXCEEDED"
            return None


SCHEMA = "mi-results-v1"


# ========================================================================
# 17. 带有效期的结果发布与订阅
# ========================================================================
class ResultPublisher:
    # 为结果附加会话 ID、递增序号、模型版本、运行模式及有效期。
    def __init__(self, directory, model_version, *, live=False, clock=time.monotonic,
                 stream_name="MI_Results", ttl=0.5, outlet=None):
        self.clock, self.ttl = clock, ttl
        self.model_version, self.live = model_version, live
        self.session_id = str(uuid.uuid4())
        self.sequence = 0
        self.lock = threading.Lock()
        self.outlet = outlet
        if live and outlet is None:
            import pylsl
            info = pylsl.StreamInfo(stream_name, "MI_Results", 1, 0, "string", self.session_id)
            info.desc().append_child_value("schema", SCHEMA)
            self.outlet = pylsl.StreamOutlet(info)
        self.stream = (Path(directory) / "results.jsonl").open("x", encoding="utf-8")

    def publish(self, result, *, valid_for=None, prepare=None, dispatch_lock=None):
        with self.lock:
            def envelope(item, kind_override=None, deadline=None):
                now = self.clock()
                self.sequence += 1
                payload = asdict(item) if is_dataclass(item) else dict(item)
                kind = type(item).__name__ if is_dataclass(item) else payload.pop("type", "Status")
                lifetime = min(self.ttl, valid_for if valid_for is not None else self.ttl)
                if deadline is not None:
                    lifetime = min(lifetime, deadline - now)
                if kind == "Command" and payload.get("direction") in {"LEFT", "RIGHT"} and lifetime <= 0:
                    payload.update(direction="NO_COMMAND", reason="EXPIRED_BEFORE_SEND")
                return {"schema": SCHEMA, "session_id": self.session_id, "sequence": self.sequence,
                        "model_version": self.model_version, "mode": "live" if self.live else "replay",
                        "type": kind_override or kind, "issued_at": now,
                        "valid_for_seconds": max(0.0, lifetime), "payload": payload}

            def log(row):
                self.stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                self.stream.flush()

            # PreparedCommand 仅写审计日志；磁盘操作结束后重新确认状态，才发送最终指令。
            if prepare is not None:
                log(envelope(result, "PreparedCommand"))
                with dispatch_lock if dispatch_lock is not None else nullcontext():
                    result, deadline = prepare(result)
                    row = envelope(result, deadline=deadline)
                    if self.outlet is not None:
                        self.outlet.push_sample([json.dumps(row, ensure_ascii=False, allow_nan=False)],
                                                timestamp=row["issued_at"])
                log(row)
            else:
                row = envelope(result)
                log(row)
                if self.outlet is not None:
                    self.outlet.push_sample([json.dumps(row, ensure_ascii=False, allow_nan=False)],
                                            timestamp=row["issued_at"])
            return row

    def close(self):
        self.stream.close()


class ResultConsumer:

    # 订阅端独立验证会话、序号、时钟与有效期，并阻止同一试次重复执行。
    def __init__(self, timeout=0.5):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Subscription timeout must be positive and finite")
        self.timeout = timeout
        self.session_id = None
        self.sequence = 0
        self.seen_trials = set()
        self.retired_sessions = set()
        self.deadline = None

    def accept(self, row, corrected_lsl_timestamp, now):
        if row.get("schema") != SCHEMA or row.get("mode") != "live":
            raise ValueError("Unsupported result schema or non-live source")
        session = row.get("session_id")
        seq = row.get("sequence")
        if not isinstance(session, str) or not session or isinstance(seq, bool) or not isinstance(seq, int) or seq < 1:
            raise ValueError("Invalid result identity")
        if session in self.retired_sessions:
            return {"direction": "NO_COMMAND", "reason": "OLD_SESSION"}
        if session != self.session_id:
            if self.session_id is not None:
                self.retired_sessions.add(self.session_id)
            self.session_id, self.sequence = session, 0
            self.seen_trials.clear()
        if seq <= self.sequence:
            return {"direction": "NO_COMMAND", "reason": "DUPLICATE_RESULT"}
        self.sequence = seq
        ttl = row.get("valid_for_seconds")
        if isinstance(ttl, bool) or not isinstance(ttl, (int, float)) or not math.isfinite(ttl) or ttl < 0:
            raise ValueError("Invalid result lifetime")
        if not math.isfinite(corrected_lsl_timestamp) or not math.isfinite(now):
            raise ValueError("Invalid clock")
        deadline = corrected_lsl_timestamp + min(ttl, self.timeout)
        if now >= deadline or corrected_lsl_timestamp - now > 0.002:
            return {"direction": "NO_COMMAND", "reason": "EXPIRED_RESULT"}
        payload = row.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("Invalid result payload")
        # Prediction 仅用于观测，不更新方向指令的有效期。
        if row.get("type") not in {"Command", "Status"}:
            return None
        direction = payload.get("direction", "NO_COMMAND")
        if direction not in {"LEFT", "RIGHT", "NO_COMMAND"}:
            raise ValueError("Invalid direction")
        if row.get("type") != "Command" and direction != "NO_COMMAND":
            raise ValueError("Only a Command can carry a direction")
        if direction != "NO_COMMAND":
            trial_id = payload.get("trial_id")
            if not isinstance(trial_id, str) or not trial_id:
                raise ValueError("Direction requires a trial ID")
            if trial_id in self.seen_trials:
                return {"direction": "NO_COMMAND", "reason": "DUPLICATE_TRIAL"}
            self.seen_trials.add(trial_id)
        self.deadline = deadline if direction != "NO_COMMAND" else None
        return payload

    def watchdog(self, now):
        if self.deadline is not None and now >= self.deadline:
            self.deadline = None
            return {"direction": "NO_COMMAND", "reason": "OUTPUT_EXPIRED"}
        return None


def subscribe(args):
    import pylsl
    consumer = ResultConsumer(args.timeout)
    streams = pylsl.resolve_byprop("name", args.stream_name, timeout=5)
    if len(streams) != 1:
        raise ValueError("Exactly one result stream must match")
    inlet = pylsl.StreamInlet(streams[0], recover=False, max_buflen=1,
                             processing_flags=pylsl.proc_clocksync)
    def emit(payload):
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)
    try:
        header = inlet.info(timeout=5)
        if header.channel_count() != 1 or header.channel_format() != pylsl.cf_string:
            raise ValueError("Result stream must contain one JSON string channel")
        emit({"direction": "NO_COMMAND", "reason": "SUBSCRIBED"})
        while True:
            sample, stamp = inlet.pull_sample(timeout=min(0.05, args.timeout))
            now = pylsl.local_clock()
            if sample:
                payload = consumer.accept(json.loads(sample[0]), stamp, now)
                if payload is not None:
                    emit(payload)
            expired = consumer.watchdog(now)
            if expired:
                emit(expired)
    finally:
        emit({"direction": "NO_COMMAND", "reason": "SUBSCRIBER_STOPPED"})
        inlet.close_stream()


TIMING_LIMITATION = (
    "时间戳在终端写入并 flush 返回后立即取 LSL local_clock；"
    "它不等于已测量的视觉 onset。显示延迟及 EEG 对齐误差须用共同采集的外部触发/光电信号验证。"
)


# ========================================================================
# 18. 终端提示、随机平衡试次与独立标签日志
# ========================================================================
@dataclass(frozen=True)
class ExperimentConfig:
    # 终端实验的组数、试次数、各阶段时长及随机种子配置。
    subject_id: str
    session_id: str = "T"
    runs: int = 2
    trials_per_run: int = 20
    seed: int = 1
    imagery_seconds: float = 4.0
    rest_seconds: float = 2.0
    inter_trial_seconds: float = 2.0
    run_rest_seconds: float = 3.0
    marker_name: str = "MI_Markers"
    consumer_timeout_seconds: float = 120.0
    drain_seconds: float = 0.5
    acquisition_config_sha256: str | None = None

    def __post_init__(self):
        if not isinstance(self.subject_id, str) or not self.subject_id.strip():
            raise ValueError("subject_id 必须为非空字符串")
        if self.session_id not in {"T", "E"}:
            raise ValueError("session_id 必须为 T（校准）或 E（独立测试）")
        for name in ("runs", "trials_per_run", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} 必须为整数")
        if self.runs < (2 if self.session_id == "T" else 1):
            raise ValueError("校准至少需要两组；独立测试至少需要一组")
        if self.trials_per_run < 2 or self.trials_per_run % 2:
            raise ValueError("trials_per_run 必须为至少 2 的偶数，确保每组左右手平衡")
        for name in ("imagery_seconds", "rest_seconds", "inter_trial_seconds",
                     "run_rest_seconds", "consumer_timeout_seconds", "drain_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须为有限正数")
        if not isinstance(self.marker_name, str) or not self.marker_name.strip():
            raise ValueError("marker_name 必须为非空字符串")


def make_schedule(config, experiment_id):
    # 相同种子复现类别顺序，独立 UUID 避免不同实验复用试次标识。
    rng = random.Random(config.seed)
    schedule = []
    for run in range(1, config.runs + 1):
        classes = [1, 2] * (config.trials_per_run // 2)
        rng.shuffle(classes)
        for index, original_class in enumerate(classes, 1):
            schedule.append({"run_id": run, "trial_index": index,
                             "trial_id": f"{experiment_id}:r{run:03d}:t{index:04d}",
                             "original_class": original_class})
    return schedule


def terminal_presenter(message):
    sys.stdout.write("\n" + message + "\n")
    sys.stdout.flush()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _save_manifest(directory, manifest):
    temporary = directory / "manifest.json.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(_json(manifest) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(directory / "manifest.json")


def _append_row(handle, row):
    handle.write(_json(row) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _native_outlet(config, experiment_id):
    import pylsl
    info = pylsl.StreamInfo(config.marker_name, "Markers", 1, 0,
                            pylsl.cf_string, "mi-experiment-" + experiment_id)
    info.desc().append_child_value("schema", "neutral-kind-trial_id-v1")
    info.desc().append_child_value("timestamp_semantics", "local_clock after terminal flush")
    return pylsl.StreamOutlet(info, chunk_size=1), pylsl.local_clock


def run_experiment(config, output, *, outlet=None, clock=None, sleep=time.sleep,
                   presenter=terminal_presenter):
    # 向 LSL 发送中性阶段标记，将类别保存到独立标签日志。
    # 注入测试依赖时，outlet 与 LSL 时钟须同时提供；失败会保留未完成状态。
    if (outlet is None) != (clock is None):
        raise ValueError("outlet 和 clock 必须一起注入")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    experiment_id = str(uuid4())
    schedule = make_schedule(config, experiment_id)
    manifest = {"format": "mi-experiment-v1", "status": "starting",
                "experiment_id": experiment_id, "config": asdict(config),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "schedule": schedule, "timing_limitation": TIMING_LIMITATION,
                "labels_file": "labels.jsonl", "completed_trials": 0,
                "emitted_markers": 0, "error": None, "cleanup_errors": []}
    _save_manifest(output, manifest)
    active_trial = None
    active_run = None
    previous_timestamp = -float("inf")
    labels = None

    def checked_clock():
        stamp = float(clock())
        if not isfinite(stamp):
            raise RuntimeError("LSL 时钟返回了非有限时间戳")
        return stamp

    def wait_until(deadline):
        while True:
            if not outlet.have_consumers():
                raise RuntimeError("标记接收端已断开；实验中止")
            remaining = deadline - checked_clock()
            if remaining <= 0:
                return
            sleep(min(remaining, 0.1))

    def emit(kind, message, *, trial=None, original_class=None, check_consumer=True):
        nonlocal previous_timestamp
        if check_consumer and not outlet.have_consumers():
            raise RuntimeError("标记接收端已断开；实验中止")
        presenter(message)
        # 提示输出返回后立即取时钟，减少日志等额外操作带来的时间偏差。
        stamp = checked_clock()
        if stamp < previous_timestamp:
            raise RuntimeError("LSL 源时钟倒退；实验中止")
        previous_timestamp = stamp
        marker = {"kind": kind, "trial_id": trial}
        send_start = checked_clock()
        outlet.push_sample([_json(marker)], timestamp=stamp, pushthrough=True)
        send_end = checked_clock()
        if send_start < stamp or send_end < send_start:
            raise RuntimeError("LSL 源时钟倒退；实验中止")
        manifest["emitted_markers"] += 1
        row = {**marker, "timestamp": stamp, "run_id": active_run,
               "marker_send_start": send_start, "marker_send_end": send_end,
               "presentation_clock": "pylsl.local_clock_after_terminal_flush"}
        if original_class is not None:
            row["original_class"] = original_class
        # 类别只写入本地真值日志；若写入失败则中止，避免把不完整实验标为完成。
        _append_row(labels, row)
        return stamp

    try:
        labels = (output / "labels.jsonl").open("x", encoding="utf-8")
        if outlet is None:
            outlet, clock = _native_outlet(config, experiment_id)
        presenter(f"{TIMING_LIMITATION}\n已创建 {config.marker_name}。请在另一终端启动 record/live；等待标记接收端连接。")
        if not outlet.wait_for_consumers(config.consumer_timeout_seconds):
            raise RuntimeError("等待标记接收端超时；未开始试次")
        manifest["status"] = "running"
        _save_manifest(output, manifest)
        for trial in schedule:
            run = trial["run_id"]
            if run != active_run:
                active_run = run
                onset = emit("RUN_START", f"第 {run}/{config.runs} 组：保持放松，等待开始。")
                wait_until(onset + config.run_rest_seconds)
            active_trial = trial["trial_id"]
            onset = emit("TRIAL_START", f"第 {run} 组，第 {trial['trial_index']}/{config.trials_per_run} 次：休息，放松双手。",
                         trial=active_trial)
            wait_until(onset + config.rest_seconds)
            direction = "左手 ←" if trial["original_class"] == 1 else "右手 →"
            onset = emit("MI_START", f"想象 {direction} 运动；保持实际身体不动。",
                         trial=active_trial, original_class=trial["original_class"])
            wait_until(onset + config.imagery_seconds)
            onset = emit("MI_END", "停止想象，休息。", trial=active_trial)
            manifest["completed_trials"] += 1
            active_trial = None
            wait_until(onset + config.inter_trial_seconds)
        emit("SESSION_END", "本次提示已完成，请等待采集器保存数据。")
        sleep(config.drain_seconds)
        manifest["status"] = "complete"
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _save_manifest(output, manifest)
        return manifest
    except BaseException as exc:
        manifest["status"] = "aborted"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        if outlet is not None and clock is not None and labels is not None:
            try:
                emit("DISCONNECT", "实验已中止；这次记录不应作为完整校准/验收数据。",
                     trial=active_trial, check_consumer=False)
                sleep(config.drain_seconds)
            except BaseException as cleanup:
                manifest["cleanup_errors"].append(f"DISCONNECT: {type(cleanup).__name__}: {cleanup}")
        try:
            _save_manifest(output, manifest)
        except BaseException as cleanup:
            print(f"无法写入实验异常状态：{cleanup}；原始异常：{exc}", file=sys.stderr)
        raise
    finally:
        if labels is not None:
            labels.close()


def add_experiment_arguments(parser):
    parser.add_argument("--subject", required=True)
    parser.add_argument("--session-id", choices=["T", "E"], default="T")
    parser.add_argument("--config", required=True, help="与采集器一致的设备 JSON，或冻结模型的 manifest.json")
    parser.add_argument("--output", required=True, help="新的提示记录目录（必须不存在）")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--trials-per-run", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rest-seconds", type=float, default=2.0)
    parser.add_argument("--inter-trial-seconds", type=float, default=2.0)
    parser.add_argument("--run-rest-seconds", type=float, default=3.0)
    parser.add_argument("--marker-name", default="MI_Markers")
    parser.add_argument("--consumer-timeout-seconds", type=float, default=120.0)


def config_from_args(args):
    source = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError("设备配置/模型 manifest 必须为 JSON 对象")
    acquisition = source.get("config", source)
    if not isinstance(acquisition, dict) or "imagery_seconds" not in acquisition or "warmup_seconds" not in acquisition:
        raise ValueError("配置缺少 imagery_seconds 或 warmup_seconds")
    imagery = acquisition["imagery_seconds"]
    warmup = acquisition["warmup_seconds"]
    if isinstance(warmup, bool) or not isinstance(warmup, (int, float)) or not isfinite(warmup) or warmup < 0:
        raise ValueError("warmup_seconds 必须为有限非负数")
    if args.run_rest_seconds < warmup:
        raise ValueError("run_rest_seconds 不得小于采集配置的 warmup_seconds")
    if "model_version" in source:
        metadata = source.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("training_subject") != args.subject:
            raise ValueError("实验受试者与冻结模型的校准受试者不一致")
    canonical = json.dumps(acquisition, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return ExperimentConfig(subject_id=args.subject, session_id=args.session_id, runs=args.runs,
                            trials_per_run=args.trials_per_run, seed=args.seed,
                            imagery_seconds=imagery, rest_seconds=args.rest_seconds,
                            inter_trial_seconds=args.inter_trial_seconds,
                            run_rest_seconds=args.run_rest_seconds, marker_name=args.marker_name,
                            consumer_timeout_seconds=args.consumer_timeout_seconds,
                            acquisition_config_sha256=sha256(canonical.encode()).hexdigest())


def cmd_experiment(args):
    manifest = run_experiment(config_from_args(args), args.output)
    print(f"提示记录完成：{Path(args.output).resolve()}；已完成 {manifest['completed_trials']} 次提示，EEG 有效性须另行检查。")


# ========================================================================
# 19. 真实设备预检与采集质量报告
# ========================================================================
@dataclass(frozen=True)
class QualityThresholds:

    # 工程预检初始门限，需要结合真实设备核定；失败项会阻止预检就绪。
    min_duration_seconds: float = 5.0
    max_rate_error_fraction: float = 0.05
    max_timestamp_error_seconds: float = 0.002
    max_sample_age_seconds: float = 0.5
    max_silence_seconds: float = 0.5
    flat_window_seconds: float = 1.0
    flat_std_volts: float = 1e-9
    max_abs_volts: float = 500e-6
    max_ac_rms_volts: float = 100e-6
    line_frequency_hz: float = 50.0
    line_bandwidth_hz: float = 2.0
    max_line_power_fraction: float = 0.35
    high_frequency_hz: float = 40.0
    max_high_frequency_power_fraction: float = 0.65
    saturation_run_seconds: float = 0.02

    def __post_init__(self):
        for name, value in asdict(self).items():
            minimum_inclusive = name in {"max_timestamp_error_seconds", "flat_std_volts"}
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)
                    or value < 0 or (value == 0 and not minimum_inclusive)):
                raise ValueError(f"{name} must be finite and {'nonnegative' if minimum_inclusive else 'positive'}")
        for name in ("max_rate_error_fraction", "max_line_power_fraction", "max_high_frequency_power_fraction"):
            if getattr(self, name) >= 1:
                raise ValueError(f"{name} must be below 1")
        if self.flat_window_seconds > self.min_duration_seconds:
            raise ValueError("flat_window_seconds cannot exceed min_duration_seconds")

    @classmethod
    def for_config(cls, config, overrides=None):
        values = {p.name: getattr(config, p.name) for p in fields(cls) if hasattr(config, p.name)}
        values.update({"max_timestamp_error_seconds": config.timestamp_tolerance_seconds,
                  "max_sample_age_seconds": config.max_age_seconds,
                  "max_abs_volts": config.max_abs_volts,
                  "flat_std_volts": config.flat_std_volts,
                  "flat_window_seconds": min(getattr(config, "flat_window_seconds", cls.flat_window_seconds),
                                             getattr(config, "window_seconds", cls.min_duration_seconds))})
        values["min_duration_seconds"] = max(cls.min_duration_seconds, values["flat_window_seconds"])
        for name, value in (overrides or {}).items():
            if name in values and value != values[name]:
                raise ValueError(f"Preflight cannot override frozen channel/timing limit {name}; update the device configuration and recalibrate")
        values.update(overrides or {})
        return cls(**values)

    @classmethod
    def for_window(cls, config):
        return replace(cls.for_config(config), min_duration_seconds=config.window_seconds,
                       flat_window_seconds=min(config.flat_window_seconds, config.window_seconds))


def _summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"min": None, "median": None, "p95": None, "max": None}
    return {"min": float(np.min(values)), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95)), "max": float(np.max(values))}


def _longest_run(mask):
    boundaries = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(int))
    lengths = np.flatnonzero(boundaries == -1) - np.flatnonzero(boundaries == 1)
    return int(max(lengths, default=0))


def channel_quality_report(data_volts, sfreq, channels, thresholds=None):
    # 输入为原始伏特信号 (通道数, 采样点数)，逐通道检查幅度、平直及频谱噪声。
    limits = thresholds or QualityThresholds()
    data = np.asarray(data_volts, dtype=float)
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("sfreq must be positive")
    if data.ndim != 2 or data.shape[0] != len(channels) or len(set(channels)) != len(channels):
        raise ValueError("Expected one row per unique configured channel")
    reports = []
    window = max(2, round(limits.flat_window_seconds * sfreq))
    for name, samples in zip(channels, data):
        reasons = []
        finite = np.isfinite(samples)
        good = samples[finite]
        if len(samples) < np.ceil(limits.min_duration_seconds * sfreq):
            reasons.append("INSUFFICIENT_DATA")
        if not finite.all():
            reasons.append("NONFINITE")
        maximum = float(np.max(np.abs(good))) if len(good) else None
        std = float(np.std(good)) if len(good) else None
        # 重叠检查能发现跨固定窗口边界的平直区间。
        starts = list(range(0, max(0, len(samples) - window + 1), max(1, window // 2)))
        if len(samples) >= window and (not starts or starts[-1] != len(samples) - window):
            starts.append(len(samples) - window)
        flat_count = sum(np.isfinite(samples[start:start + window]).all()
                         and np.std(samples[start:start + window]) <= limits.flat_std_volts for start in starts)
        if flat_count or (std is not None and std <= limits.flat_std_volts):
            reasons.append("FLATLINE")
        if maximum is not None and maximum > limits.max_abs_volts:
            reasons.append("EXCESSIVE_AMPLITUDE")
        if std is not None and std > limits.max_ac_rms_volts:
            reasons.append("EXCESSIVE_AC_RMS")
        # 频谱检查要求完整有限的原始数据，不跨坏样本插值或拼接。
        line_fraction = high_fraction = None
        line_observable = limits.line_frequency_hz + limits.line_bandwidth_hz < sfreq / 2
        high_observable = limits.high_frequency_hz < sfreq / 2
        if len(samples) >= window and finite.all():
            frequencies, density = welch(samples, fs=sfreq, nperseg=min(len(samples), max(window, round(2 * sfreq))))
            included = frequencies >= 1.0
            total = float(np.sum(density[included]))
            if total > 0:
                if line_observable:
                    line_fraction = float(np.sum(density[np.abs(frequencies - limits.line_frequency_hz)
                                                        <= limits.line_bandwidth_hz]) / total)
                if high_observable:
                    high_fraction = float(np.sum(density[frequencies >= limits.high_frequency_hz]) / total)
        if line_fraction is not None and line_fraction > limits.max_line_power_fraction:
            reasons.append("LINE_NOISE")
        if high_fraction is not None and high_fraction > limits.max_high_frequency_power_fraction:
            reasons.append("HIGH_FREQUENCY_NOISE")
        longest_rail = 0
        if len(good):
            longest_rail = max(_longest_run(finite & (samples == np.max(good))),
                               _longest_run(finite & (samples == np.min(good))))
        if longest_rail >= max(3, int(np.ceil(limits.saturation_run_seconds * sfreq))):
            reasons.append("SATURATION_SUSPECTED")
        reports.append({"channel": name, "status": "ok" if not reasons else "needs_adjustment",
                        "reasons": reasons, "sample_count": len(samples), "nonfinite_count": int((~finite).sum()),
                        "max_abs_volts": maximum, "ac_rms_volts": std, "flat_windows": int(flat_count),
                        "checked_flat_windows": len(starts), "longest_extremum_run_samples": longest_rail,
                        "line_power_fraction": line_fraction, "line_band_observable": line_observable,
                        "high_frequency_power_fraction": high_fraction, "high_frequency_band_observable": high_observable})
    return reports


def timing_report(timestamps, received_at, sfreq, elapsed_seconds, max_silence_seconds=0.0, thresholds=None):
    # 样本时间与其所属数据包的接收时间必须处于同一时钟域，才能计算数据年龄。
    limits = thresholds or QualityThresholds()
    stamps, arrivals = np.asarray(timestamps, dtype=float), np.asarray(received_at, dtype=float)
    if stamps.ndim != 1 or arrivals.shape != stamps.shape:
        raise ValueError("One arrival time is required for every timestamp")
    if not np.isfinite(sfreq) or sfreq <= 0 or not np.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        raise ValueError("Sampling rate and elapsed duration must be finite and positive")
    if not np.isfinite(max_silence_seconds) or max_silence_seconds < 0:
        raise ValueError("max_silence_seconds must be finite and nonnegative")
    reasons = []
    delta = np.diff(stamps)
    good_delta = delta[np.isfinite(delta) & (delta > 0)]
    expected = 1.0 / sfreq
    errors = np.abs(good_delta - expected)
    invalid = int((~np.isfinite(stamps)).sum())
    nonmonotonic = int((np.isfinite(delta) & (delta <= 0)).sum())
    gaps = int((good_delta > expected + limits.max_timestamp_error_seconds).sum())
    violations = int((errors > limits.max_timestamp_error_seconds + 1e-12).sum())
    span = float(stamps[-1] - stamps[0]) if len(stamps) >= 2 and np.isfinite(stamps[[0, -1]]).all() else 0
    measured = (len(stamps) - 1) / span if span > 0 else None
    received_rate = len(stamps) / elapsed_seconds
    age = arrivals - stamps
    if elapsed_seconds < limits.min_duration_seconds or len(stamps) < np.ceil(limits.min_duration_seconds * sfreq):
        reasons.append("INSUFFICIENT_DATA")
    if invalid or not np.isfinite(arrivals).all():
        reasons.append("NONFINITE_TIMESTAMPS")
    if nonmonotonic:
        reasons.append("NONMONOTONIC_TIMESTAMPS")
    if violations:
        reasons.append("TIMESTAMP_INTERVAL_VIOLATIONS")
    if measured is None or abs(measured / sfreq - 1) > limits.max_rate_error_fraction:
        reasons.append("TIMESTAMP_RATE_MISMATCH")
    if abs(received_rate / sfreq - 1) > limits.max_rate_error_fraction:
        reasons.append("RECEIVED_RATE_MISMATCH")
    if np.any(age > limits.max_sample_age_seconds):
        reasons.append("STALE_SAMPLES")
    if np.any(age < -limits.max_timestamp_error_seconds):
        reasons.append("FUTURE_TIMESTAMPS")
    if max_silence_seconds > limits.max_silence_seconds:
        reasons.append("RECEIVE_DROPOUT")
    return {"status": "ok" if not reasons else "needs_adjustment", "reasons": reasons,
            "sample_count": len(stamps), "declared_sfreq": sfreq, "timestamp_sfreq": measured,
            "received_sfreq": received_rate, "elapsed_seconds": elapsed_seconds,
            "interval_seconds": _summary(good_delta), "absolute_interval_error_seconds": _summary(errors),
            "sample_age_seconds": _summary(age), "nonfinite_timestamp_count": invalid,
            "nonmonotonic_count": nonmonotonic, "gap_count": gaps,
            "interval_violation_count": violations, "max_receive_silence_seconds": max_silence_seconds,
            "sample_age_scope": "Clock-corrected timestamp to local pull completion; includes buffering and is not physical acquisition latency.",
            "alignment_scope": "EEG timestamps only; actual cue-to-EEG alignment requires a shared hardware observation."}


def _stream_mapping(header, config, units):
    if abs(header.nominal_srate() - config.sfreq) > 1e-8:
        raise ValueError("Declared sampling rate differs from device configuration")
    if units and config.lsl_input_units and UNIT_SCALE.get(units) != UNIT_SCALE.get(config.lsl_input_units):
        raise ValueError("Input units differ from device configuration")
    units = units or config.lsl_input_units
    if units is not None and units not in UNIT_SCALE:
        raise ValueError("Unknown input units")
    node = header.desc().child("channels").child("channel")
    labels, scales = [], []
    for _ in range(header.channel_count()):
        labels.append(node.child_value("label"))
        reported_unit = node.child_value("unit")
        if reported_unit and units and UNIT_SCALE.get(reported_unit) != UNIT_SCALE.get(units):
            raise ValueError("Explicit units conflict with stream metadata")
        scales.append(UNIT_SCALE.get(reported_unit or units))
        node = node.next_sibling()
    reported_labels = list(labels)
    if config.lsl_channel_labels is not None:
        if len(config.lsl_channel_labels) != header.channel_count():
            raise ValueError("Explicit channel mapping length differs from stream")
        labels = list(config.lsl_channel_labels)
    if len(set(labels)) != len(labels) or any(channel not in labels for channel in config.channels):
        raise ValueError("Every configured electrode must occur exactly once in LSL mapping")
    order = [labels.index(channel) for channel in config.channels]
    if any(scales[index] is None for index in order):
        raise ValueError("Missing/unknown units; verify V, mV or uV before preflight")
    reference = header.desc().child_value("reference")
    if reference and reference != config.reference:
        raise ValueError("Stream reference conflicts with device configuration")
    return order, np.asarray([scales[index] for index in order])[:, None], reported_labels


def capture_preflight(config, eeg_name, duration_seconds=15.0, thresholds=None, units=None,
                      resolve_timeout=5.0, *, lsl_module=None, monotonic=None):
    # 限时采集指定真实 EEG 流，关闭入口后生成通道质量和时序预检结果。
    limits = thresholds or QualityThresholds.for_config(config)
    if not np.isfinite(duration_seconds) or duration_seconds < limits.min_duration_seconds:
        raise ValueError(f"Preflight duration must be at least {limits.min_duration_seconds:g} seconds")
    if lsl_module is None:
        import pylsl as lsl_module
    monotonic = monotonic or time.monotonic
    matches = lsl_module.resolve_byprop("name", eeg_name, timeout=resolve_timeout)
    if len(matches) != 1:
        raise ValueError("Exactly one numeric EEG stream must match --eeg-name")
    inlet = lsl_module.StreamInlet(matches[0], max_buflen=max(1, int(np.ceil(duration_seconds))),
                                  recover=False, processing_flags=lsl_module.proc_clocksync)
    chunks, stamps, arrivals, errors = [], [], [], []
    max_silence = 0.0
    try:
        header = inlet.info(timeout=resolve_timeout)
        if header.channel_format() == lsl_module.cf_string:
            raise ValueError("Preflight requires a numeric raw EEG stream")
        order, scales, labels = _stream_mapping(header, config, units)
        metadata = {"name": header.name(), "source_id": header.source_id(), "reported_labels": labels,
                    "configured_channels": list(config.channels), "reference": config.reference,
                    "unit": "V", "processing_flags": "proc_clocksync (no dejitter)"}
        inlet.open_stream(timeout=resolve_timeout)
        metadata["initial_clock_correction_seconds"] = float(inlet.time_correction(timeout=resolve_timeout))
        # 丢弃时钟初始化期间积压的样本，避免预检一开始就统计过期数据。
        inlet.flush()
        began = last_received = monotonic()
        while monotonic() - began < duration_seconds:
            try:
                values, times = inlet.pull_chunk(timeout=min(.05, max(0.0, duration_seconds - (monotonic() - began))),
                                                max_samples=config.chunk_samples)
                now = monotonic()
                max_silence = max(max_silence, now - last_received)
                if times:
                    received = lsl_module.local_clock()
                    array = np.asarray(values, dtype=float)
                    if array.shape != (len(times), header.channel_count()):
                        raise ValueError("Malformed sample/channel dimensions in EEG chunk")
                    chunks.append(array.T[order] * scales)
                    stamps.extend(times)
                    arrivals.extend([received] * len(times))
                    last_received = now
            except Exception as exc:
                errors.append({"type": type(exc).__name__, "message": str(exc)})
                break
        elapsed = max(monotonic() - began, np.finfo(float).eps)
    finally:
        inlet.close_stream()
    data = np.concatenate(chunks, axis=1) if chunks else np.empty((len(config.channels), 0))
    channel_report = channel_quality_report(data, config.sfreq, config.channels, limits)
    timing = timing_report(stamps, arrivals, config.sfreq, elapsed, max_silence, limits)
    ready = not errors and timing["status"] == "ok" and all(row["status"] == "ok" for row in channel_report)
    return {"format": "mi-bci-preflight-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "ready" if ready else "needs_adjustment", "ready": ready,
            "decision": "可以开始" if ready else "需要调整", "thresholds": asdict(limits),
            "stream": metadata, "timing": timing, "channels": channel_report, "acquisition_errors": errors,
            "bad_channel_policy": "Stop and correct electrode/contact/noise problems, then rerun. Never drop or reorder configured channels automatically.",
            "contact_quality_scope": "Signal proxies only; no impedance was measured. Use the amplifier's impedance/contact check separately.",
            "validation_scope": "This preflight does not validate real human classification or actual cue-to-EEG alignment."}


def cmd_preflight(args, config=None):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Report already exists: {output}")
    if config is None:
        config = Config.from_dict(json.loads(Path(args.config).read_text(encoding="utf-8")))
    overrides = json.loads(Path(args.thresholds).read_text(encoding="utf-8")) if getattr(args, "thresholds", None) else {}
    limits = QualityThresholds.for_config(config, overrides)
    report = capture_preflight(config, args.eeg_name, args.duration, limits, getattr(args, "units", None))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"{report['decision']} ({report['status']}) — {output.resolve()}")
    for row in report["channels"]:
        if row["reasons"]:
            print(f"  {row['channel']}: {', '.join(row['reasons'])}")
    for reason in report["timing"]["reasons"]:
        print(f"  timing: {reason}")
    return 0 if report["ready"] else 2


def _save_diagnostic(directory, filename, report):
    # 附加诊断写盘失败时明确报告，但不替换采集过程的原始错误。
    try:
        with (Path(directory) / filename).open("w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    except Exception as exc:
        report["status"] = "incomplete"
        report["report_write_error"] = {"type": type(exc).__name__, "message": str(exc)}
    return report


def _capture_maps(directory, config):
    # 通过内存映射访问录制文件，避免将长会话的原始 EEG 整体复制进内存。
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    stream = manifest.get("stream", {})
    if (tuple(stream.get("channels", ())) != tuple(config.channels) or stream.get("sfreq") != config.sfreq
            or stream.get("unit") != "V" or stream.get("reference") != config.reference):
        raise ValueError("Captured channel order/rate/unit/reference differs from configuration")
    raw_path, timestamps_path = directory / "raw_eeg.f64", directory / "timestamps.f64"
    raw_bytes, timestamp_bytes = raw_path.stat().st_size, timestamps_path.stat().st_size
    stride = 8 * len(config.channels)
    available = min(raw_bytes // stride, timestamp_bytes // 8)
    problems = []
    if raw_bytes % stride or timestamp_bytes % 8 or raw_bytes // stride != timestamp_bytes // 8:
        problems.append("RAW_TIMESTAMP_FILE_LENGTH_MISMATCH")
    declared = manifest.get("raw_samples")
    if not isinstance(declared, int) or isinstance(declared, bool) or declared != available:
        problems.append("MANIFEST_SAMPLE_COUNT_MISMATCH")
    finalizing_closed_capture = (manifest.get("status") == "finalizing"
                                and manifest.get("finalization_target_status") == "complete"
                                and manifest.get("storage_verification", {}).get("status") == "confirmed"
                                and manifest.get("accepted_raw_samples") == declared)
    if manifest.get("status") != "complete" and not finalizing_closed_capture:
        problems.append("CAPTURE_NOT_COMPLETE")
    raw = np.memmap(raw_path, dtype="<f8", mode="r", shape=(available, len(config.channels))) if available else None
    stamps = np.memmap(timestamps_path, dtype="<f8", mode="r", shape=(available,)) if available else None
    return manifest, raw, stamps, available, problems


def _capture_segments(directory, available, problems):
    # 依据轻量块索引划分连续段，保留原始文件中可能不完整的数据供诊断。
    segments, expected = [], 0
    with (Path(directory) / "chunks.jsonl").open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                chunk = json.loads(line)
                offset, count, start, segment = (chunk[key] for key in ("offset_samples", "count", "start_sample", "segment_id"))
                if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                       for value in (offset, count, start, segment)) or count == 0:
                    raise ValueError("Invalid chunk indices")
                if offset != expected or offset + count > available:
                    raise ValueError("Chunk log has a gap/overlap or exceeds available raw data")
                same = (segments and segments[-1]["segment_id"] == segment
                        and segments[-1]["next_start_sample"] == start)
                if same:
                    segments[-1]["stop_offset"] = offset + count
                    segments[-1]["next_start_sample"] = start + count
                else:
                    segments.append({"segment_id": segment, "start_offset": offset,
                                     "stop_offset": offset + count, "next_start_sample": start + count})
                expected = offset + count
            except Exception as exc:
                problems.append(f"CHUNK_LOG_INVALID_AT_LINE_{line_number}: {exc}")
                break
    if expected != available:
        problems.append("RAW_SAMPLES_NOT_FULLY_INDEXED")
    return segments, expected


def write_capture_quality(directory, config):
    # 按连续段和非重叠原始窗口汇总质量，控制诊断内存用量。
    # 此处的原始信号窗口通过率，不等于因果滤波后模型窗口的有效比例。
    report = {"format": "mi-bci-capture-channel-quality-v1", "status": "incomplete", "issues": [],
              "channels": [], "segments": [],
              "scope": "Raw-voltage nonoverlapping diagnostic windows; not the causal model's valid-window fraction.",
              "contact_quality_scope": "Signal proxies only, not hardware impedance measurements.",
              "bad_channel_policy": "Do not drop/reorder channels. Correct the cause and reacquire before training/control."}
    try:
        manifest, raw, stamps, available, problems = _capture_maps(directory, config)
        report.update({"session_status": manifest.get("status"), "session_error": manifest.get("error"),
                       "available_samples": available, "issues": problems})
        segments, indexed = _capture_segments(directory, available, problems)
        report["indexed_samples"] = indexed
        seconds = config.window_seconds
        width = config.samples(seconds)
        base = QualityThresholds.for_config(config)
        limits = replace(base, min_duration_seconds=seconds, flat_window_seconds=min(base.flat_window_seconds, seconds))
        report["thresholds"] = asdict(limits)
        report["diagnostic_window_seconds"] = seconds
        totals = [{"channel": channel, "sample_count": 0, "nonfinite_count": 0, "max_abs_volts": None,
                   "complete_windows": 0, "bad_windows": 0, "reason_window_counts": Counter(),
                   "incomplete_tail_samples": 0} for channel in config.channels]
        for segment in segments:
            segment_report = {"segment_id": segment["segment_id"], "start_offset": segment["start_offset"],
                              "stop_offset": segment["stop_offset"], "complete_windows": 0,
                              "any_bad_channel_windows": 0, "timestamp_bad_windows": 0,
                              "incomplete_tail_samples": 0,
                              "channel_bad_windows": {channel: 0 for channel in config.channels}}
            for begin in range(segment["start_offset"], segment["stop_offset"], width):
                end = min(begin + width, segment["stop_offset"])
                complete = end - begin == width
                rows = channel_quality_report(raw[begin:end].T, config.sfreq, config.channels, limits)
                times = stamps[max(segment["start_offset"], begin - 1):end]
                delta = np.diff(times)
                timing_bad = bool(not np.isfinite(times).all() or np.any(delta <= 0)
                                  or np.any(np.abs(delta - 1 / config.sfreq) > config.timestamp_tolerance_seconds + 1e-12))
                if complete:
                    segment_report["complete_windows"] += 1
                    segment_report["timestamp_bad_windows"] += int(timing_bad)
                else:
                    segment_report["incomplete_tail_samples"] += end - begin
                any_bad = False
                for total, row in zip(totals, rows):
                    total["sample_count"] += row["sample_count"]
                    total["nonfinite_count"] += row["nonfinite_count"]
                    if row["max_abs_volts"] is not None:
                        total["max_abs_volts"] = max(total["max_abs_volts"] or 0, row["max_abs_volts"])
                    if complete:
                        total["complete_windows"] += 1
                        total["reason_window_counts"].update(row["reasons"])
                        if row["reasons"]:
                            total["bad_windows"] += 1
                            segment_report["channel_bad_windows"][row["channel"]] += 1
                            any_bad = True
                    else:
                        total["incomplete_tail_samples"] += end - begin
                        for reason in row["reasons"]:
                            if reason != "INSUFFICIENT_DATA":
                                total.setdefault("tail_reasons", set()).add(reason)
                if complete and any_bad:
                    segment_report["any_bad_channel_windows"] += 1
            count = segment_report["complete_windows"]
            segment_report["bad_channel_window_fraction"] = segment_report["any_bad_channel_windows"] / count if count else None
            segment_report["timestamp_bad_window_fraction"] = segment_report["timestamp_bad_windows"] / count if count else None
            report["segments"].append(segment_report)
        for total in totals:
            count = total["complete_windows"]
            total["bad_window_fraction"] = total["bad_windows"] / count if count else None
            total["tail_reasons"] = sorted(total.get("tail_reasons", ()))
            total["reason_window_counts"] = dict(total["reason_window_counts"])
            total["status"] = "ok" if count and not total["bad_windows"] and not total["tail_reasons"] else "needs_adjustment"
        report["channels"] = totals
        if problems:
            report["status"] = "incomplete"
        elif (any(row["status"] != "ok" for row in totals)
              or any(segment["timestamp_bad_windows"] for segment in report["segments"])):
            report["status"] = "needs_adjustment"
        else:
            report["status"] = "ok"
    except Exception as exc:
        report["issues"].append(f"{type(exc).__name__}: {exc}")
    return _save_diagnostic(directory, "channel_quality.json", report)


class _BoundedStats:

    # 蓄水池抽样限制统计内存：数量与极值精确，抽样分位数为近似值。
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.values = np.empty(0)
        self.keys = np.empty(0)
        self.count = 0
        self.minimum = self.maximum = None
        self.rng = np.random.default_rng(712)

    def add(self, values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            return
        self.count += len(values)
        self.minimum = min(self.minimum if self.minimum is not None else np.inf, float(values.min()))
        self.maximum = max(self.maximum if self.maximum is not None else -np.inf, float(values.max()))
        self.values = np.r_[self.values, values]
        self.keys = np.r_[self.keys, self.rng.random(len(values))]
        if len(self.keys) > self.capacity:
            keep = np.argpartition(self.keys, self.capacity - 1)[:self.capacity]
            self.keys, self.values = self.keys[keep], self.values[keep]

    def report(self):
        result = _summary(self.values)
        result.update({"count": self.count, "min": self.minimum, "max": self.maximum,
                       "p99": float(np.quantile(self.values, .99)) if len(self.values) else None,
                       "quantiles_approximate": self.count > self.capacity,
                       "quantile_sample_count": len(self.values)})
        return result


def write_capture_timing(directory, config, timing_records=None):
    # 分别报告 EEG 抖动、标记接收年龄与采样格点对齐，不能据此确认物理提示起点。
    report = {"format": "mi-bci-capture-timing-v1", "status": "incomplete", "issues": [],
              "physical_cue_alignment": {"status": "not_measured", "reason": "Need a photodiode/audio-loopback/trigger channel to observe actual cue onset on the EEG acquisition clock."},
              "scope": "Software clock-corrected timestamps and receive times; not physical acquisition/display onset latency.",
              "eeg_sample_age_scope": {
                  "start_field": "timestamps.f64: clock-corrected EEG sample timestamp",
                  "end_field": "chunks.jsonl.received_at: live pipeline release time",
                  "meaning": "Includes the live marker holdback and buffering before pipeline release; not pure network transport latency. Pending raw chunks in a failed capture can retain their original pull time."},
              "recommendations": {"applied": False}}
    try:
        manifest, raw, stamps, available, problems = _capture_maps(directory, config)
        report.update({"session_status": manifest.get("status"), "session_error": manifest.get("error"),
                       "available_samples": available, "issues": problems})
        _, indexed = _capture_segments(directory, available, problems)
        report["indexed_samples"] = indexed
        interval_errors, sample_ages = _BoundedStats(), _BoundedStats()
        gaps = nonmonotonic = invalid_stamps = interval_violations = missing_receipts = 0
        stale_samples = future_samples = 0
        for begin in range(0, available, 65536):
            times = stamps[max(0, begin - 1):min(available, begin + 65536)]
            invalid_stamps += int((~np.isfinite(stamps[begin:min(available, begin + 65536)])).sum())
            delta = np.diff(times)
            positive = delta[np.isfinite(delta) & (delta > 0)]
            nonmonotonic += int((np.isfinite(delta) & (delta <= 0)).sum())
            gaps += int((positive > 1.5 / config.sfreq).sum())
            errors = np.abs(positive - 1 / config.sfreq)
            interval_violations += int((errors > config.timestamp_tolerance_seconds + 1e-12).sum())
            interval_errors.add(errors[positive <= 1.5 / config.sfreq])
        recorded_grid_ranges = []
        with (Path(directory) / "chunks.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                chunk = json.loads(line)
                offset, count = chunk["offset_samples"], chunk["count"]
                if not isinstance(offset, int) or not isinstance(count, int) or offset < 0 or count < 1 or offset + count > available:
                    raise ValueError("Chunk indices exceed available raw data")
                start_sample = chunk.get("start_sample")
                if isinstance(start_sample, bool) or not isinstance(start_sample, int) or start_sample < 0:
                    raise ValueError("Chunk start_sample must be a nonnegative integer")
                recorded_grid_ranges.append((start_sample, start_sample + count, offset))
                received = chunk.get("received_at")
                if not isinstance(received, (int, float)) or not np.isfinite(received):
                    missing_receipts += count
                    continue
                for begin in range(offset, offset + count, 65536):
                    ages = received - stamps[begin:min(begin + 65536, offset + count)]
                    sample_ages.add(ages)
                    stale_samples += int((ages > config.max_age_seconds).sum())
                    future_samples += int((ages < -config.timestamp_tolerance_seconds).sum())
        markers = []
        with (Path(directory) / "events.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("type") == "Marker":
                    markers.append(row)
        transport = list(timing_records) if timing_records is not None else markers
        marker_ages, alignment = _BoundedStats(), _BoundedStats()
        missing_marker_age = missing_alignment = 0
        future_markers = beyond_grace = bad_grid = 0
        identities = Counter()
        for row in transport:
            identities[(row.get("kind"), row.get("trial_id"), row.get("timestamp"))] += 1
            received, timestamp = row.get("received_at"), row.get("timestamp")
            if (isinstance(received, (int, float)) and np.isfinite(received)
                    and isinstance(timestamp, (int, float)) and np.isfinite(timestamp)):
                age = received - timestamp
                marker_ages.add([age])
                future_markers += int(age < -config.timestamp_tolerance_seconds)
                beyond_grace += int(age > config.lsl_marker_grace_seconds)
            else:
                missing_marker_age += 1
        grid_measurements, grid_sources = [], Counter()
        for marker in markers:
            value = marker.get("alignment_seconds")
            source, missing_reason, recorded_offset = "recorded_marker_alignment", None, None
            if value is None:
                sample, timestamp = marker.get("sample"), marker.get("timestamp")
                valid_sample = isinstance(sample, int) and not isinstance(sample, bool) and sample >= 0
                valid_timestamp = (isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool)
                                   and np.isfinite(timestamp))
                if valid_sample and valid_timestamp:
                    matches = [offset + sample - start for start, stop, offset in recorded_grid_ranges
                               if start <= sample < stop]
                    if len(matches) == 1:
                        recorded_offset = matches[0]
                        if np.isfinite(stamps[recorded_offset]):
                            value = float(stamps[recorded_offset] - timestamp)
                            source = "derived_from_recorded_grid"
                        else:
                            missing_reason = "NONFINITE_RECORDED_TIMESTAMP"
                    else:
                        missing_reason = "NO_RECORDED_SAMPLE_MATCH" if not matches else "AMBIGUOUS_RECORDED_SAMPLE_MATCH"
                else:
                    missing_reason = "INVALID_MARKER_SAMPLE_OR_TIMESTAMP"
            if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
                alignment.add([value])
                bad_grid += int(value < 0 or value > 1 / config.sfreq + config.timestamp_tolerance_seconds)
            else:
                source = "missing"
                value = None
                missing_reason = missing_reason or "INVALID_RECORDED_ALIGNMENT"
                missing_alignment += 1
            grid_sources[source] += 1
            grid_measurements.append({"kind": marker.get("kind"), "trial_id": marker.get("trial_id"),
                                      "sample": marker.get("sample"), "alignment_seconds": value,
                                      "source": source, "recorded_offset_samples": recorded_offset,
                                      "missing_reason": missing_reason})
        interval, eeg_age, marker_age, grid = (stats.report() for stats in (interval_errors, sample_ages, marker_ages, alignment))
        report.update({"eeg_interval_error_seconds_excluding_large_gaps": interval,
                       "eeg_sample_age_seconds": eeg_age, "marker_transport_age_seconds": marker_age,
                       "marker_to_next_eeg_sample_seconds": grid,
                       "marker_grid_measurements": grid_measurements,
                       "marker_grid_source_counts": dict(grid_sources),
                       "expected_grid_quantization_seconds": [0, 1 / config.sfreq],
                       "sample_gap_count": gaps, "nonmonotonic_timestamp_count": nonmonotonic,
                       "nonfinite_timestamp_count": invalid_stamps, "timestamp_interval_violation_count": interval_violations,
                       "stale_eeg_sample_count": stale_samples, "future_eeg_sample_count": future_samples,
                       "future_marker_count": future_markers, "marker_age_beyond_current_grace_count": beyond_grace,
                       "marker_age_scope": "Clock-corrected source-to-receipt age only. EEG holdback starts at EEG receipt, so marker age greater than holdback is not itself evidence of a late marker.",
                       "unexpected_grid_alignment_count": bad_grid,
                       "missing_eeg_receipt_samples": missing_receipts, "received_marker_count": len(transport),
                       "aligned_event_count": len(markers), "missing_marker_receipts": missing_marker_age,
                       "missing_marker_grid_measurements": missing_alignment,
                       "duplicate_received_marker_count": sum(max(0, count - 1) for count in identities.values())})
        forwarded = Counter((row.get("kind"), row.get("trial_id"), row.get("timestamp")) for row in markers)
        report["unforwarded_received_marker_count"] = sum((identities - forwarded).values())
        missing = []
        if not interval["count"]:
            missing.append("EEG_INTERVAL_MEASUREMENTS")
        if not marker_age["count"] or missing_marker_age:
            missing.append("MARKER_RECEIPT_MEASUREMENTS")
        if not grid["count"] or missing_alignment:
            missing.append("MARKER_GRID_ALIGNMENT_MEASUREMENTS")
        if not eeg_age["count"] or missing_receipts:
            missing.append("EEG_RECEIPT_MEASUREMENTS")
        report["missing_measurements"] = missing
        tolerance = (interval["max"] * 1.25 + .0002
                     if interval["count"] and not gaps and not nonmonotonic and not invalid_stamps else None)
        grace = max(0, marker_age["max"]) + .02 if marker_age["count"] else None
        grace_fits = (grace is not None and eeg_age["count"] > 0
                      and eeg_age["max"] + max(0.0, grace - config.lsl_marker_grace_seconds) < config.max_age_seconds)
        report["recommendations"].update({
            "candidate_timestamp_tolerance_seconds": tolerance,
            "candidate_marker_grace_seconds": grace if grace_fits else None,
            "marker_grace_fits_data_age_budget": grace_fits,
            "current_timestamp_tolerance_seconds": config.timestamp_tolerance_seconds,
            "current_marker_grace_seconds": config.lsl_marker_grace_seconds,
            "method": "Candidate timestamp tolerance = maximum observed non-gap interval error × 1.25 + 0.2 ms; unavailable after gaps/clock faults. Conservative marker holdback candidate = maximum marker age + 20 ms (not a measured differential delay). Extra holdback plus observed maximum EEG pipeline age must stay within max_age_seconds.",
            "review_required": "Engineering candidates from this capture only. Repeat with load and hardware onset validation; never widen tolerance to hide gaps, and recalibrate/revalidate when changing frozen processing."})
        report["status"] = "incomplete" if problems or missing else "measured"
        if not problems and (gaps or nonmonotonic or invalid_stamps or interval_violations or stale_samples
                             or future_samples or future_markers or bad_grid
                             or report["unforwarded_received_marker_count"]
                             or report["duplicate_received_marker_count"]):
            report["status"] = "needs_adjustment"
    except Exception as exc:
        report["issues"].append(f"{type(exc).__name__}: {exc}")
    return _save_diagnostic(directory, "alignment_report.json", report)



# ========================================================================
# 20. 命令行处理与主入口
# ========================================================================
def _timeline_digest(recording):
    # 摘要覆盖信号、时间戳及中性阶段定位，刻意排除类别真值。
    hasher = sha256()
    for array in (recording.data, recording.timestamps):
        hasher.update(np.ascontiguousarray(array).view(np.uint8))
    neutral = {"subject": recording.subject_id, "session": recording.session_id,
               "stream": asdict(recording.info), "run_starts": recording.run_starts,
               "trials": [{key: getattr(trial, key) for key in ("trial_id", "run_id", "cue_sample", "start_sample")}
                          for trial in recording.trials]}
    hasher.update(canonical_json(neutral).encode("utf-8"))
    return hasher.hexdigest()


def _config(path):
    return Config.from_dict(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def _brief(report):
    scores = report["all"]
    def percent(value):
        return "N/A" if value is None else f"{value:.1%}"
    print(f"Trial accuracy: {percent(scores['accuracy'])} | balanced: {percent(scores['balanced_accuracy'])}")
    print(f"Command accuracy: {percent(scores['command_accuracy'])} | coverage: {percent(scores['coverage'])} | correct / all: {percent(scores['correct_command_fraction'])}")


def _replay(recording, bundle, output, realtime=False, speed=1.0):
    if recording.subject_id != bundle.metadata.get("training_subject"):
        raise ValueError("Replay subject does not match this subject-specific model's training subject")
    context = {"format": "mi-bci-replay-context-v1", "model_version": bundle.version,
               "config": bundle.config.to_dict(), "subject_id": recording.subject_id,
               "session_id": recording.session_id, "timeline_sha256": _timeline_digest(recording)}
    source = ReplaySource(recording, chunk_samples=bundle.config.chunk_samples,
                          realtime=realtime, speed=speed, imagery_seconds=bundle.config.imagery_seconds)
    predictions, commands = run_session(source, bundle, output)
    write_json(Path(output) / "replay_context.json", context)
    report = evaluate(recording.trials, predictions, commands, **_evaluation_parameters(bundle.config))
    report["data_provenance"] = recording.metadata
    report["mode"] = "continuous_replay"
    report["evaluation_role"] = "independent_E_session" if recording.session_id == "E" else "training_session_replay_not_independent"
    report["session_status"] = json.loads((Path(output) / "manifest.json").read_text(encoding="utf-8"))["status"]
    write_json(Path(output) / "evaluation.json", report)
    _brief(report)
    return report


def cmd_train(args):
    if Path(args.model).exists():
        raise FileExistsError(f"Model destination exists: {args.model}")
    recording = load_recording(args.data)
    config = _config(args.config) if args.config else None
    try:
        bundle = train_recording(recording, config,
                                 min_trials_per_class_per_run=args.min_trials_per_class_per_run,
                                 max_class_ratio=args.max_class_ratio,
                                 max_rejected_trial_fraction=args.max_rejected_trial_fraction,
                                 max_rejected_window_fraction=args.max_rejected_window_fraction)
    except CalibrationDataError as exc:
        audit = Path(args.audit_output or (str(args.model) + ".audit.json"))
        if audit.exists():
            raise FileExistsError(f"Audit destination exists: {audit}; original error: {exc}") from exc
        audit.parent.mkdir(parents=True, exist_ok=True)
        write_json(audit, exc.report)
        raise ValueError(f"{exc}; audit saved to {audit}") from exc
    bundle.save(args.model)
    print(f"Saved frozen model {bundle.version}: {Path(args.model).resolve()}")
    print(json.dumps(bundle.report["selected_policy"], indent=2))


def cmd_replay(args):
    bundle = ModelBundle.load(args.model)
    recording = load_recording(args.data)
    _replay(recording, bundle, args.output, realtime=args.realtime, speed=args.speed)


def cmd_evaluate(args):
    if Path(args.output).exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    recording = load_recording(args.data)
    context_path = Path(args.session) / "replay_context.json"
    context = json.loads(context_path.read_text(encoding="utf-8")) if context_path.exists() else None
    if context is not None and context.get("timeline_sha256") != _timeline_digest(recording):
        raise ValueError("Scoring recording differs from the replay signal or neutral trial timeline")
    config = Config.from_dict(context["config"]) if context else _recording_config(recording)
    predictions, commands = [], []
    with (Path(args.session) / "events.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            kind = event.pop("type", None)
            if kind == "Prediction":
                if context and event.get("model_version") != context["model_version"]:
                    raise ValueError("Session contains predictions from a different model version")
                predictions.append(Prediction(**event))
            elif kind == "Command":
                commands.append(Command(**event))
    report = evaluate(recording.trials, predictions, commands, **_evaluation_parameters(config))
    report["data_provenance"] = recording.metadata
    report["mode"] = "post_hoc_replay_scoring" if context else "post_hoc_scoring_without_replay_context"
    report["alignment_verified"] = context is not None
    report["label_alignment_verified"] = context is not None
    report["alignment_verification_scope"] = "Recording/phase identity only; physical cue alignment is not measured"
    manifest_path = Path(args.session) / "manifest.json"
    report["session_status"] = json.loads(manifest_path.read_text(encoding="utf-8")).get("status") if manifest_path.exists() else "unknown"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    _brief(report)


def _lsl_description(header):
    node = header.desc().child("channels").child("channel")
    channels = []
    for index in range(header.channel_count()):
        channels.append({"index": index + 1, "label": node.child_value("label"), "unit": node.child_value("unit")})
        node = node.next_sibling()
    return {"name": header.name(), "type": header.type(), "source_id": header.source_id(),
            "sfreq": header.nominal_srate(), "channel_count": header.channel_count(),
            "channel_format": header.channel_format(), "reference": header.desc().child_value("reference"),
            "channels": channels}


def _lsl_header(eeg_name):
    import pylsl
    matches = pylsl.resolve_byprop("name", eeg_name, timeout=5)
    if len(matches) != 1:
        raise ValueError("Exactly one stream must match --eeg-name; use inspect-lsl first")
    inlet = pylsl.StreamInlet(matches[0], recover=False)
    try:
        header = inlet.info(timeout=5)
        if header.channel_format() == pylsl.cf_string:
            raise ValueError("Select the numeric raw EEG stream")
        return _lsl_description(header)
    finally:
        inlet.close_stream()


def cmd_inspect_lsl(args):
    import pylsl
    descriptions = []
    for stream in pylsl.resolve_streams(wait_time=args.timeout):
        inlet = pylsl.StreamInlet(stream, recover=False)
        try:
            descriptions.append(_lsl_description(inlet.info(timeout=3)))
        finally:
            inlet.close_stream()
    print(json.dumps(descriptions, ensure_ascii=False, indent=2))
    if not descriptions:
        print("No LSL streams found. Start raw EEG sharing in OpenBCI GUI first.")


def cmd_configure_lsl(args):
    # 读取实际设备流生成配置，电极映射及参考必须由使用者确认。
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(f"Configuration already exists: {target}")
    header = _lsl_header(args.eeg_name)
    rate = float(header["sfreq"])
    if not np.isfinite(rate) or rate <= 60:
        raise ValueError("Raw EEG must declare a regular sample rate above 60 Hz for the 8–30 Hz filter")
    if header["channel_count"] != 16:
        raise ValueError("This setup expects the 16-channel raw EEG stream, without AUX/FFT/band-power channels")
    labels = args.channels
    if args.channels_file:
        labels = json.loads(Path(args.channels_file).read_text(encoding="utf-8-sig"))
    if labels is None:
        labels = [channel["label"] for channel in header["channels"]]
        if any(not name or re.fullmatch(r"(?i)(?:ch(?:annel)?[ _-]*)?\d+", name) for name in labels):
            raise ValueError("Stream has no electrode-position labels. Supply --channels in CH1–CH16 wiring order")
    if not isinstance(labels, list) or len(labels) != 16:
        raise ValueError("Supply exactly 16 actual electrode names in transmitted CH1–CH16 order")
    if any(not isinstance(label, str) or label.startswith("REPLACE_") for label in labels):
        raise ValueError("Replace every electrode template placeholder with the actual wired electrode position")
    units = args.units
    reported_units = [c["unit"] for c in header["channels"]]
    if not units:
        if len(set(reported_units)) != 1 or reported_units[0] not in UNIT_SCALE:
            raise ValueError("Units are absent/ambiguous; supply --units after checking the raw stream's units")
        units = reported_units[0]
    if any(unit and UNIT_SCALE.get(unit) != UNIT_SCALE[units] for unit in reported_units):
        raise ValueError("Requested input units conflict with stream metadata")
    if header["reference"] and header["reference"] != args.reference:
        raise ValueError("Requested reference conflicts with stream metadata")
    timings = {name: round(seconds * rate) / rate for name, seconds in
               (("warmup_seconds", 1), ("window_seconds", 2), ("step_seconds", .2),
                ("cue_offset_seconds", .5), ("imagery_seconds", 4), ("buffer_seconds", 10))}
    config = Config(sfreq=rate, channels=tuple(labels), reference=args.reference,
                    lsl_channel_labels=tuple(labels), lsl_input_units=units,
                    chunk_samples=max(1, round(rate * .04)), **timings)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, config.to_dict())
    print(f"Saved {target.resolve()}: {rate:g} Hz, 16 channels, input {units}, reference {args.reference}")
    print(f"Cue offset {config.cue_offset_seconds:g}s; verify CH1–CH16 wiring against your electrode diagram.")


def _capture_context(args, config, source, mode, model_version=None):
    if not args.subject.strip():
        raise ValueError("A nonempty subject identifier is required")
    return {"format": "mi-bci-personal-capture-v1", "subject_id": args.subject,
            "session_id": "E" if mode == "live" else args.session_id,
            "mode": mode, "config": config.to_dict(), "model_version": model_version,
            "application_version": __version__,
            "prediction_timestamp_semantics": "last_actual_EEG_sample_in_complete_window",
            "eeg_name": args.eeg_name, "marker_name": args.marker_name,
            "reported_lsl_labels": list(getattr(source, "reported_labels", [])),
            "started_at_utc": datetime.now(timezone.utc).isoformat()}


def cmd_record(args):
    if Path(args.output).exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    config = _config(args.config)
    source = LiveSource(config, args.eeg_name, args.marker_name, config.reference, args.units)
    try:
        context = _capture_context(args, config, source, "calibration")
        print("Connected. Waiting for the external experiment markers. Acquisition saves automatically at SESSION_END; Ctrl+C stops early.", flush=True)
        summary = record_session(source, config, args.output,
                                 observer=lambda item: print(f"{item.kind}: {item.trial_id or ''}", flush=True)
                                 if isinstance(item, Marker) else None, context=context)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        source.close()


def cmd_import_session(args):
    if Path(args.output).exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    recording = load_personal_session(args.session, args.labels)
    save_recording(args.output, recording)
    print(f"Saved {Path(args.output).resolve()}: {recording.subject_id}/{recording.session_id}, "
          f"{len(recording.trials)} trials, {len(recording.run_starts)} runs")


def _verify_live_output(directory, context, predictions, commands):
    # 交叉核对实际运行记录与发布日志，确认评分对象确实来自在线会话。
    directory = Path(directory)
    summary = json.loads((directory / "runtime_summary.json").read_text(encoding="utf-8"))
    if (summary.get("mode") != "live" or summary.get("model_version") != context["model_version"]
            or summary.get("status") != "complete" or summary.get("session_end_observed") is not True
            or summary.get("acquisition_thread_still_running") or summary.get("cleanup_errors")):
        raise ValueError("Online scoring requires a completed live runtime with the same frozen model")
    if (summary.get("predictions") != len(predictions)
            or summary.get("commands") != sum(c.direction != "NO_COMMAND" for c in commands)):
        raise ValueError("Runtime prediction/command counts disagree with event records")
    for key in ("pending_acquisition_packets", "failed_to_retain_samples", "unwritten_accepted_samples"):
        if summary.get(key) != 0:
            raise ValueError("Online runtime reports unrecorded or unfinished data")
    published = []
    session_id, sequence = None, 0
    with (directory / "results.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if (row.get("schema") != SCHEMA or row.get("mode") != "live"
                    or row.get("model_version") != context["model_version"]):
                raise ValueError("Published results are not from this live model")
            identity, number = row.get("session_id"), row.get("sequence")
            if (not isinstance(identity, str) or not identity or (session_id is not None and identity != session_id)
                    or isinstance(number, bool) or not isinstance(number, int) or number <= sequence):
                raise ValueError("Published result identity/sequence is invalid")
            session_id, sequence = identity, number
            if row.get("type") == "Command":
                published.append(row["payload"])
    if not published or published != [asdict(command) for command in commands]:
        raise ValueError("Published live commands and recorded commands do not match exactly")
    for command in commands:
        if command.direction in {"LEFT", "RIGHT"}:
            if any(getattr(command, name) is None for name in
                   ("emitted_at", "cue_to_output_seconds", "result_age_seconds")):
                raise ValueError("Live direction commands require measured output times and result age")
    return {"runtime_mode_verified": True, "published_commands_verified": True,
            "result_session_id": session_id, "hardware_or_human_validation": "not_established_by_software_logs"}


def cmd_evaluate_live(args):
    # 对独立 E 会话中当时实际产生的预测及指令评分。
    if Path(args.output).exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    recording = load_personal_session(args.session, args.labels)
    context = recording.metadata["capture_context"]
    if context["mode"] != "live" or recording.session_id != "E" or not context.get("model_version"):
        raise ValueError("Online scoring requires a live E capture with a frozen model version")
    predictions, commands = [], []
    with (Path(args.session) / "events.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            kind = event.pop("type", None)
            if kind == "Prediction":
                if event.get("model_version") != context["model_version"]:
                    raise ValueError("Predictions do not match the capture's frozen model version")
                predictions.append(Prediction(**event))
            elif kind == "Command":
                commands.append(Command(**event))
    output_verification = _verify_live_output(args.session, context, predictions, commands)
    config = Config.from_dict(context["config"])
    report = evaluate(recording.trials, predictions, commands, **_evaluation_parameters(config))
    if report["stability"]["unknown_trial_predictions"] or report["stability"]["unknown_trial_direction_commands"]:
        raise ValueError("Predictions or commands reference trials outside this capture")
    report.update(mode="personal_online_test", alignment_verified=True, label_alignment_verified=True,
                  alignment_verification_scope="Exact label/phase sequence only; physical cue alignment is not measured",
                  model_version=context["model_version"],
                  output_verification=output_verification,
                  subject_id=recording.subject_id, session_status="complete", data_provenance=recording.metadata)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    _brief(report)


def cmd_live(args):
    if Path(args.output).exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    bundle = ModelBundle.load(args.model)
    if args.subject != bundle.metadata.get("training_subject"):
        raise ValueError("The live subject must match the model's calibration subject")
    source = LiveSource(bundle.config, args.eeg_name, args.marker_name, args.reference or bundle.config.reference, args.units)
    try:
        context = _capture_context(args, bundle.config, source, "live", bundle.version)
        print("Connected. Waiting for external test markers. Predictions and commands are saved to the session directory; Ctrl+C stops early.", flush=True)
        run_session(source, bundle, args.output, context=context, result_stream_name=args.result_stream_name)
    finally:
        source.close()


def main(argv=None):
    # 注册子命令并分派处理函数；无参数时仅显示帮助，不启动设备或训练。
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="main.py", description="真实 EEG 运动想象命令行系统：LSL 原始数据采集与在线解码",
        epilog="首次使用：inspect-lsl → configure-lsl → preflight → experiment + record → import-session → train。\n"
               "在线测试：live --model <模型目录> --subject <受试者ID> --eeg-name <原始EEG流名> --output <新目录>。\n"
               "experiment 提供终端提示、阶段标记（默认 MI_Markers）及独立标签日志；本程序不提供 GUI。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Fit personal calibration CSP/LDA with run-grouped validation")
    train.add_argument("data", help="Personal calibration NPZ created by import-session")
    train.add_argument("--model", required=True)
    train.add_argument("--config", help="Override processing settings; defaults to the recorded acquisition configuration")
    train.add_argument("--min-trials-per-class-per-run", type=int, default=10)
    train.add_argument("--max-class-ratio", type=float, default=2.0)
    train.add_argument("--max-rejected-trial-fraction", type=float, default=0.5)
    train.add_argument("--max-rejected-window-fraction", type=float, default=0.5)
    train.add_argument("--audit-output", help="Insufficient-data report; default <model>.audit.json")
    train.set_defaults(func=cmd_train)
    replay = commands.add_parser("replay", help="Replay personal recorded EEG with neutral markers")
    replay.add_argument("data", help="Personal recording NPZ created by import-session")
    replay.add_argument("--model", required=True)
    replay.add_argument("--output", required=True)
    replay.set_defaults(func=cmd_replay)
    replay.add_argument("--realtime", action="store_true")
    replay.add_argument("--speed", type=float, default=1.0)
    evaluation = commands.add_parser("evaluate", help="Score personal replay predictions against the imported truth")
    evaluation.add_argument("--data", required=True, help="Personal recording NPZ created by import-session")
    evaluation.add_argument("--session", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.set_defaults(func=cmd_evaluate)
    inspect_lsl = commands.add_parser("inspect-lsl", help="List actual LSL stream names, sample rates, channels and units")
    inspect_lsl.add_argument("--timeout", type=float, default=3)
    inspect_lsl.set_defaults(func=cmd_inspect_lsl)
    configure = commands.add_parser("configure-lsl", help="Create a personal 16-channel config from the actual raw EEG stream")
    configure.add_argument("--eeg-name", required=True)
    labels = configure.add_mutually_exclusive_group(required=True)
    labels.add_argument("--channels", nargs="+", help="Verified electrode positions in transmitted CH1–CH16 order")
    labels.add_argument("--channels-file", help="JSON array of the 16 verified electrode positions")
    configure.add_argument("--reference", required=True, help="Actual physical reference; this option does not re-reference data")
    configure.add_argument("--units", choices=["V", "mV", "uV"], help="Verified input unit if missing from stream metadata")
    configure.add_argument("--output", required=True)
    configure.set_defaults(func=cmd_configure_lsl)
    experiment = commands.add_parser("experiment", help="Run balanced terminal cues with neutral LSL markers and separate truth")
    add_experiment_arguments(experiment)
    experiment.set_defaults(func=cmd_experiment)
    preflight = commands.add_parser("preflight", help="Measure real EEG timing and per-channel quality before collection")
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--eeg-name", required=True)
    preflight.add_argument("--duration", type=float, default=15.0)
    preflight.add_argument("--thresholds", help="Explicit quality threshold JSON")
    preflight.add_argument("--units", choices=["V", "mV", "uV"])
    preflight.add_argument("--output", required=True)
    preflight.set_defaults(func=cmd_preflight)
    subscription = commands.add_parser("subscribe", help="Subscribe to expiring MI_Results commands; stdout JSONL")
    subscription.add_argument("--stream-name", default="MI_Results")
    subscription.add_argument("--timeout", type=float, default=0.5)
    subscription.set_defaults(func=subscribe)
    record = commands.add_parser("record", help="Record personal calibration EEG and neutral cues without a model")
    record.add_argument("--config", required=True)
    record.add_argument("--subject", required=True)
    record.add_argument("--session-id", choices=["T", "E"], default="T")
    record.add_argument("--eeg-name", required=True)
    record.add_argument("--marker-name", default="MI_Markers")
    record.add_argument("--units", choices=["V", "mV", "uV"])
    record.add_argument("--output", required=True)
    record.set_defaults(func=cmd_record)
    personal_import = commands.add_parser("import-session", help="Join a completed personal capture with independent cue truth into NPZ")
    personal_import.add_argument("--session", required=True)
    personal_import.add_argument("--labels", required=True)
    personal_import.add_argument("--output", required=True)
    personal_import.set_defaults(func=cmd_import_session)
    online_score = commands.add_parser("evaluate-live", help="Score the predictions actually made during an independent online E capture")
    online_score.add_argument("--session", required=True)
    online_score.add_argument("--labels", required=True)
    online_score.add_argument("--output", required=True)
    online_score.set_defaults(func=cmd_evaluate_live)
    live = commands.add_parser("live", help="Acquire strict metadata-checked LSL EEG and neutral markers")
    live.add_argument("--model", required=True)
    live.add_argument("--subject", required=True, help="Must match the personal calibration subject")
    live.add_argument("--eeg-name", required=True)
    live.add_argument("--marker-name", default="MI_Markers")
    live.add_argument("--reference", help="Defaults to the verified reference stored in the personal model")
    live.add_argument("--units", choices=["V", "mV", "uV"])
    live.add_argument("--output", required=True)
    live.add_argument("--result-stream-name", default="MI_Results")
    live.set_defaults(func=cmd_live)
    if not argv:
        parser.print_help()
        return 0
    # 各子命令通过 set_defaults(func=...) 绑定处理函数，此处统一解析与调用。
    args = parser.parse_args(argv)
    try:
        result = args.func(args)
    except KeyboardInterrupt:
        print("Stopped; session records were finalized.", file=sys.stderr)
        return 130
    except ImportError as exc:
        parser.exit(1, f"Missing dependency: {exc}. Install with python -m pip install numpy scipy scikit-learn mne joblib pylsl\n")
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
