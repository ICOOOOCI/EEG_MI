"""实时 EEG 预处理、窗口缓存和阶段状态机。

Runtime 只依赖领域对象、配置和质量接口，因此离线回放与实时 LSL 可共享。
"""

from dataclasses import dataclass
from math import isfinite
from time import perf_counter

import numpy as np
from scipy.signal import butter, sosfilt

from .config import Config
from .constants import CLASS_NAMES, N_CLASSES
from .domain import Command, EEGChunk, Marker, Prediction
from .quality import QualityThresholds, channel_quality_report


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
        if p.shape != (N_CLASSES,) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1) or not np.isclose(p.sum(), 1):
            self.clear_streak()
            return output("NO_COMMAND", "MODEL_ERROR")
        label = int(np.argmax(p))
        if p[label] < self.config.threshold or np.count_nonzero(p == p[label]) != 1:
            self.clear_streak()
            return output("NO_COMMAND", "LOW_CONFIDENCE")
        if self._direction == label:
            self._streak += 1
        else:
            self._direction, self._streak = label, 1
        if self._streak < self.config.confirmations:
            return output("NO_COMMAND", "AWAITING_CONFIRMATION")
        self._committed_trials.add(self.trial_id)
        return output(CLASS_NAMES[label], "CONFIRMED")

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
            # 标记向后量化到采样网格最多占一个采样间隔，独立 LSL 时钟校正
            # 还可能带来允许范围内的微小漂移；两项预算相加，与采集时序报告一致。
            tolerance = 1.0 / self.config.sfreq + self.config.timestamp_tolerance_seconds + 1e-9
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
                if p.shape != (1, N_CLASSES) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1) or not np.isclose(p.sum(), 1):
                    raise ValueError("Model probabilities must be a normalized 1 x 4 array")
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





__all__ = [
    "FilteredChunk", "CausalPreprocessor", "BufferedWindow", "RingBuffer",
    "DecisionPolicy", "NEUTRAL_MARKERS", "iter_window_starts", "Runtime",
]
