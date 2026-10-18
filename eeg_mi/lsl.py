"""LSL 元数据读取与实时 EEG/Marker 传输适配器。"""

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





from dataclasses import replace
import json
import threading

import numpy as np

from .constants import UNIT_SCALE
from .domain import EEGChunk, Marker, StreamInfo
from .processing import NEUTRAL_MARKERS
from .safety import LiveSafety

NEUTRAL_KINDS = set(NEUTRAL_MARKERS)

# 10. LSL 设备采集与阶段标记对齐
# ========================================================================
def parse_neutral_marker(payload):
    # 只允许 kind 和 trial_id，防止四类真值通过阶段流进入解码器。
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


__all__ = ["_lsl_description", "_lsl_header", "LiveSource", "parse_neutral_marker"]
