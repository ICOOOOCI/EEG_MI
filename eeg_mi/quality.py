"""真实设备预检、信号质量和时序报告。

质量规则集中在独立模块，供实时窗口校验与离线采集审计共同复用。
"""

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
import json
import time

import numpy as np
from scipy.signal import welch

from .config import Config
from .constants import UNIT_SCALE


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





__all__ = [
    "QualityThresholds", "channel_quality_report", "timing_report",
    "capture_preflight", "cmd_preflight", "write_capture_quality",
    "write_capture_timing",
]
