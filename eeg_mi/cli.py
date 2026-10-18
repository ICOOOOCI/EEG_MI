# 真实 EEG 运动想象命令行系统：采集、校准训练、连续回放与在线解码。
# 主要流程：inspect-lsl → configure-lsl → preflight → experiment + record
#           → import-session → train → experiment + live → evaluate-live。
# experiment 提供终端提示；阶段标记经 LSL 传输，四类真值单独保存到标签日志。
# 程序内部统一使用伏特（V），EEG 数组按“通道 × 采样点”排列。
# T 会话用于校准训练，E 会话用于独立测试；在线解码器只接收不含类别的阶段标记。
# 运行 python main.py 查看帮助；各子命令的参数可用 --help 查看。
# 默认 hybrid：Riemannian/TS/LR 四类概率 → EEGNet（保留同一 EEG 窗口输入）→ 四分类。
# Hybrid 校准至少 3 个 run；前级折外概率训练后级，完整模型按 run 留出验证。
#
from collections import Counter, OrderedDict, defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version as package_version
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
from scipy.signal import butter, sosfilt, welch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError

from .constants import *
from .config import Config
from .domain import (Command, EEGChunk, Marker, Prediction, Recording, Source, StreamInfo,
                     Trial, WindowModel, save_recording, load_recording)
from .models import (EEGNetClassifier, HybridClassifier, RiemannianClassifier, build_model,
                     validate_fitted_model)


__version__ = "0.6.0"



from .processing import (BufferedWindow, CausalPreprocessor, DecisionPolicy,
                        FilteredChunk, NEUTRAL_MARKERS, RingBuffer, Runtime,
                        iter_window_starts)
NEUTRAL_KINDS = set(NEUTRAL_MARKERS)

from .transport import AsyncRecorder, ReplaySource, json_value, write_json

# ========================================================================
from .lsl import LiveSource, parse_neutral_marker

from .training import (CalibrationDataError, ExtractedWindows, ModelBundle,
                      _WindowProbe, _evaluation_parameters, _fit_model,
                      _recording_config, _recording_digest, _validation_commands,
                      audit_calibration, canonical_json, dependency_versions, digest,
                      evaluate, extract_windows, load_bundle, load_personal_session,
                      make_pipeline, train_recording, validate_model_profile)

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


from .safety import SCHEMA, LiveSafety, MarkerProtocol

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
            info.desc().append_child_value("class_mapping", json.dumps(CLASS_MAPPING))
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
                if kind == "Command" and payload.get("direction") in CLASS_NAMES and lifetime <= 0:
                    payload.update(direction="NO_COMMAND", reason="EXPIRED_BEFORE_SEND")
                return {"schema": SCHEMA, "class_mapping": CLASS_MAPPING,
                        "session_id": self.session_id, "sequence": self.sequence,
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
        if row.get("schema") != SCHEMA or row.get("mode") != "live" or row.get("class_mapping") != CLASS_MAPPING:
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
        if direction not in (*CLASS_NAMES, "NO_COMMAND"):
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
    runs: int = 3
    trials_per_run: int = 40
    seed: int = 1
    imagery_seconds: float = 5.0
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
        if self.trials_per_run < N_CLASSES or self.trials_per_run % N_CLASSES:
            raise ValueError("trials_per_run 必须为至少 4 的四倍数，确保每组四类平衡")
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
        classes = list(TRUTH_IDS) * (config.trials_per_run // N_CLASSES)
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
                "truth_mapping": TRUTH_MAPPING, "class_mapping": CLASS_MAPPING,
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
            onset = emit("TRIAL_START", f"第 {run} 组，第 {trial['trial_index']}/{config.trials_per_run} 次：休息，放松身体。",
                         trial=active_trial)
            wait_until(onset + config.rest_seconds)
            direction = CLASS_CUES[trial["original_class"]]
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
    parser.add_argument("--runs", type=int, default=3, help="默认 3 组；Hybrid 校准验证至少需要 3 组")
    parser.add_argument("--trials-per-run", type=int, default=40)
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


from .quality import (QualityThresholds, capture_preflight, channel_quality_report,
                     cmd_preflight, timing_report, write_capture_quality, write_capture_timing)

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
                                 algorithm=args.algorithm, seed=args.seed, eegnet_epochs=args.eegnet_epochs,
                                 eegnet_batch_size=args.eegnet_batch_size,
                                 eegnet_learning_rate=args.eegnet_learning_rate,
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


from .lsl import _lsl_description, _lsl_header

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
        print("No LSL streams found. Start the device's raw EEG LSL outlet (16 channels, 250 Hz) first.")


def cmd_configure_lsl(args):
    # 读取实际设备流生成配置，电极映射及参考必须由使用者确认。
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(f"Configuration already exists: {target}")
    header = _lsl_header(args.eeg_name)
    rate = float(header["sfreq"])
    if not np.isfinite(rate) or not math.isclose(rate, 250.0, rel_tol=0, abs_tol=1e-8):
        raise ValueError("This setup requires a raw EEG stream at 250 Hz; configure the device before continuing")
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
               (("warmup_seconds", 1), ("window_seconds", 3), ("step_seconds", .2),
                ("cue_offset_seconds", .5), ("imagery_seconds", 5), ("buffer_seconds", 10))}
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
            "application_version": __version__, "truth_mapping": TRUTH_MAPPING, "class_mapping": CLASS_MAPPING,
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
                    or row.get("class_mapping") != CLASS_MAPPING
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
        if command.direction in CLASS_NAMES:
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


def cmd_gui(args):
    from collector_gui import launch
    return launch()


def main(argv=None):
    # 注册子命令并分派处理函数；无参数时仅显示帮助，不启动设备或训练。
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="main.py", description="真实 EEG 运动想象命令行系统：LSL 原始数据采集与在线解码",
        epilog="首次使用：inspect-lsl → configure-lsl → preflight → experiment + record → import-session → train。\n"
               "在线测试：live --model <模型目录> --subject <受试者ID> --eeg-name <原始EEG流名> --output <新目录>。\n"
               "experiment 提供终端提示、阶段标记（默认 MI_Markers）及独立标签日志。\n"
               "GUI 采集：python main.py gui，接收 OpenBCI GUI Networking 的 TimeSeriesRaw LSL 输出。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    gui = commands.add_parser("gui", help="打开 OpenBCI LSL 接收、范式与原始 EEG 保存界面")
    gui.set_defaults(func=cmd_gui)
    train = commands.add_parser("train", help="Train the ML-to-EEGNet cascade (default), or a single-model baseline")
    train.add_argument("data", help="Personal calibration NPZ created by import-session")
    train.add_argument("--model", required=True)
    train.add_argument("--config", help="Processing config; default: recorded config with a 3-second window")
    train.add_argument("--algorithm", choices=ALGORITHMS, default="hybrid",
                       help="默认 hybrid：机器学习 → EEGNet 串联；另两个值为单模型对照")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--eegnet-epochs", type=int, default=100, help="EEGNet or hybrid neural-stage epochs; never selected using E data")
    train.add_argument("--eegnet-batch-size", type=int, default=32)
    train.add_argument("--eegnet-learning-rate", type=float, default=1e-3)
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
    configure = commands.add_parser("configure-lsl", help="Verify actual 16-channel/250-Hz LSL stream and configure 3-second windows")
    configure.add_argument("--eeg-name", required=True)
    labels = configure.add_mutually_exclusive_group()
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
        parser.exit(1, f"Missing dependency: {exc}. Install with python -m pip install -r requirements.lock.txt\n")
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
