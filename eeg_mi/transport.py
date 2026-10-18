"""采集传输适配器与持久化记录器。

ReplaySource、AsyncRecorder 通过统一领域对象工作，CLI/GUI 不需要知道文件布局
或线程队列细节。真实 LSL LiveSource 仍由 CLI 编排层组合实时安全协议。
"""

from dataclasses import asdict, is_dataclass
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
import json
import queue
import threading
import time
import uuid
from uuid import uuid4

import numpy as np

from .domain import EEGChunk, Marker, Recording, StreamInfo
from .processing import NEUTRAL_MARKERS


# ========================================================================
# 09. 按原始时间线连续回放
# ========================================================================
class ReplaySource:
    # 按阶段边界拆分数据包，确保标记先于该位置的 EEG 到达解码器。
    def __init__(self, recording, chunk_samples=25, realtime=False, speed=1.0, imagery_seconds=5.0):
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



NEUTRAL_KINDS = set(NEUTRAL_MARKERS)




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


__all__ = ["ReplaySource", "json_value", "write_json", "AsyncRecorder"]
