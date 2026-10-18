"""OpenBCI GUI Networking 的 LSL 接收器：只负责范式和 EEG 记录。"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import json
import math
import queue
import random
import sys
import tempfile
import threading
import time
import uuid
import wave

import numpy as np
import pylsl
from PyQt6.QtCore import Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QDesktopServices, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtMultimedia import QSoundEffect
from PyQt6.QtWidgets import (QApplication, QComboBox, QFileDialog, QFormLayout, QFrame,
                             QHBoxLayout, QLabel, QLineEdit, QMainWindow, QProgressBar,
                             QPushButton, QSpinBox, QVBoxLayout, QWidget)

from .constants import UNIT_SCALE
from .domain import EEGChunk, StreamInfo
from .lsl import _lsl_description
from .transport import AsyncRecorder, write_json

CLASSES = {1: ("←", "左手"), 2: ("→", "右手"), 3: ("↓", "双脚"), 4: ("↑", "舌头")}
TIMING_NOTE = ("GUI 事件记录 repaint 返回后的 pylsl.local_clock，提示音记录播放请求。"
               "EEG 时间戳经 LSL 时钟校正，不去抖、不重排。未测量上游设备/传输缓冲、实际显示和声音延迟。")


def stream_metadata(header):
    """兼容 OpenBCI GUI source_id=openbcigui 且没有 channels 描述的流。"""
    if header.channel_format() == pylsl.cf_string or header.type().lower() != "eeg":
        raise ValueError("请选择 OpenBCI Networking 的 EEG 数值流")
    if not math.isfinite(header.nominal_srate()) or header.nominal_srate() <= 0:
        raise ValueError("EEG 流没有有效采样率")
    description = _lsl_description(header)
    description.update(uid=header.uid(), xml=header.as_xml())
    channels, scales, units, fallback = [], [], [], []
    for i, channel in enumerate(description["channels"], 1):
        channels.append(channel["label"] or f"CH{i}")
        unit = channel["unit"]
        if not unit and header.source_id() == "openbcigui":
            unit = "uV"
            fallback.append(i)
        if unit not in UNIT_SCALE:
            raise ValueError("该流缺少可识别单位且来源不是 OpenBCI GUI，请检查 Networking 输出")
        units.append(unit)
        scales.append(UNIT_SCALE[unit])
    if len(set(channels)) != len(channels):
        channels = [f"CH{i + 1}" for i in range(header.channel_count())]
    description.update(channel_names=channels, input_units=units, unit_fallback_channels=fallback,
                       unit_fallback_basis="OpenBCI GUI TimeSeriesRaw uses microvolts",
                       reference_recorded=bool(description["reference"]),
                       channel_positions_verified=False)
    return description, np.asarray(scales)[:, None]


def make_trials(runs, per_class, seed, experiment_id):
    if runs < 1 or per_class < 1:
        raise ValueError("组数和每类试次数必须大于零")
    rng = random.Random(seed)
    trials = []
    for run in range(1, runs + 1):
        labels = list(CLASSES) * per_class
        rng.shuffle(labels)
        for index, original_class in enumerate(labels, 1):
            trials.append(dict(run=run, index=index, original_class=original_class,
                               trial_id=f"{experiment_id}:r{run}:t{index}",
                               rest_seconds=round(rng.uniform(1.5, 2.5), 3)))
    return trials


def paradigm_steps(trials):
    """延时相对上一阶段实际呈现时刻；图片中的类别提示延续至 3.25 秒。"""
    steps, previous_run = deque(), None
    for trial in trials:
        prepare = trial["run"] != previous_run
        if prepare:
            steps.append((0, "RUN_START", trial))
            previous_run = trial["run"]
        steps.extend([(3 if prepare else 0, "TRIAL_START", trial), (2, "CUE_ON", trial),
                      (1, "MI_START", trial), (.25, "CUE_OFF", trial), (2.75, "MI_END", trial),
                      (trial["rest_seconds"], "BREAK_END", trial)])
    steps.append((0, "SESSION_END", None))
    return steps


def export_recording(directory, metadata):
    """采集 NPZ 独立于模型训练格式；不修改或推测上游时间戳。"""
    directory = Path(directory)
    raw_manifest = json.loads((directory / "manifest.json").read_text())
    if raw_manifest["status"] != "complete":
        raise ValueError("只有完整采集才生成 NPZ")
    count = raw_manifest["raw_samples"]
    channels = len(metadata["stream"]["channel_names"])
    data = np.memmap(directory / "raw_eeg.f64", dtype="<f8", mode="r", shape=(count, channels))
    stamps = np.memmap(directory / "timestamps.f64", dtype="<f8", mode="r", shape=(count,))
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    monotonic = bool(np.isfinite(stamps).all() and np.all(np.diff(stamps) > 0))
    for event in events:
        event["sample_index"] = None
        if "timestamp" in event and monotonic:
            index = int(np.searchsorted(stamps, event["timestamp"], side="left"))
            if index < count:
                event["sample_index"] = index
    metadata["timestamps_strictly_increasing"] = monotonic
    metadata["event_alignment"] = "first sample at/after event in corrected LSL time; null if outside data or timestamps not increasing"
    temporary = directory / "recording.npz.tmp"
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, data=data.T, timestamps=stamps,
                                events_json=np.array(json.dumps(events, ensure_ascii=False)),
                                metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)))
        temporary.replace(directory / "recording.npz")
    finally:
        temporary.unlink(missing_ok=True)


class ScanWorker(QThread):
    result = pyqtSignal(object)
    failed = pyqtSignal(str)

    def run(self):
        try:
            streams = [s for s in pylsl.resolve_streams(wait_time=1)
                       if s.type().lower() == "eeg" and s.channel_format() != pylsl.cf_string]
            streams.sort(key=lambda s: (s.name() != "obci_eeg1", s.name()))
            self.result.emit(streams)
        except Exception as exc:
            self.failed.emit(str(exc))


class Receiver(QThread):
    connected = pyqtSignal(object)
    recording = pyqtSignal()
    result = pyqtSignal(object)
    saving = pyqtSignal()

    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.commands = queue.Queue()
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.samples = 0
        self.directory = None
        self.metadata = None

    def take_preview(self):
        with self.lock:
            value, self.latest = self.latest, None
        return value

    def run(self):
        inlet = recorder = None
        result = {"status": "disconnected", "error": None}
        announced = first_record = False
        previous_timestamp = None
        breaks = 0
        events_count = 0
        expected_trials = 0
        completed_trials = 0
        saw_session_end = False
        started_at = None
        try:
            inlet = pylsl.StreamInlet(self.stream, max_buflen=10, recover=False,
                                      processing_flags=pylsl.proc_clocksync)
            header = inlet.info(timeout=3)
            description, scales = stream_metadata(header)
            info = StreamInfo(header.nominal_srate(), tuple(description["channel_names"]),
                                   description["reference"] or "upstream reference; position not recorded")
            inlet.open_stream(timeout=3)
            correction = inlet.time_correction(timeout=3)
            inlet.flush()
            buffer = np.empty((header.channel_count(), 0))
            last_data = time.monotonic()
            maximum_silence = 0.0
            max_samples = max(1, round(info.sfreq * .05))
            while not self.stop_event.is_set():
                while not self.commands.empty():
                    command, payload = self.commands.get_nowait()
                    if command == "start":
                        if recorder is not None:
                            raise RuntimeError("已有记录正在运行")
                        self.directory = Path(payload.pop("directory"))
                        recorder = AsyncRecorder(self.directory, info)
                        self.metadata = {"format": "openbci-lsl-capture-v1", "status": "recording",
                                         "stream": description, "saved_unit": "V", "layout": "channels x samples",
                                         "lsl_processing": "proc_clocksync only; no dejitter/resampling/filter",
                                         "initial_clock_correction_seconds": correction, "timing_note": TIMING_NOTE,
                                         "created_utc": datetime.now(timezone.utc).isoformat(), **payload}
                        expected_trials = len(payload["trials"])
                        write_json(self.directory / "session.json", self.metadata)
                        inlet.flush()
                        first_record = True
                        started_at = pylsl.local_clock()
                    elif command == "event" and recorder is not None:
                        # 不将接收时的样本总数假装成刺激对齐位置；最终根据保存的时间戳对齐。
                        recorder.record({"type": "GUIEvent", **payload}, timeout=.2)
                        events_count += 1
                        completed_trials += payload["kind"] == "MI_END"
                        saw_session_end |= payload["kind"] == "SESSION_END"
                    elif command == "stop":
                        result["status"] = payload["status"]
                        result["error"] = payload.get("error")
                        self.stop_event.set()
                if self.stop_event.is_set():
                    break
                values, stamps = inlet.pull_chunk(timeout=.05, max_samples=max_samples)
                now = time.monotonic()
                if stamps:
                    data = np.asarray(values, dtype=float).T * scales
                    times = np.asarray(stamps, dtype=float)
                    if data.shape != (header.channel_count(), len(times)):
                        raise RuntimeError("收到的数据维度与 LSL 描述不符")
                    if recorder is not None:
                        chunk = EEGChunk(data, times, self.samples, info, received_at=pylsl.local_clock())
                        self.samples += len(times)
                        recorder.record(chunk, timeout=.2)
                        if not np.isfinite(times).all():
                            raise RuntimeError("收到非有限时间戳；原始样本已保留")
                        intervals = np.diff(times if previous_timestamp is None else np.r_[previous_timestamp, times])
                        breaks += int(np.sum((intervals <= 0) | (intervals > 1.5 / info.sfreq)))
                        previous_timestamp = float(times[-1])
                        if first_record:
                            first_record = False
                            self.recording.emit()
                    buffer = np.concatenate((buffer, data), axis=1)[:, -round(5 * info.sfreq):]
                    with self.lock:
                        self.latest = buffer.copy()
                    maximum_silence = max(maximum_silence, now - last_data)
                    last_data = now
                    if not announced:
                        announced = True
                        self.connected.emit(description)
                elif now - last_data > (3 if not announced else 1.5):
                    raise RuntimeError("LSL 流存在但未收到 EEG，请检查 OpenBCI 的 Start Data Stream 和 Start LSL Stream")
            if recorder is not None and result["status"] == "disconnected":
                result["status"] = "cancelled"
            if result["status"] == "complete" and (not saw_session_end or completed_trials != expected_trials or not self.samples):
                raise RuntimeError("范式或 EEG 未完整结束，记录已保留")
        except Exception as exc:
            result.update(status="failed", error=str(exc))
        finally:
            if inlet is not None:
                try:
                    inlet.close_stream()
                except Exception as exc:
                    result.update(status="failed", error=str(exc))
            if recorder is not None:
                self.saving.emit()
                # 故障发生时也保留主线程已经发出的提示事件。
                while not self.commands.empty():
                    command, payload = self.commands.get_nowait()
                    if command == "event":
                        try:
                            recorder.record({"type": "GUIEvent", **payload}, timeout=.2)
                            events_count += 1
                        except Exception as exc:
                            result.update(status="failed", error=str(exc))
                try:
                    recorder.record({"type": "SessionEnd", "status": result["status"], "error": result["error"]}, timeout=.2)
                except Exception as exc:
                    result.update(status="failed", error=str(exc))
                try:
                    recorder.close(status=result["status"], error=result["error"], received_raw_samples=self.samples)
                except Exception as exc:
                    result.update(status="failed", error=str(exc))
                if self.metadata is not None:
                    self.metadata.update(status=result["status"], error=result["error"],
                                         received_samples=self.samples, completed_trials=completed_trials,
                                         event_count=events_count, timestamp_interval_anomalies=breaks,
                                         maximum_silence_seconds=maximum_silence,
                                         recording_start_lsl=started_at,
                                         finished_utc=datetime.now(timezone.utc).isoformat())
                    try:
                        write_json(self.directory / "session.json", self.metadata)
                        if result["status"] == "complete":
                            export_recording(self.directory, self.metadata)
                            self.metadata["npz_export_status"] = "complete"
                            write_json(self.directory / "session.json", self.metadata)
                    except Exception as exc:
                        result.update(status="export_failed" if result["status"] == "complete" else "failed", error=str(exc))
                        self.metadata.update(npz_export_status="failed", export_error=str(exc))
                        try:
                            write_json(self.directory / "session.json", self.metadata)
                        except OSError:
                            pass
                result.update(directory=str(self.directory), samples=self.samples)
            self.result.emit(result)


class Waveform(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(215)
        self.data, self.names = None, []

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QColor("#8495a1"))
        if self.data is None:
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "等待 OpenBCI GUI 的 LSL 数据\n连接预览时不保存")
            return
        width, height = self.width() - 75, self.height() - 24
        rows, count = self.data.shape
        step = height / rows
        for column in range(6):
            x = 60 + width * column / 5
            p.setPen(QPen(QColor("#e8eef3"), 1))
            p.drawLine(int(x), 0, int(x), height)
        for row, name in enumerate(self.names):
            y = (row + .5) * step
            font = QFont("Arial")
            font.setPixelSize(max(8, min(12, int(step * .9))))
            p.setFont(font)
            p.setPen(QColor("#627384"))
            p.drawText(0, int(y + 4), name[:9])
            stride = max(1, count // max(1, width))
            v = self.data[row, ::stride] * 1e6
            center = np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else 0
            ys = y - np.clip((v - center) / 100, -1, 1) * step * .42
            xs = np.linspace(60, width + 60, len(v))
            path, drawing = QPainterPath(), False
            for x, yy in zip(xs, ys):
                if not np.isfinite(yy):
                    drawing = False
                elif drawing:
                    path.lineTo(x, yy)
                else:
                    path.moveTo(x, yy)
                    drawing = True
            p.setPen(QPen(QColor("#168780"), 1))
            p.drawPath(path)
        p.setFont(QFont("Arial", 10))
        p.setPen(QColor("#8495a1"))
        p.drawText(60, height + 19, "最近接收的样本")
        p.drawText(width + 27, height + 19, "现在")


STYLE = """
QMainWindow { background: #f2f6f8; }
QWidget { color: #203c4b; font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif; font-size: 13px; }
QFrame#card { background: white; border: 1px solid #e0e8ed; border-radius: 12px; }
QLabel#title { font-size: 25px; font-weight: 600; }
QLabel#muted { color: #748894; font-size: 12px; }
QLabel#heading { font-size: 15px; font-weight: 600; }
QLineEdit, QComboBox, QSpinBox { background: #f9fbfc; border: 1px solid #d8e2e8; border-radius: 6px; padding: 7px; min-height: 18px; }
QPushButton { border: 1px solid #cfdee5; border-radius: 7px; padding: 9px 15px; background: white; }
QPushButton:hover { background: #e7f0f2; }
QPushButton:disabled { color: #a5b0b8; background: #f0f4f6; }
QPushButton#primary { color: white; background: #187e78; border: 0; }
QPushButton#primary:disabled { background: #b9cccc; }
QProgressBar { border: 0; border-radius: 3px; background: #e1eaf0; max-height: 6px; }
QProgressBar::chunk { background: #218e87; }
"""


def text_label(text, style=None):
    w = QLabel(text)
    if style:
        w.setObjectName(style)
    return w


def card():
    w = QFrame()
    w.setObjectName("card")
    layout = QVBoxLayout(w)
    layout.setContentsMargins(18, 16, 18, 16)
    layout.setSpacing(12)
    return w, layout


class CollectorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG 采集 · OpenBCI LSL")
        self.resize(1150, 820)
        self.setMinimumSize(1040, 800)
        self.setStyleSheet(STYLE)
        self.receiver = self.scanner = None
        self.last_directory = None
        self.metadata = None
        self.state = "idle"
        self.steps = deque()
        self.deadline = None
        self.close_pending = False
        self.stop_at = None
        self.terminal_result = None
        self._build()
        self._audio()
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(5)
        self.timer.timeout.connect(self.tick)
        self.timer.start()
        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(100)
        self.preview_timer.timeout.connect(self.refresh)
        self.preview_timer.start()
        self.set_state("idle", "先在 OpenBCI GUI 开启数据采集和 LSL 输出，再点击扫描。")

    def _build(self):
        w = QWidget()
        self.setCentralWidget(w)
        root = QVBoxLayout(w)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(16)
        root.addWidget(text_label("EEG 采集", "title"))
        root.addWidget(text_label("第三方设备 → OpenBCI GUI（连接 / 阻抗）→ LSL → 本界面（范式 / 保存）", "muted"))
        body = QHBoxLayout()
        body.setSpacing(16)
        left, side = card()
        left.setFixedWidth(315)
        side.setSpacing(10)
        side.addWidget(text_label("接收 OpenBCI 数据", "heading"))
        guide = text_label("Networking 设置\nTimeSeriesRaw · EEG · obci_eeg1", "muted")
        side.addWidget(guide)
        self.streams = QComboBox()
        self.streams.addItem("等待扫描 obci_eeg1", None)
        side.addWidget(self.streams)
        actions = QHBoxLayout()
        self.scan_button = QPushButton("扫描 LSL")
        self.scan_button.clicked.connect(self.scan)
        self.connect_button = QPushButton("接收并预览")
        self.connect_button.clicked.connect(self.connect_stream)
        actions.addWidget(self.scan_button)
        actions.addWidget(self.connect_button)
        side.addLayout(actions)
        self.stream_info = text_label("自动读取通道数和采样率", "muted")
        self.stream_info.setWordWrap(True)
        side.addWidget(self.stream_info)
        side.addSpacing(12)
        side.addWidget(text_label("本次记录", "heading"))
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setSpacing(8)
        self.subject = QLineEdit("S01")
        self.runs = QSpinBox()
        self.runs.setRange(1, 30)
        self.per_class = QSpinBox()
        self.per_class.setRange(1, 100)
        self.per_class.setValue(5)
        form.addRow("受试者编号", self.subject)
        form.addRow("组数", self.runs)
        form.addRow("每组每类次数", self.per_class)
        side.addLayout(form)
        self.total = text_label("", "muted")
        self.total.setWordWrap(True)
        side.addWidget(self.total)
        self.runs.valueChanged.connect(self.update_total)
        self.per_class.valueChanged.connect(self.update_total)
        self.update_total()
        side.addWidget(text_label("保存位置", "muted"))
        self.folder = QLineEdit(str(Path(__file__).resolve().parent / "recordings"))
        side.addWidget(self.folder)
        self.browse = QPushButton("选择文件夹")
        self.browse.clicked.connect(self.choose_folder)
        side.addWidget(self.browse)
        side.addStretch()
        note = text_label("保持 OpenBCI GUI 和 LSL 运行。\n设备连接与阻抗检测沿用你的原有流程。", "muted")
        note.setWordWrap(True)
        side.addWidget(note)
        body.addWidget(left)
        right = QVBoxLayout()
        right.setSpacing(14)
        self.stimulus, stim = card()
        self.stimulus.setStyleSheet("QFrame#card {background: #153343; border: 0; border-radius: 12px;}")
        self.stimulus.setMinimumHeight(225)
        self.phase = text_label("等待开始")
        self.phase.setStyleSheet("color: #8bcac3;")
        self.symbol = text_label("+")
        self.symbol.setStyleSheet("color: white; font-size: 82px; font-weight: 300;")
        self.symbol.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.instruction = text_label("接收 EEG 后开始范式采集")
        self.instruction.setStyleSheet("color: #c5d7df; font-size: 15px;")
        self.instruction.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stim.addWidget(self.phase)
        stim.addWidget(self.symbol, 1)
        stim.addWidget(self.instruction)
        right.addWidget(self.stimulus)
        timeline, tl = card()
        tl.setContentsMargins(18, 12, 18, 12)
        tl.addWidget(text_label("0s 提示音  →  0–2s 注视  →  2–3.25s 类别提示", "muted"))
        tl.addWidget(text_label("3–6s 运动想象  →  1.5–2.5s 随机休息", "muted"))
        right.addWidget(timeline)
        waves, wl = card()
        wl.addWidget(text_label("收到的 EEG  ·  最近 5 秒  ·  显示去直流 / 每通道 ±100 μV", "muted"))
        self.waveform = Waveform()
        wl.addWidget(self.waveform)
        right.addWidget(waves, 1)
        body.addLayout(right, 1)
        root.addLayout(body, 1)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setValue(0)
        root.addWidget(self.progress)
        footer = QHBoxLayout()
        self.status = text_label("", "muted")
        self.status.setWordWrap(True)
        footer.addWidget(self.status, 1)
        self.open_button = QPushButton("打开记录")
        self.open_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_directory))) if self.last_directory else None)
        self.stop_button = QPushButton("停止并保存")
        self.stop_button.clicked.connect(self.stop)
        self.start_button = QPushButton("开始采集")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self.start)
        for button in (self.open_button, self.stop_button, self.start_button):
            footer.addWidget(button)
        root.addLayout(footer)

    def _audio(self):
        self.audio_dir = tempfile.TemporaryDirectory(prefix="eeg-cue-")
        path = Path(self.audio_dir.name) / "cue.wav"
        t = np.arange(5292) / 44100
        envelope = np.minimum(1, t / .008) * np.minimum(1, (.12 - t) / .008)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(44100)
            wav.writeframes((6000 * envelope * np.sin(2 * np.pi * 800 * t)).astype("<i2").tobytes())
        self.sound = QSoundEffect(self)
        self.sound.setSource(QUrl.fromLocalFile(str(path)))
        self.sound.setVolume(.5)

    def set_state(self, state, message=None):
        self.state = state
        idle = state in {"idle", "done", "error"}
        self.scan_button.setEnabled(idle and not (self.scanner and self.scanner.isRunning()))
        self.streams.setEnabled(idle)
        self.connect_button.setEnabled((idle and self.streams.currentData() is not None) or state == "preview")
        self.connect_button.setText("断开预览" if state == "preview" else "接收并预览")
        for widget in (self.subject, self.runs, self.per_class, self.folder, self.browse):
            widget.setEnabled(idle or state == "preview")
        self.start_button.setEnabled(state == "preview")
        self.stop_button.setEnabled(state in {"starting", "recording"})
        self.open_button.setEnabled(self.last_directory is not None and self.last_directory.exists())
        if message is not None:
            self.status.setText(message)

    def update_total(self):
        n = self.runs.value() * self.per_class.value() * 4
        self.total.setText(f"左手 / 右手 / 双脚 / 舌头\n均衡随机，共 {n} 次，约 {(n * 8 + self.runs.value() * 3) / 60:.1f} 分钟")

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "保存位置", self.folder.text())
        if folder:
            self.folder.setText(folder)

    def scan(self):
        self.set_state("scanning", "正在查找 OpenBCI GUI 的 EEG 流…")
        self.scanner = ScanWorker()
        self.scanner.result.connect(self.found)
        self.scanner.failed.connect(lambda error: self.set_state("error", error))
        self.scanner.finished.connect(lambda: self.set_state(self.state))
        self.scanner.start()

    def found(self, streams):
        self.streams.clear()
        for stream in streams:
            self.streams.addItem(f"{stream.name()} · {stream.channel_count()} 通道 · {stream.nominal_srate():g} Hz", stream)
        if not streams:
            self.streams.addItem("未发现 obci_eeg1 / EEG 流", None)
        self.set_state("idle", "已发现 EEG 流，请接收并预览。" if streams else "未发现 LSL：请在 OpenBCI 点击 Start Data Stream 和 Start LSL Stream。")

    def connect_stream(self):
        if self.state == "preview":
            self.receiver.stop_event.set()
            self.set_state("saving", "正在断开 LSL 接收…")
            return
        stream = self.streams.currentData()
        if stream is None:
            return
        self.waveform.data = None
        self.waveform.update()
        self.terminal_result = None
        self.receiver = Receiver(stream)
        self.receiver.connected.connect(self.connected)
        self.receiver.recording.connect(self.recording_started)
        self.receiver.saving.connect(lambda: self.set_state("saving", "正在核对并保存已收到的数据…"))
        self.receiver.result.connect(self.receiver_result)
        self.receiver.finished.connect(self.receiver_finished)
        self.set_state("connecting", "正在接收 OpenBCI LSL 数据…")
        self.receiver.start()

    def connected(self, metadata):
        self.metadata = metadata
        self.waveform.names = metadata["channel_names"]
        self.stream_info.setText(f"{metadata['channel_count']} 通道 · {metadata['sfreq']:g} Hz\n保留上游通道顺序和参考")
        self.stream_info.setToolTip(json.dumps(metadata, ensure_ascii=False, indent=2))
        if self.state == "connecting":
            self.set_state("preview", "已收到 EEG。预览不保存，点击开始采集后记录。")

    def start(self):
        if self.state != "preview":
            return
        subject = self.subject.text().strip()
        if not subject or any(c in subject for c in '/\\\0'):
            self.status.setText("请填写有效受试者编号，不使用路径分隔符。")
            return
        if not self.folder.text().strip():
            self.status.setText("请选择保存位置。")
            return
        if self.sound.status() != QSoundEffect.Status.Ready:
            self.status.setText("提示音未就绪，请检查音频输出后重试。")
            return
        identifier = uuid.uuid4().hex
        self.seed = random.SystemRandom().randrange(2**31)
        self.trials = make_trials(self.runs.value(), self.per_class.value(), self.seed, identifier)
        self.steps = paradigm_steps(self.trials)
        self.completed = 0
        self.deadline = self.stop_at = None
        self.last_directory = Path(self.folder.text()).expanduser().resolve() / f"{subject}_{datetime.now():%Y%m%d_%H%M%S}_{identifier[:6]}"
        self.receiver.commands.put(("start", {"directory": str(self.last_directory), "subject": subject,
                                             "trials": self.trials, "seed": self.seed,
                                             "expected_networking_data_type": "TimeSeriesRaw",
                                             "paradigm": {"cue_on": 2, "mi_on": 3, "cue_off": 3.25,
                                                          "mi_end": 6, "break_seconds": [1.5, 2.5]}}))
        self.progress.setRange(0, len(self.trials))
        self.progress.setValue(0)
        self.set_state("starting", "正在创建记录，等待首批保存的 EEG…")

    def recording_started(self):
        if self.state == "starting":
            self.deadline = pylsl.local_clock()
            self.began = self.deadline
            self.set_state("recording", "正在记录原始 EEG")

    def mark(self, kind, trial, stamp, planned=None):
        event = {"kind": kind, "timestamp": stamp, "planned_timestamp": planned,
                 "trial_id": trial["trial_id"] if trial else None,
                 "original_class": trial["original_class"] if trial else None,
                 "run": trial["run"] if trial else None}
        self.receiver.commands.put(("event", event))

    def present(self, kind, trial):
        symbol, name = CLASSES[trial["original_class"]] if trial else ("", "")
        view = {
            "RUN_START": ("准备", "+", "保持放松，等待提示音"),
            "TRIAL_START": ("注视", "+", "注视十字，保持身体不动"),
            "CUE_ON": ("类别提示", symbol, f"准备想象{name}运动"),
            "MI_START": ("运动想象", symbol, f"开始想象{name}运动，身体保持不动"),
            "CUE_OFF": ("运动想象", "+", "继续想象刚才提示的动作"),
            "MI_END": ("休息", "·", "停止想象，放松"),
            "SESSION_END": ("范式结束", "✓", "正在保存数据"),
        }
        if kind in view:
            phase, symbol, instruction = view[kind]
            self.phase.setText(phase + (f"  ·  第 {trial['run']} 组 / 第 {trial['index']} 次" if trial else ""))
            self.symbol.setText(symbol)
            self.instruction.setText(instruction)
            self.stimulus.layout().activate()
            self.stimulus.repaint()

    def tick(self):
        if self.state != "recording":
            return
        now = pylsl.local_clock()
        if self.stop_at is not None:
            if now >= self.stop_at:
                self.stop(status="complete")
            return
        if self.deadline is None or now < self.deadline:
            return
        if now - self.deadline > .25:
            self.stop(status="failed", error="界面延迟超过 250 ms，范式已中止，已收到的数据会保留。")
            return
        try:
            _, kind, trial = self.steps.popleft()
            self.present(kind, trial)
            stamp = pylsl.local_clock()
            self.mark(kind, trial, stamp, self.deadline)
            if kind == "TRIAL_START":
                beep_stamp = pylsl.local_clock()
                self.sound.play()
                self.mark("BEEP_REQUEST", trial, beep_stamp)
            if kind == "MI_END":
                self.completed += 1
                self.progress.setValue(self.completed)
            if kind == "SESSION_END":
                self.deadline = None
                self.stop_at = stamp + .3  # 保存提示结束后的尾部样本。
            else:
                self.deadline = stamp + self.steps[0][0]
        except Exception as exc:
            self.stop(status="failed", error=str(exc))

    def stop(self, checked=False, *, status="cancelled", error=None):
        if self.state not in {"starting", "recording"}:
            return
        self.steps.clear()
        self.deadline = self.stop_at = None
        self.sound.stop()
        self.receiver.commands.put(("stop", {"status": status, "error": error}))
        self.set_state("saving", "正在保存；提前停止的记录会标为未完成。" if status != "complete" else "正在保存完整记录…")

    def receiver_result(self, result):
        self.deadline = self.stop_at = None
        self.steps.clear()
        self.terminal_result = result
        self.set_state("saving", "正在关闭接收器…")

    def receiver_finished(self):
        result = self.terminal_result or {"status": "failed", "error": "接收器意外退出"}
        status = result["status"]
        if status == "complete":
            self.phase.setText("保存完成")
            self.symbol.setText("✓")
            self.instruction.setText("原始 EEG、范式事件和 recording.npz 已保存")
            self.set_state("done", f"已保存 {result['samples']:,} 个采样点 · {self.last_directory}")
        elif status == "cancelled":
            self.phase.setText("记录未完成")
            self.symbol.setText("·")
            self.instruction.setText("已保留收到的数据，可在记录目录中查看")
            self.set_state("done", f"已停止并保留数据 · {self.last_directory}")
        elif status == "disconnected":
            self.set_state("idle", "已断开接收；OpenBCI GUI 的设备连接不受影响。")
        else:
            self.phase.setText("采集停止")
            self.symbol.setText("·")
            self.instruction.setText("请检查连接；已收到的数据保留在记录目录")
            self.set_state("error", result.get("error") or "接收失败")
        if self.close_pending:
            self.close()

    def refresh(self):
        if self.receiver:
            data = self.receiver.take_preview()
            if data is not None:
                self.waveform.data = data
                self.waveform.update()
            if self.state == "recording":
                duration = int(pylsl.local_clock() - self.began)
                self.status.setText(f"正在采集 · {duration // 60:02d}:{duration % 60:02d} · {self.receiver.samples:,} 个采样点")

    def closeEvent(self, event):
        self.close_pending = True
        if self.receiver and self.receiver.isRunning():
            if self.state in {"starting", "recording"}:
                self.stop()
            elif self.state != "saving":
                self.receiver.stop_event.set()
            event.ignore()
            return
        if self.scanner and self.scanner.isRunning():
            event.ignore()
            QTimer.singleShot(100, self.close)
            return
        self.timer.stop()
        self.preview_timer.stop()
        self.sound.stop()
        self.audio_dir.cleanup()
        event.accept()


def launch():
    app = QApplication.instance() or QApplication([sys.argv[0]])
    app.setApplicationName("EEG 采集")
    window = CollectorWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(launch())
