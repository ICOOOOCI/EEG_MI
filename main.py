"""单文件运动想象 BCI 主程序（Python 3.10+）。

直接启动界面：  python main.py
查看命令帮助：  python main.py --help
无界面演示：    python main.py demo --output <新目录>
导入真实数据：  python main.py import-gdf A01T.gdf --output A01T.npz
训练模型：      python main.py train A01T.npz --model <模型目录>
回放与反馈：    python main.py replay A01E.npz --model <模型目录> --output <新目录> --ui
设备接入：      python main.py live --help
查看设备流：    python main.py inspect-lsl
生成设备配置：  python main.py configure-lsl --help
个人校准采集：  python main.py record --help
导入个人采集：  python main.py import-session --help
真实在线评分：  python main.py evaluate-live --help

依赖安装：      python -m pip install numpy scipy scikit-learn mne joblib
LSL 可选依赖：  python -m pip install pylsl
也可通过同目录 pyproject.toml 安装：python -m pip install -e .

无参数运行使用合成数据，不代表真人脑电准确率。
默认演示结果存入用户数据目录 MI_BCI/runs；显式 --output 可选择保存位置。
所有项目代码都在本文件中，不依赖原 mi_bci 文件夹。
"""

from dataclasses import asdict, dataclass, fields
from math import isfinite
from numbers import Integral, Real
from dataclasses import dataclass
from typing import Protocol
import numpy as np
from typing import Iterator, Protocol
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json
import re
from scipy.signal import butter, sosfilt
from time import perf_counter
import threading
import time
from dataclasses import asdict, is_dataclass
import queue
from dataclasses import dataclass, field
from hashlib import sha256
from importlib.metadata import version as package_version
import platform
import warnings
import joblib
from scipy.signal import butter
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import mne
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
import argparse
from dataclasses import asdict, replace
import sys
import os
from uuid import uuid4

__version__ = "0.1.0"


# ========================================================================
# 01. 参数配置
# ========================================================================

EEG_CHANNELS = ("Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz")


@dataclass(frozen=True)
class Config:
    sfreq: float = 250.0
    channels: tuple[str, ...] = EEG_CHANNELS
    unit: str = "V"
    reference: str = "left_mastoid"
    low_hz: float = 8.0
    high_hz: float = 30.0
    filter_order: int = 4
    warmup_seconds: float = 1.0
    window_seconds: float = 2.0
    step_seconds: float = 0.2
    cue_offset_seconds: float = 0.5
    imagery_seconds: float = 4.0
    buffer_seconds: float = 10.0
    chunk_samples: int = 25
    n_components: int = 4
    threshold: float = 0.70
    confirmations: int = 3
    max_abs_volts: float = 500e-6
    flat_std_volts: float = 1e-9
    max_age_seconds: float = 0.5
    timestamp_tolerance_seconds: float = 0.002
    lsl_channel_labels: tuple[str, ...] | None = None
    lsl_input_units: str | None = None
    lsl_marker_grace_seconds: float = 0.10

    def __post_init__(self):
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
        if not 0 <= self.lsl_marker_grace_seconds < self.max_age_seconds:
            raise ValueError("LSL marker grace must be nonnegative and below the maximum data age")
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
        if "channels" in values:
            values["channels"] = tuple(values["channels"])
        return cls(**values)


# ========================================================================
# 02. 数据结构与模型接口
# ========================================================================

@dataclass(frozen=True)
class StreamInfo:
    sfreq: float
    channels: tuple[str, ...]
    unit: str = "V"
    reference: str = "left_mastoid"


@dataclass(frozen=True)
class EEGChunk:
    data: np.ndarray  # channels x samples, volts
    timestamps: np.ndarray
    start_sample: int
    info: StreamInfo
    segment_id: int = 0
    received_at: float | None = None  # same clock as timestamps for a live source


@dataclass(frozen=True)
class Marker:
    kind: str  # RUN_START, TRIAL_START, MI_START, MI_END, DISCONNECT
    sample: int
    trial_id: str | None = None
    timestamp: float | None = None
    segment_id: int = 0


@dataclass(frozen=True)
class Prediction:
    trial_id: str
    start_sample: int
    stop_sample: int  # exclusive
    probabilities: tuple[float, float] | None
    timestamp: float  # acquisition time when full window becomes available
    model_version: str
    quality: str = "OK"
    processing_seconds: float = 0.0


@dataclass(frozen=True)
class Command:
    trial_id: str | None
    direction: str  # LEFT, RIGHT, NO_COMMAND
    sample: int
    timestamp: float
    reason: str
    processing_seconds: float = 0.0


class WindowModel(Protocol):
    def predict_proba(self, windows: np.ndarray) -> np.ndarray: ...


# ========================================================================
# 03. 采集接口
# ========================================================================

class Source(Protocol):
    info: StreamInfo

    def __iter__(self) -> Iterator[EEGChunk | Marker]: ...

    def close(self) -> None: ...


# ========================================================================
# 04. Dataset 2a 导入与标签对齐
# ========================================================================

EVENT_ID = {str(code): code for code in (768, 769, 770, 771, 772, 783, 1023, 32766)}
CUE_CODES = (769, 770, 771, 772, 783)


@dataclass(frozen=True)
class Trial:
    subject_id: str
    session_id: str
    run_id: int
    trial_id: str
    cue_sample: int
    original_class: int | None
    artifact_flag: bool
    start_sample: int | None = None


@dataclass
class Recording:
    data: np.ndarray
    timestamps: np.ndarray
    info: StreamInfo
    trials: list[Trial]
    run_starts: list[int]
    subject_id: str
    session_id: str
    eog: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
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
        if self.eog is not None:
            self.eog = np.asarray(self.eog, dtype=np.float64)
            if self.eog.shape != (3, self.data.shape[1]):
                raise ValueError("EOG must have three channels and the EEG sample count")
        if len({t.trial_id for t in self.trials}) != len(self.trials):
            raise ValueError("Trial identifiers must be unique")
        if any(t.original_class not in (None, 1, 2, 3, 4) for t in self.trials):
            raise ValueError("Original classes must be 1..4 or unknown")
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


def _event_pairs(events) -> np.ndarray:
    """Accept Nx2 (sample, original_code) or MNE Nx3 (sample, previous, code)."""
    array = np.asarray(events)
    if array.ndim != 2 or array.shape[1] not in (2, 3):
        raise ValueError("Events must be Nx2 sample/code or MNE Nx3")
    try:
        numeric = array.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Events must contain integer samples and original numeric codes") from exc
    if not np.all(np.isfinite(numeric)) or np.any(numeric != np.floor(numeric)):
        raise ValueError("Events must contain finite integer values")
    pairs = numeric[:, (0, array.shape[1] - 1)].astype(np.int64)
    if np.any(pairs[:, 0] < 0):
        raise ValueError("Event sample positions must be nonnegative")
    return pairs[np.argsort(pairs[:, 0], kind="stable")]


def _label_vector(labels) -> np.ndarray:
    vector = np.asarray(labels)
    if vector.ndim not in (1, 2) or (vector.ndim == 2 and 1 not in vector.shape):
        raise ValueError("classlabel must be a row or column vector")
    try:
        values = vector.reshape(-1).astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("classlabel must contain numeric classes 1..4") from exc
    if not np.all(np.isfinite(values)) or np.any(~np.isin(values, (1, 2, 3, 4))):
        raise ValueError("classlabel must contain integer classes 1..4")
    return values.astype(np.int64)


def index_trials(events, subject_id: str, session_id: str, labels=None, strict: bool = True) -> list[Trial]:
    """Index every cue chronologically, attaching truth BEFORE any filtering.

    ``events`` is Nx2 (sample, original_code) or MNE Nx3. Calibration runs without
    768/MI cues do not count as task runs. Run IDs are 1-based task run indices.
    Unknown 783 cues are allowed in E without labels; never guessed as rest.
    ``strict=False`` relaxes only dataset-size/run-count/class-balance checks,
    permitting small fixtures, while preserving alignment and structure checks.
    """
    session_id = session_id.upper()
    if session_id not in ("T", "E"):
        raise ValueError("Session must be T or E")
    pairs = _event_pairs(events)
    starts = pairs[pairs[:, 1] == 768, 0]
    cues = pairs[np.isin(pairs[:, 1], CUE_CODES)]
    run_starts = pairs[pairs[:, 1] == 32766, 0]
    artifacts = pairs[pairs[:, 1] == 1023, 0]
    if len(cues) == 0 or len(starts) != len(cues):
        raise ValueError("Each 768 trial start must have exactly one cue")
    if len(np.unique(starts)) != len(starts) or len(np.unique(cues[:, 0])) != len(cues):
        raise ValueError("Duplicate trial start or cue sample")
    if len(np.unique(run_starts)) != len(run_starts):
        raise ValueError("Duplicate run start sample")
    if len(run_starts) == 0:
        if strict:
            raise ValueError("No run boundary events (32766)")
        run_starts = np.array([0], dtype=np.int64)
    if starts[0] < run_starts[0]:
        raise ValueError("Trial occurs before first run boundary")
    aligned_labels = None if labels is None else _label_vector(labels)
    if aligned_labels is not None and len(aligned_labels) != len(cues):
        raise ValueError(f"Label count {len(aligned_labels)} != cue count {len(cues)}; labels must include every trial")
    assigned_runs = np.searchsorted(run_starts, starts, side="right") - 1
    task_runs = sorted(set(assigned_runs.tolist()))
    run_ids = {raw_run: index + 1 for index, raw_run in enumerate(task_runs)}
    trials = []
    for index, (start, (cue, code)) in enumerate(zip(starts, cues)):
        next_start = int(starts[index + 1]) if index + 1 < len(starts) else np.inf
        raw_run = int(assigned_runs[index])
        next_run = int(run_starts[raw_run + 1]) if raw_run + 1 < len(run_starts) else np.inf
        stop = min(next_start, next_run)
        if not start <= cue < stop:
            raise ValueError(f"Trial {index + 1} has missing, duplicate, or out-of-run cue")
        original_class = int(code - 768) if code != 783 else None
        if aligned_labels is not None:
            truth = int(aligned_labels[index])
            if original_class is not None and truth != original_class:
                raise ValueError(f"Label/cue mismatch at trial {index + 1}, including artifact-marked trials")
            original_class = truth
        if session_id == "T" and original_class is None:
            raise ValueError("Training session contains an unknown class")
        trials.append(Trial(
            subject_id=subject_id, session_id=session_id, run_id=run_ids[raw_run],
            trial_id=f"{subject_id}{session_id}-{index + 1:03d}", cue_sample=int(cue),
            original_class=original_class,
            artifact_flag=bool(np.any((artifacts >= start) & (artifacts < stop))),
            start_sample=int(start),
        ))
    if strict:
        if len(trials) != 288:
            raise ValueError(f"Dataset 2a requires 288 trials; found {len(trials)}")
        if len(task_runs) != 6:
            raise ValueError(f"Dataset 2a requires 6 task runs; found {len(task_runs)}")
        for run_id in range(1, 7):
            members = [t for t in trials if t.run_id == run_id]
            if len(members) != 48:
                raise ValueError(f"Run {run_id} requires 48 trials; found {len(members)}")
            known = [t.original_class for t in members if t.original_class is not None]
            counts = {cls: known.count(cls) for cls in (1, 2, 3, 4)}
            if any(count > 12 for count in counts.values()) or (len(known) == 48 and any(count != 12 for count in counts.values())):
                raise ValueError(f"Run {run_id} requires 12 trials per class; found {counts}")
    return trials


def load_labels(path) -> np.ndarray:
    """Read official MATLAB ``classlabel`` without truncating or filtering it."""
    from scipy.io import loadmat
    contents = loadmat(Path(path))
    if "classlabel" not in contents:
        raise ValueError("MAT label file has no classlabel variable")
    return _label_vector(contents["classlabel"])


def _normalize_gdf_missing(data: np.ndarray, extras: dict) -> tuple[np.ndarray, str]:
    """Keep missing/saturated lower-bound samples invalid before filtering.

    MNE calibrates GDF as ``(digital * cal + offsets) * units``. Its modern
    reader removes digital_min after constructing cal; in that case the
    signed storage dtype's minimum is used. Negative digital_max is also
    checked because BioSig reserves negative full-scale values for overflow.
    Any channel at/below that boundary invalidates the entire sample. This
    intentionally treats negative saturation as unusable data too. Unknown
    calibration layouts raise, rather than guessing zeros or signal units.
    """
    result = np.array(data, dtype=np.float64, copy=True)
    count = result.shape[0]
    try:
        cal = np.broadcast_to(np.asarray(extras["cal"], dtype=float), (count,))
        offset = np.broadcast_to(np.asarray(extras["offsets"], dtype=float), (count,))
        unit = np.broadcast_to(np.asarray(extras["units"], dtype=float), (count,))
        if "digital_min" in extras:
            digital_min = np.broadcast_to(np.asarray(extras["digital_min"], dtype=float), (count,))
        else:
            dtype = np.dtype(extras["dtype_np"])
            if dtype.kind == "i":
                digital_min = np.full(count, np.iinfo(dtype).min, dtype=float)
            elif "digital_max" in extras:
                # GDF also permits float storage of digital full-scale values.
                digital_min = -np.abs(np.asarray(extras["digital_max"], dtype=float))
            else:
                raise ValueError("GDF storage has no recoverable negative digital full scale")
        if "digital_max" in extras:
            digital_min = np.maximum(digital_min, -np.abs(np.asarray(extras["digital_max"], dtype=float)))
        if np.any(cal <= 0) or np.any(unit <= 0) or np.any(digital_min >= 0):
            raise ValueError("Invalid GDF calibration")
        boundary = (digital_min * cal + offset) * unit
        tolerance = np.abs(cal * unit) * 0.25
        if not np.all(np.isfinite(boundary)):
            raise ValueError("Nonfinite GDF calibration")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Cannot identify GDF missing-value sentinel from MNE calibration metadata; import aborted") from exc
    invalid = np.any(~np.isfinite(result) | (result <= (boundary + tolerance)[:, None]), axis=0)
    result[:, invalid] = np.nan
    return result, "calibrated negative digital full-scale; lower-bound saturation also invalid"


def load_gdf(path, labels_path=None, strict: bool = True) -> Recording:
    """Load a standard AxxT/AxxE GDF, retaining all 4 classes and time samples.

    The official channel order is 22 EEG then 3 EOG. Numeric annotation strings
    are mapped explicitly rather than relying on MNE's generated event IDs.
    GDF support is exercised with fixtures; real files are not bundled.
    """
    import mne
    path = Path(path)
    match = re.fullmatch(r"(A\d{2})([TE])", path.stem, flags=re.IGNORECASE)
    if not match:
        raise ValueError("Expected Dataset 2a filename AxxT.gdf or AxxE.gdf")
    subject_id, session_id = (value.upper() for value in match.groups())
    raw = mne.io.read_raw_gdf(path, eog=[22, 23, 24], stim_channel=None, preload=True, verbose="ERROR")
    if len(raw.ch_names) != 25 or not np.isclose(raw.info["sfreq"], 250.0):
        raise ValueError("Dataset 2a requires exactly 25 channels at 250 Hz")
    descriptions = set(raw.annotations.description)
    mapped = {name: value for name, value in EVENT_ID.items() if name in descriptions}
    if not any(str(code) in mapped for code in CUE_CODES):
        raise ValueError("No recognized numeric GDF cue annotations; refusing implicit event mapping")
    events, _ = mne.events_from_annotations(raw, event_id=mapped, verbose="ERROR")
    events[:, 0] -= raw.first_samp
    labels = None if labels_path is None else load_labels(labels_path)
    trials = index_trials(events, subject_id, session_id, labels=labels, strict=strict)
    extras = raw._raw_extras[0]
    if "n_samps" in extras and len(set(np.asarray(extras["n_samps"]).tolist())) != 1:
        raise ValueError("Mixed channel sampling rates are not accepted for Dataset 2a")
    data, missing_method = _normalize_gdf_missing(raw.get_data(), extras)
    return Recording(
        data=data[:22], timestamps=np.arange(data.shape[1]) / 250.0,
        info=StreamInfo(250.0, EEG_CHANNELS), trials=trials,
        run_starts=sorted(set(int(x) for x in events[events[:, 2] == 32766, 0])),
        subject_id=subject_id, session_id=session_id, eog=data[22:25],
        metadata={"source": "BCI Competition IV Dataset 2a", "synthetic": False,
                  "source_file": path.name, "original_channel_names": list(raw.ch_names),
                  "eog_channel_names": list(raw.ch_names[22:]),
                  "labels_file": None if labels_path is None else Path(labels_path).name,
                  "missing_value_policy": missing_method,
                  "missing_samples": int(np.any(~np.isfinite(data), axis=0).sum()),
                  "strict_validation": bool(strict)},
    )


def save_recording(path, recording: Recording) -> None:
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
    if recording.eog is not None:
        arrays["eog"] = recording.eog
    with path.open("wb") as file:
        np.savez_compressed(file, **arrays)


def load_recording(path) -> Recording:
    path = Path(path)
    if path.suffix.lower() == ".gdf":
        return load_gdf(path)
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
            session_id=payload["session_id"], eog=arrays["eog"] if "eog" in arrays else None,
            metadata=payload.get("metadata", {}),
        )


# ========================================================================
# 05. 合成演示数据
# ========================================================================

def make_synthetic_recording(session_id="T", subject_id="SYN01", runs=3, trials_per_run=8, seed=42) -> Recording:
    """Create balanced 4-class runs with fixed spatial patterns and session noise.

    Each trial is 8 seconds: cue at +2 seconds, imagery for 4 seconds. A 100
    sample NaN gap separates runs. The known artificial class-dependent power
    makes this a pipeline demonstration, not an estimate of human MI accuracy.
    """
    session_id = str(session_id).upper()
    if session_id not in ("T", "E"):
        raise ValueError("Session must be T or E")
    if runs < 1 or int(runs) != runs or trials_per_run < 4 or int(trials_per_run) != trials_per_run or trials_per_run % 4:
        raise ValueError("Use positive integer runs and trials_per_run divisible by 4")
    runs, trials_per_run = int(runs), int(trials_per_run)
    sfreq, trial_samples, gap_samples, lead_samples = 250, 2000, 100, 500
    run_samples = lead_samples + trials_per_run * trial_samples
    sample_count = runs * run_samples + (runs - 1) * gap_samples
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0 if session_id == "T" else 1]))
    times = np.arange(sample_count) / sfreq
    # Independent background plus a shared low-frequency drift and line noise.
    data = rng.normal(0, 4e-6, (len(EEG_CHANNELS), sample_count))
    data += rng.normal(0, 1e-6, (1, sample_count))
    data += 3e-6 * np.sin(2 * np.pi * 1.3 * times)[None, :]
    data += 0.4e-6 * np.sin(2 * np.pi * 50 * times)[None, :]
    eog = rng.normal(0, 7e-6, (3, sample_count))
    # Shared patterns are tied to electrode positions, not random session seed.
    left = np.array([0.05, .4, .25, .1, 0, 0, .65, 1, .6, .1, 0, 0, 0, .5, .3, .1, 0, 0, .05, 0, 0, 0])
    right = np.array([0.05, 0, 0, .1, .25, .4, 0, 0, 0, .1, .6, 1, .65, 0, 0, .1, .3, .5, 0, 0, .05, 0])
    middle = np.array([.3, .1, .3, .6, .3, .1, .05, .15, .5, 1, .5, .15, .05, .1, .3, .6, .3, .1, .2, .5, .2, .1])
    posterior = np.array([0, 0, 0, 0, 0, 0, .1, .1, .1, .2, .1, .1, .1, .3, .4, .5, .4, .3, .6, 1, .6, .8])
    trials, run_starts = [], []
    for run in range(runs):
        run_start = run * (run_samples + gap_samples)
        run_starts.append(run_start)
        if run:
            data[:, run_start - gap_samples:run_start] = np.nan
            eog[:, run_start - gap_samples:run_start] = np.nan
        classes = rng.permutation(np.tile(np.arange(1, 5), trials_per_run // 4))
        for ordinal, cls in enumerate(classes):
            start = run_start + lead_samples + ordinal * trial_samples
            cue = start + 2 * sfreq
            mi = slice(cue, cue + 4 * sfreq)
            phase = rng.uniform(0, 2 * np.pi, 4)
            t = np.arange(4 * sfreq) / sfreq
            envelope = np.minimum(1, t / .15) * np.minimum(1, (4 - t) / .15)
            # Contralateral mu suppression: left imagery lowers right-side mu.
            amplitudes = {1: (13, 3, 3, 2), 2: (3, 13, 3, 2), 3: (6, 6, 13, 2), 4: (6, 6, 3, 13)}[int(cls)]
            for pattern, amplitude, frequency, offset in zip((left, right, middle, posterior), amplitudes, (11, 12, 18, 23), phase):
                oscillation = np.sin(2 * np.pi * (frequency + rng.normal(0, .15)) * t + offset)
                data[:, mi] += pattern[:, None] * (amplitude * 1e-6 * rng.uniform(.85, 1.15)) * oscillation * envelope
            trials.append(Trial(str(subject_id), session_id, run + 1,
                                f"{subject_id}{session_id}-{len(trials) + 1:03d}", cue,
                                int(cls), False, start))
    return Recording(data, times, StreamInfo(float(sfreq), EEG_CHANNELS), trials, run_starts,
                     str(subject_id), session_id, eog=eog,
                     metadata={"source": "synthetic engineering demonstration", "synthetic": True,
                               "seed": int(seed), "runs": runs, "trials_per_run": trials_per_run,
                               "description": "Artificial spatial oscillatory classes; not recorded EEG or real participant results"})


# ========================================================================
# 06. 因果预处理
# ========================================================================

@dataclass(frozen=True)
class FilteredChunk:
    data: np.ndarray
    timestamps: np.ndarray
    sample_indices: np.ndarray
    valid: np.ndarray
    breaks: np.ndarray
    reasons: np.ndarray


class CausalPreprocessor:
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
            # An acquisition timestamp in the future indicates a clock mismatch.
            stale |= (timestamps - chunk.received_at) > cfg.timestamp_tolerance_seconds
            reasons[stale & good_signal] = "STALE_DATA"
        starts = np.unique(np.r_[0, np.flatnonzero(breaks), np.flatnonzero(~good_signal), n])
        for left, right in zip(starts[:-1], starts[1:]):
            if breaks[left]:
                self._reset_filter()
            if not good_signal[left]:
                # Each invalid sample is a boundary, so this interval is one sample.
                self._reset_filter()
                continue
            filtered[:, left:right], self._zi = sosfilt(
                self.sos, data[:, left:right], axis=-1, zi=self._zi)
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
# 07. 环形缓冲
# ========================================================================

@dataclass(frozen=True)
class BufferedWindow:
    data: np.ndarray
    raw_data: np.ndarray
    timestamps: np.ndarray
    valid: np.ndarray
    epochs: np.ndarray
    reasons: np.ndarray


class RingBuffer:
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
        count = stop - start
        if count <= 0 or count > self.size:
            return None
        latest_position = (self._position - 1) % self.capacity
        offset = int(self.indices[latest_position]) - (stop - 1)
        if offset < 0 or offset + count > self.size:
            return None
        positions = (np.arange(self._position - offset - count,
                               self._position - offset) % self.capacity)
        if not np.array_equal(self.indices[positions], np.arange(start, stop)):
            return None
        return BufferedWindow(self.data[:, positions].copy(), self.raw_data[:, positions].copy(), self.timestamps[positions].copy(),
                              self.valid[positions].copy(), self.epochs[positions].copy(),
                              self.reasons[positions].copy())


# ========================================================================
# 08. 指令判决
# ========================================================================

class DecisionPolicy:
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
        """Invalidate accumulated evidence without unlocking a committed trial."""
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


# ========================================================================
# 09. 流式解码与状态机
# ========================================================================

NEUTRAL_MARKERS = frozenset({"RUN_START", "TRIAL_START", "MI_START", "MI_END", "DISCONNECT", "SESSION_END"})


def iter_window_starts(config: Config, cue_sample: int):
    first = cue_sample + config.samples(config.cue_offset_seconds)
    last = cue_sample + config.samples(config.imagery_seconds) - config.window_samples
    return range(first, last + 1, config.step_samples)


class Runtime:
    def __init__(self, config: Config, model, sos=None, model_version="unversioned", window_observer=None):
        self.config, self.model, self.model_version = config, model, model_version
        self.preprocessor = CausalPreprocessor(config, sos)
        self.buffer = RingBuffer(len(config.channels), config.samples(config.buffer_seconds))
        self.decision = DecisionPolicy(config)
        self.window_observer = window_observer
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
        # Consume in acquisition order, not whole-chunk order: even a chunk larger
        # than the ring cannot overwrite a window before its prediction is made.
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
        if window is None:
            quality = "INSUFFICIENT_DATA"
        elif np.any(window.reasons == "STALE_DATA"):
            quality = "STALE_DATA"
        elif not window.valid.all() or np.any(window.epochs != window.epochs[0]):
            quality = "BAD_SIGNAL"
        elif (not np.isfinite(window.data).all()
              or np.any(np.std(window.data, axis=1) <= self.config.flat_std_volts)
              or np.any(np.std(window.raw_data, axis=1) <= self.config.flat_std_volts)):
            quality = "BAD_SIGNAL"
        elif (np.any(np.diff(window.timestamps) <= 0)
              or np.any(np.abs(np.diff(window.timestamps) - 1.0 / self.config.sfreq)
                        > self.config.timestamp_tolerance_seconds)):
            quality = "BAD_SIGNAL"
        elif self._cue_timestamp is not None:
            # A source may number only delivered samples after a packet loss.
            # Keep the cue's real acquisition-time anchor so later windows cannot
            # silently slide past the intended imagery phase after such a loss.
            tolerance = max(1.0 / self.config.sfreq, self.config.timestamp_tolerance_seconds) + 1e-9
            expected_start = self._cue_timestamp + (start - self._cue) / self.config.sfreq
            expected_last = self._cue_timestamp + (stop - 1 - self._cue) / self.config.sfreq
            if (abs(window.timestamps[0] - expected_start) > tolerance
                    or abs(window.timestamps[-1] - expected_last) > tolerance):
                quality = "BAD_SIGNAL"
        if quality == "OK":
            timestamp = float(window.timestamps[-1]) + 1.0 / self.config.sfreq
            try:
                p = np.asarray(self.model.predict_proba(window.data[np.newaxis]), dtype=float)
                if p.shape != (1, 2) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1) or not np.isclose(p.sum(), 1):
                    raise ValueError("Model probabilities must be a normalized 1 x 2 array")
                probabilities = tuple(float(x) for x in p[0])
            except Exception:
                quality = "MODEL_ERROR"
        prediction = Prediction(self.trial_id, start, stop, probabilities, timestamp,
                                self.model_version, quality, perf_counter() - began)
        if quality != "OK":
            self.stats["invalid_windows"] += 1
        if quality == "MODEL_ERROR":
            self.stats["model_errors"] += 1
        if quality == "OK" and self.window_observer is not None:
            self.window_observer(window.data.copy(), prediction)
        return [prediction, self.decision.decide(prediction)]

    def _flush(self, timestamp=None):
        """Close pending windows as invalid; never infer retrospectively at phase end."""
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
            self.preprocessor.reset()
            self.buffer.clear()
            self._epoch += 1
            self.decision.clear_streak()
            return [Command(self.trial_id, "NO_COMMAND", marker.sample,
                            marker.timestamp if marker.timestamp is not None else self._safe_timestamp(marker.sample),
                            "STALE_DATA")]
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
        # A mismatched old end marker must not terminate the current trial.
        if marker.trial_id is not None and marker.trial_id != self.trial_id:
            return []
        outputs = self._flush(marker.timestamp)
        self._active = False
        self.decision.end()
        return outputs

    def finish(self):
        outputs = self._flush()
        self._active = False
        self.decision.end()
        return outputs


# ========================================================================
# 10. 连续回放
# ========================================================================

class ReplaySource:
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
        # Never inspect original_class or artifact_flag when producing packets.
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


# ========================================================================
# 11. LSL 设备采集
# ========================================================================

UNIT_SCALE = {"V": 1.0, "mV": 1e-3, "uV": 1e-6, "µV": 1e-6, "μV": 1e-6}
NEUTRAL_KINDS = set(NEUTRAL_MARKERS)


def parse_neutral_marker(payload):
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
    is_live = True

    def __init__(self, config, eeg_name, marker_name, reference, units=None, resolve_timeout=5.0):
        try:
            import pylsl
        except ImportError as exc:
            raise RuntimeError("Install the live extra: pip install -e .[live]") from exc
        if reference != config.reference:
            raise ValueError("Device reference must match the frozen model; recalibrate after changing reference")
        if units and config.lsl_input_units and UNIT_SCALE.get(units) != UNIT_SCALE[config.lsl_input_units]:
            raise ValueError("Input units differ from the frozen device configuration")
        units = units or config.lsl_input_units
        self.config = config
        self.clock = pylsl.local_clock
        self.info = StreamInfo(config.sfreq, config.channels, "V", reference)
        self._stop = threading.Event()
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
                # This mapping is a user-declared wiring map, never inferred from channel count.
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
            # Synchronization is performed by proc_clocksync on both inlets.
            self.eeg.open_stream(timeout=resolve_timeout)
            self.markers.open_stream(timeout=resolve_timeout)
        except BaseException:
            self.close()
            raise

    def close(self):
        self._stop.set()
        # A running iterator owns inlet teardown, avoiding races with pull_chunk.
        # Constructor failure or close-before-iteration must release them here.
        with self._state_lock:
            if not self._iterator_active:
                self._close_inlets()

    def _close_inlets(self):
        for name in ("eeg", "markers"):
            inlet = getattr(self, name, None)
            if inlet is not None:
                try:
                    inlet.close_stream()
                finally:
                    setattr(self, name, None)

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
        try:
            while not self._stop.is_set():
                samples, stamps = self.eeg.pull_chunk(timeout=0.1, max_samples=self.config.chunk_samples)
                # Consume all available markers before advancing the EEG timeline.
                while True:
                    payload, stamp = self.markers.pull_sample(timeout=0.0)
                    if payload is None:
                        break
                    if not np.isfinite(stamp):
                        raise ValueError("Marker timestamp must be finite")
                    pending.append((float(stamp), parse_neutral_marker(payload[0])))
                if stamps:
                    last_data = self.clock()
                    times = np.asarray(stamps, dtype=float)
                    values = np.asarray(samples, dtype=float).T[self.order] * self.scales
                    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
                        raise RuntimeError("EEG timestamps must be finite and strictly increasing")
                    buffered.append((values, times, last_data))
                # Hold EEG briefly so independently transported phase markers can catch up.
                # Timestamps stay untouched; data age still includes this bounded holdback.
                pending.sort(key=lambda pair: pair[0])
                while buffered and self.clock() - buffered[0][2] >= self.config.lsl_marker_grace_seconds:
                    values, times, _ = buffered.pop(0)
                    position = 0
                    while pending and pending[0][0] <= times[-1] + 1 / self.config.sfreq:
                        stamp, event = pending.pop(0)
                        if previous_end is not None and stamp <= previous_end:
                            raise RuntimeError("Late marker arrived after its EEG was decoded; stopping to preserve causal alignment")
                        boundary = int(np.searchsorted(times, stamp, side="left"))
                        if boundary > position:
                            yield EEGChunk(values[:, position:boundary], times[position:boundary], index + position, self.info, segment, self.clock())
                        if event["kind"] in {"RUN_START", "DISCONNECT"}:
                            segment += 1
                        yield Marker(event["kind"], index + boundary, event.get("trial_id"), stamp, segment)
                        if event["kind"] == "SESSION_END":
                            return
                        if event["kind"] == "DISCONNECT":
                            raise RuntimeError("Paradigm aborted; this session is incomplete")
                        position = boundary
                    if position < len(times):
                        yield EEGChunk(values[:, position:], times[position:], index + position, self.info, segment, self.clock())
                    index += len(times)
                    previous_end = float(times[-1])
                if not stamps and self.clock() - last_data > 1.0:
                    yield Marker("DISCONNECT", index, timestamp=self.clock(), segment_id=segment)
                    raise RuntimeError("No EEG for one second; restart source after reconnecting")
        finally:
            with self._state_lock:
                self._iterator_active = False
                self._close_inlets()


# ========================================================================
# 12. 异步记录
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
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_value, allow_nan=False), encoding="utf-8")


class AsyncRecorder:
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
        self.closed = False
        self.thread = threading.Thread(target=self._worker, name="mi-recorder", daemon=True)
        write_json(self.directory / "manifest.json", {"format": "mi-bci-session-v1", "status": "recording", "stream": info, "raw_layout": "sample-major little-endian float64, raw_eeg.f64 and timestamps.f64; chunk offsets in chunks.jsonl"})
        self.thread.start()

    def record(self, item):
        if self.closed:
            raise RuntimeError("Recorder already closed")
        if self.error:
            raise RuntimeError("Recorder write failed") from self.error
        if isinstance(item, EEGChunk):
            # Sources are free to reuse their acquisition buffers.
            item = EEGChunk(item.data.copy(), item.timestamps.copy(), item.start_sample, item.info, item.segment_id, item.received_at)
        try:
            self.queue.put_nowait(item)
        except queue.Full as exc:
            raise RuntimeError("Recorder backlog full; session stopped to prevent silent loss") from exc
        self.max_backlog = max(self.max_backlog, self.queue.qsize())

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
                            chunks.write(json.dumps({"offset_samples": self.raw_samples, "count": n, "start_sample": item.start_sample, "segment_id": item.segment_id, "received_at": item.received_at}) + "\n")
                            self.raw_samples += n
                        else:
                            events.write(json.dumps(item, ensure_ascii=False, default=json_value, allow_nan=False) + "\n")
                    finally:
                        self.queue.task_done()
        except BaseException as exc:
            self.error = exc

    def close(self, status="complete", error=None):
        if self.closed:
            return
        self.closed = True
        deadline = time.monotonic() + 10
        while self.thread.is_alive():
            if time.monotonic() >= deadline:
                self.error = RuntimeError("Recorder did not accept shutdown within 10 seconds")
                break
            try:
                self.queue.put(None, timeout=0.1)
                break
            except queue.Full:
                continue
        self.thread.join(timeout=max(0, deadline - time.monotonic()))
        if self.thread.is_alive():
            self.error = RuntimeError("Recorder did not finish within 10 seconds")
        write_json(self.directory / "manifest.json", {
            "format": "mi-bci-session-v1", "status": "failed" if self.error or error else status,
            "error": str(self.error or error) if self.error or error else None,
            "stream": self.info, "raw_samples": self.raw_samples, "max_recorder_backlog": self.max_backlog,
            "raw_layout": "sample-major little-endian float64, raw_eeg.f64 and timestamps.f64; chunk offsets in chunks.jsonl",
        })
        if self.error:
            raise RuntimeError("Recorder failed") from self.error


# ========================================================================
# 12a. 真人采集记录与独立标签合并
# ========================================================================

def load_personal_session(directory, labels_path, *, subject_id=None, session_id=None) -> Recording:
    """Import a completed personal capture without changing its EEG timeline.

    Cue positions come exclusively from the recorder's neutral Marker.sample.
    The paradigm's source clock can differ from the receiving LSL clock, so its
    timestamps are never used to locate EEG samples. Truth joins by trial ID;
    the complete neutral event sequence must also agree with the truth log.
    """
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
    if not isinstance(config_values, dict) or set(config_values) != {f.name for f in fields(Config)}:
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
    for event in read_rows(directory / "events.jsonl"):
        event_type = event.get("type")
        if event_type == "Marker":
            if event.get("kind") not in allowed_phases:
                raise ValueError("Disconnected, aborted, or unrecognized marker in personal capture")
            if any(key in event for key in ("original_class", "class_label", "label")):
                raise ValueError("Recorded neutral markers must not contain class labels")
            markers.append(event)
        elif event_type == "SessionEnd":
            if event.get("error") or event.get("cancelled") or event.get("pending_acquisition_packets", 0) or event.get("cleanup_errors"):
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
                     metadata={"source": "personal_lsl", "synthetic": False,
                               "capture_directory": directory.name, "labels_file": labels_path.name,
                               "capture_context": context, "config": config.to_dict(),
                               "label_alignment": "Exact trial IDs and phase sequence; cue samples from received neutral markers",
                               "artifact_annotation": "Not manually reviewed; runtime signal-quality checks remain active"})


# ========================================================================
# 13. 冻结模型包
# ========================================================================

CLASS_MAPPING = {"0": "LEFT", "1": "RIGHT"}


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def dependency_versions():
    return {"python": platform.python_version(), **{
        name: package_version(name) for name in ("numpy", "scipy", "scikit-learn", "mne", "joblib")}}


@dataclass
class ModelBundle:
    config: Config
    model: object
    sos: np.ndarray
    version: str
    metadata: dict = field(default_factory=dict)
    report: dict = field(default_factory=dict)

    def validate(self):
        # Reconstructing invokes every configuration invariant.
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
        if not self.metadata.get("training_subject") or not str(self.metadata.get("training_session", "")).upper().endswith("T"):
            raise ValueError("Bundle must identify its training subject and T session")
        if self.metadata.get("class_mapping") != CLASS_MAPPING:
            raise ValueError("Unsupported class mapping")
        if self.metadata.get("filter_initialization") != "zero_state_per_continuous_segment":
            raise ValueError("Unsupported filter initialization policy")
        if self.metadata.get("reference") != self.config.reference or self.metadata.get("warmup_seconds") != self.config.warmup_seconds:
            raise ValueError("Training metadata differs from the frozen reference/warmup configuration")

    def save(self, path):
        """Save a directory containing manifest.json and model.joblib."""
        self.validate()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
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
        """Deserialize a TRUSTED local bundle. Never load an untrusted joblib."""
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
# 14. 独立评估
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
    classified = [r for r in rows if r["predicted_class"] is not None]
    emitted = [r for r in rows if r["command_class"] is not None]
    correct = sum(r["command_correct"] for r in emitted)
    truth = [r["target_class"] for r in classified]
    pred = [r["predicted_class"] for r in classified]
    # Balanced accuracy is undefined if an entire target class is unobserved.
    balanced = float(balanced_accuracy_score(truth, pred)) if len(set(truth)) == 2 else None
    return {
        "target_trials": len(rows), "classified_trials": len(classified),
        "missing_classifications": len(rows) - len(classified),
        "accuracy": float(accuracy_score(truth, pred)) if truth else None,
        "balanced_accuracy": balanced,
        "confusion_matrix": confusion_matrix(truth, pred, labels=[0, 1]).tolist() if truth else [[0, 0], [0, 0]],
        "confusion_labels": ["LEFT", "RIGHT"],
        "command_trials": len(emitted), "correct_command_trials": correct,
        "command_accuracy": correct / len(emitted) if emitted else None,
        "coverage": len(emitted) / len(rows) if rows else None,
        "correct_command_fraction": correct / len(rows) if rows else None,
        "cue_to_command_seconds": _distribution([r["cue_to_command_seconds"] for r in emitted if r["cue_to_command_seconds"] is not None]),
        "invalid_command_timing_trials": sum(not r["command_in_phase"] for r in emitted),
        "command_processing_seconds": _distribution([r["command_processing_seconds"] for r in emitted]),
        "invalid_windows": sum(r["invalid_windows"] for r in rows),
        "invalid_window_reasons": dict(sum((Counter(r["invalid_window_reasons"]) for r in rows), Counter())),
    }


def evaluate(trials, predictions, commands, sfreq=250.0, imagery_seconds=4.0):
    """Score all target trials, keeping missing/rejected trials in denominators.

    Classification averages the valid window probabilities within each trial.
    LEFT/RIGHT commands are scored separately; NO_COMMAND never counts as a
    directional command. Cue latency uses sample indices, avoiding replay wall
    clock speed. Software processing time is reported separately, as measured
    by the runtime; it is not a complete device/transport/display latency.
    """
    if sfreq <= 0:
        raise ValueError("sfreq must be positive")
    trials, predictions, commands = list(trials), list(predictions), list(commands)
    ids = [_value(t, "trial_id") for t in trials]
    if len(ids) != len(set(ids)):
        raise ValueError("Trial identifiers must be unique for label alignment")
    trial_ids = set(ids)
    pred_by, cmd_by = defaultdict(list), defaultdict(list)
    for p in predictions:
        pred_by[_value(p, "trial_id")].append(p)
    for command in commands:
        cmd_by[_value(command, "trial_id")].append(command)
    target_rows, non_target_rows, unlabeled_rows = [], [], []
    duplicate_commands = 0
    out_of_phase = 0
    for trial in trials:
        trial_id = _value(trial, "trial_id")
        cue_sample = int(_value(trial, "cue_sample"))
        phase_end = cue_sample + round(imagery_seconds * sfreq)
        trial_predictions = sorted(pred_by[trial_id], key=lambda p: _value(p, "stop_sample", 0))
        valid = []
        invalid = Counter()
        for p in trial_predictions:
            quality = _value(p, "quality", "OK")
            probs = _value(p, "probabilities")
            if not cue_sample <= _value(p, "start_sample") < _value(p, "stop_sample") <= phase_end:
                invalid["OUTSIDE_MI_PHASE"] += 1
                continue
            if quality != "OK" or probs is None:
                invalid[quality if quality != "OK" else "MISSING_PROBABILITIES"] += 1
                continue
            probs = np.asarray(probs, dtype=float)
            if probs.shape != (2,) or not np.isfinite(probs).all() or np.any(probs < 0) or not np.isclose(probs.sum(), 1):
                raise ValueError("Invalid probability vector in evaluation")
            valid.append(probs)
        mean = np.mean(valid, axis=0) if valid else None
        direction_commands = sorted(
            [c for c in cmd_by[trial_id] if _value(c, "direction") in ("LEFT", "RIGHT")],
            key=lambda c: _value(c, "sample", 0))
        duplicate_commands += max(0, len(direction_commands) - 1)
        for command in direction_commands:
            if not cue_sample <= _value(command, "sample") <= phase_end:
                out_of_phase += 1
        first = direction_commands[0] if direction_commands else None
        in_phase = bool(first and cue_sample <= _value(first, "sample") <= phase_end)
        original = _value(trial, "original_class")
        command_class = (0 if _value(first, "direction") == "LEFT" else 1) if first else None
        row = {
            "trial_id": trial_id, "run_id": _value(trial, "run_id"),
            "original_class": original, "target_class": original - 1 if original in (1, 2) else None,
            "artifact_flag": bool(_value(trial, "artifact_flag", False)),
            "valid_windows": len(valid), "invalid_windows": sum(invalid.values()),
            "invalid_window_reasons": dict(invalid),
            "mean_probabilities": mean.tolist() if mean is not None else None,
            "predicted_class": int(np.argmax(mean)) if mean is not None else None,
            "command_class": command_class,
            "command_in_phase": in_phase,
            "command_correct": bool(in_phase and original in (1, 2) and command_class == original - 1),
            "direction_commands": len(direction_commands),
            "cue_to_command_seconds": (_value(first, "sample") - cue_sample) / sfreq if in_phase else None,
            "command_processing_seconds": float(_value(first, "processing_seconds", 0.0)) if first else None,
            "no_command_reasons": dict(Counter(_value(c, "reason") for c in cmd_by[trial_id] if _value(c, "direction") == "NO_COMMAND")),
        }
        if original in (1, 2):
            target_rows.append(row)
        elif original in (3, 4):
            non_target_rows.append(row)
        else:
            unlabeled_rows.append(row)
    all_scores = _score(target_rows)
    return {
        "scope": "Left/right cue-synchronous trial scoring; not the original four-class competition metric.",
        "aggregation": "Mean valid window probabilities per trial; first directional command per trial.",
        "probability_note": "LDA scores are not calibrated correctness guarantees.",
        "latency_note": "Cue latency uses acquisition sample indices; processing time excludes device transport and UI display.",
        "all": all_scores,
        "artifact_clean": _score([r for r in target_rows if not r["artifact_flag"]]),
        "artifact_flagged": _score([r for r in target_rows if r["artifact_flag"]]),
        "non_target": {
            "trials": len(non_target_rows),
            "command_trials": sum(r["command_class"] is not None for r in non_target_rows),
            "direction_commands": sum(r["direction_commands"] for r in non_target_rows),
            "interpretation": "Feet/tongue outputs are reported separately; this binary model has no rest or non-control detector.",
            "per_trial": non_target_rows,
        },
        "unlabeled_trials": len(unlabeled_rows),
        "unlabeled_per_trial": unlabeled_rows,
        "stability": {
            "duplicate_direction_commands": duplicate_commands, "out_of_phase_direction_commands": out_of_phase,
            "unknown_trial_predictions": sum(_value(p, "trial_id") not in trial_ids for p in predictions),
            "unknown_trial_direction_commands": sum(_value(c, "trial_id") not in trial_ids and _value(c, "direction") in ("LEFT", "RIGHT") for c in commands),
            "prediction_processing_seconds": _distribution([float(_value(p, "processing_seconds", 0.0)) for p in predictions]),
        },
        "per_trial": target_rows,
    }


# ========================================================================
# 15. CSP 与 LDA 训练
# ========================================================================

class _WindowProbe:
    def predict_proba(self, windows):
        return np.full((len(windows), 2), 0.5)


@dataclass
class ExtractedWindows:
    windows: np.ndarray
    predictions: list
    valid_prediction_indices: list


def extract_windows(recording, config=None):
    """Use the deployed Runtime unchanged; truth is never passed to its model.

    The observation hook copies the exact causal, quality-approved input that
    Runtime supplies to predict_proba. Invalid predictions remain in the result
    so validation also replays streak resets caused by rejected windows.
    """
    config = config or Config()
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
    return Pipeline([
        ("csp", CSP(n_components=config.n_components, log=True, norm_trace=False,
                    transform_into="average_power")),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])


def _fit_model(config, x, y):
    if set(np.asarray(y).tolist()) != {0, 1}:
        raise ValueError("Every training fold needs valid LEFT and RIGHT windows")
    pipeline = make_pipeline(config)
    with mne.use_log_level("ERROR"):
        pipeline.fit(x, y)
    return pipeline


def _validation_commands(config, trials, predictions):
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


def train_recording(recording, config=None, *, thresholds=(0.6, 0.7, 0.8), confirmations=(2, 3), exclude_artifacts=True):
    """Train only a T session; select the decision policy using held-out runs.

    All overlapping windows from a trial and its entire run stay in one fold.
    Expert artifact labels affect fitting only, never the runtime quality gate.
    Validation reports include both clean and flagged trials. E sessions are
    rejected before any window extraction, fit, or parameter selection.
    """
    config = config or Config()
    if not str(recording.session_id).upper().endswith("T"):
        raise ValueError("Training requires a T session; E data must never select or fit parameters")
    if not thresholds or not confirmations:
        raise ValueError("At least one threshold and confirmation count are required")
    trials = list(recording.trials)
    if any(t.subject_id != recording.subject_id or t.session_id != recording.session_id for t in trials):
        raise ValueError("Training recording must contain exactly its declared subject and session")
    if any(t.original_class not in (1, 2, 3, 4) for t in trials):
        raise ValueError("Training trials require explicit original classes 1 through 4")
    truth = {t.trial_id: t for t in trials}
    if len(truth) != len(trials):
        raise ValueError("Trial IDs must be unique")
    run_ids = sorted({t.run_id for t in trials}, key=str)
    if len(run_ids) < 2:
        raise ValueError("At least two runs are required for leave-one-run-out validation")
    extraction = extract_windows(recording, config)
    valid_predictions = [extraction.predictions[i] for i in extraction.valid_prediction_indices]
    window_trials = [truth[p.trial_id] for p in valid_predictions]
    eligible = np.asarray([t.original_class in (1, 2) and (not exclude_artifacts or not t.artifact_flag)
                           for t in window_trials], dtype=bool)
    if not eligible.any():
        raise ValueError("No quality-approved left/right training windows")
    labels = np.asarray([t.original_class - 1 for t in window_trials])
    groups = np.asarray([t.run_id for t in window_trials])
    oof_predictions = list(extraction.predictions)
    fold_records = []
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
    for threshold in thresholds:
        for count in confirmations:
            candidate = replace(config, threshold=float(threshold), confirmations=int(count))
            commands = _validation_commands(candidate, trials, oof_predictions)
            metrics = evaluate(trials, oof_predictions, commands, sfreq=config.sfreq, imagery_seconds=config.imagery_seconds)
            candidates.append({"threshold": candidate.threshold, "confirmations": candidate.confirmations,
                               "metrics": metrics["all"], "artifact_clean": metrics["artifact_clean"],
                               "artifact_flagged": metrics["artifact_flagged"]})

    def selection_key(candidate):
        metrics = candidate["metrics"]
        return (metrics["correct_command_fraction"] or 0, metrics["command_accuracy"] or 0,
                candidate["threshold"], candidate["confirmations"])

    selected = max(candidates, key=selection_key)
    frozen = replace(config, threshold=selected["threshold"], confirmations=selected["confirmations"])
    selected_commands = _validation_commands(frozen, trials, oof_predictions)
    selected_metrics = evaluate(trials, oof_predictions, selected_commands, sfreq=config.sfreq, imagery_seconds=config.imagery_seconds)
    for fold in fold_records:
        fold_trials = [t for t in trials if t.run_id == fold["held_out_run"]]
        fold_ids = {t.trial_id for t in fold_trials}
        fold["metrics"] = evaluate(fold_trials, [p for p in oof_predictions if p.trial_id in fold_ids],
                                   [c for c in selected_commands if c.trial_id in fold_ids], sfreq=config.sfreq,
                                   imagery_seconds=config.imagery_seconds)
    final_model = _fit_model(frozen, extraction.windows[eligible], labels[eligible])
    source_hash = _recording_digest(recording)
    versions = dependency_versions()
    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_subject": recording.subject_id, "training_session": recording.session_id,
        "recording_metadata": dict(recording.metadata),
        "synthetic": bool(recording.metadata.get("synthetic", False)),
        "recording_sha256": source_hash, "dependencies": versions,
        "class_mapping": CLASS_MAPPING, "original_class_mapping": {"1": 0, "2": 1},
        "filter_initialization": "zero_state_per_continuous_segment",
        "warmup_seconds": frozen.warmup_seconds, "reference": frozen.reference,
        "training_artifact_policy": "exclude_expert_flagged" if exclude_artifacts else "include_expert_flagged",
        "training_windows": int(eligible.sum()),
        "training_trial_ids": sorted({t.trial_id for t, good in zip(window_trials, eligible) if good}),
        "validation": "leave-one-run-out; all overlapping trial windows grouped by run",
        "independent_test_used": False,
    }
    report = {
        "training_subject": recording.subject_id, "training_session": recording.session_id,
        "synthetic": metadata["synthetic"],
        "data_scope": "Synthetic engineering demonstration; not recorded human EEG." if metadata["synthetic"] else "Recorded EEG, subject-specific training.",
        "validation_scope": "Internal T-session selection; the selected CV score is not an independent test estimate.",
        "window_extraction": "Exact deployed Runtime window observer, causal filtering, neutral replay markers.",
        "processing_time_scope": "CV measures individual model calls plus observed extraction-gate overhead; independent replay measures the actual deployed decoder.",
        "artifact_policy": metadata["training_artifact_policy"],
        "target_trials": sum(t.original_class in (1, 2) for t in trials),
        "expert_flagged_target_trials": sum(t.original_class in (1, 2) and t.artifact_flag for t in trials),
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
                                        "artifact_policy": metadata["training_artifact_policy"]})[:16]
    bundle = ModelBundle(frozen, final_model, sos, model_version, metadata, report)
    bundle.validate()
    return bundle


# ========================================================================
# 16. 运行会话
# ========================================================================

def record_session(source, config, output, observer=None, stop_event=None, context=None):
    """Record calibration EEG and neutral phases without loading a classifier.

    Raw samples are retained even when the causal quality check rejects them.
    Cue classes belong in the paradigm's separate truth log, never this stream.
    """
    stop = stop_event if stop_event is not None else threading.Event()
    recorder = None
    iterator = None
    caught = None
    cancelled = stop.is_set()
    quality_reasons = Counter()
    counts = {"chunks": 0, "samples": 0, "valid_samples": 0, "markers": 0, "runs": 0}
    began = time.perf_counter()
    summary = None
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
                break
            except Exception:
                # LiveSource's bounded no-data timeout may finish a pending
                # pull after the UI has requested cancellation.
                if stop.is_set():
                    cancelled = True
                    break
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
                counts["markers"] += 1
                if packet.kind in {"RUN_START", "DISCONNECT"}:
                    preprocessor.reset()
                if packet.kind == "RUN_START":
                    counts["runs"] += 1
            else:
                raise TypeError("Calibration source must emit EEGChunk or neutral Marker")
            recorder.record(packet)
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
                write_json(recorder.directory / "acquisition_summary.json", summary)
            except BaseException as exc:
                cleanup_errors.append(exc)
            try:
                recorder.close(status=status, error=caught or (cleanup_errors[0] if cleanup_errors else None))
            except BaseException as exc:
                cleanup_errors.append(exc)
        if caught is None and cleanup_errors:
            raise RuntimeError("Calibration recording cleanup failed") from cleanup_errors[0]
    return summary


def run_session(source, bundle, output, observer=None, stop_event=None, queue_capacity=64, context=None):
    if queue_capacity < 1:
        raise ValueError("Acquisition queue capacity must be positive")
    stop = stop_event or threading.Event()
    try:
        decoder = Runtime(bundle.config, bundle.model, sos=bundle.sos, model_version=bundle.version)
        recorder = AsyncRecorder(output, source.info)
        if context is not None:
            write_json(recorder.directory / "capture_context.json", context)
    except BaseException as exc:
        source.close()
        if "recorder" in locals():
            recorder.close(status="failed", error=exc)
        raise
    incoming = queue.Queue(maxsize=queue_capacity)
    done = threading.Event()
    failure = []
    predictions, commands = [], []
    max_backlog = 0

    def acquire():
        iterator = None
        try:
            iterator = iter(source)
            for packet in iterator:
                if stop.is_set():
                    break
                # Offline replay may backpressure. Live input fails closed on overload.
                if getattr(source, "is_live", False):
                    incoming.put_nowait(packet)
                else:
                    while not stop.is_set():
                        try:
                            incoming.put(packet, timeout=0.1)
                            break
                        except queue.Full:
                            continue
        except BaseException as exc:
            failure.append(exc)
            stop.set()
        finally:
            try:
                if iterator is not None and hasattr(iterator, "close"):
                    iterator.close()
            except BaseException as exc:
                failure.append(exc)
                stop.set()
            finally:
                try:
                    source.close()
                except BaseException as exc:
                    failure.append(exc)
                    stop.set()
                done.set()

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
            recorder.record(packet)
            if observer:
                observer(packet)
            # Refresh live age after time spent in the acquisition queue.
            if getattr(source, "is_live", False) and hasattr(packet, "data"):
                from dataclasses import replace
                packet = replace(packet, received_at=source.clock())
            for result in decoder.process(packet):
                recorder.record(result)
                if isinstance(result, Prediction):
                    predictions.append(result)
                elif isinstance(result, Command):
                    commands.append(result)
                if observer:
                    observer(result)
        if failure:
            raise RuntimeError("Acquisition failed or queue overflowed") from failure[0]
        if not stop.is_set():
            for result in decoder.finish():
                recorder.record(result)
                if isinstance(result, Prediction):
                    predictions.append(result)
                elif isinstance(result, Command):
                    commands.append(result)
                if observer:
                    observer(result)
    except BaseException as exc:
        caught = exc
        raise
    finally:
        was_stopped = stop.is_set()
        stop.set()
        cleanup_errors = []
        try:
            source.close()
        except BaseException as exc:
            cleanup_errors.append(exc)
        worker.join(timeout=3)
        if worker.is_alive():
            cleanup_errors.append(RuntimeError("Acquisition thread did not stop within three seconds"))
        try:
            recorder.record({"type": "SessionEnd", "error": str(caught) if caught else None,
                             "cause": str(caught.__cause__) if caught and caught.__cause__ else None,
                             "cancelled": was_stopped, "pending_acquisition_packets": incoming.qsize(),
                             "cleanup_errors": [str(e) for e in cleanup_errors]})
        except BaseException as exc:
            cleanup_errors.append(exc)
        try:
            recorder.close(status="cancelled" if was_stopped else "complete", error=caught or (cleanup_errors[0] if cleanup_errors else None))
        except BaseException as exc:
            cleanup_errors.append(exc)
        summary = {
            "predictions": len(predictions), "commands": len([c for c in commands if c.direction != "NO_COMMAND"]),
            "max_acquisition_backlog": max_backlog, "elapsed_wall_seconds": time.perf_counter() - start_clock,
            "mode": "live" if getattr(source, "is_live", False) else "replay", "model_version": bundle.version,
            "validation_scope": "Live hardware requires separate calibration and validation" if getattr(source, "is_live", False) else "Continuous replay only; not a live closed-loop experiment",
            "pending_acquisition_packets": incoming.qsize(), "cleanup_errors": [str(e) for e in cleanup_errors],
            "decoder_statistics": decoder.stats,
        }
        try:
            write_json(recorder.directory / "runtime_summary.json", summary)
        except BaseException as exc:
            cleanup_errors.append(exc)
        if caught is None and cleanup_errors:
            raise RuntimeError("Session cleanup failed") from cleanup_errors[0]
    return predictions, commands


# ========================================================================
# 17. 中文界面与提示范式
# ========================================================================

REASON_TEXT = {"OK": "信号有效", "CONFIRMED": "方向已确认", "ALREADY_COMMITTED": "本试次已发出指令",
               "LOW_CONFIDENCE": "分数尚未达到阈值", "AWAITING_CONFIRMATION": "等待连续窗口确认",
               "INSUFFICIENT_DATA": "等待完整窗口", "BAD_SIGNAL": "信号质量不足，暂不发令",
               "STALE_DATA": "数据延迟或断流，暂不发令", "MODEL_ERROR": "模型推理异常，暂不发令",
               "OUTSIDE_MI_PHASE": "当前不在想象阶段"}


def show_feedback(task, title="MI · 连续回放", scope="回放验证 · 非真实闭环实验"):
    import tkinter as tk
    from tkinter import ttk
    root = tk.Tk()
    root.title(title)
    root.geometry("860x620")
    root.minsize(720, 620)
    root.configure(bg="#101a2b")
    stop = threading.Event()
    lock = threading.Lock()
    state = {"phase": "连接与预热", "trial": "—", "p": (0.5, 0.5), "command": "等待指令", "reason": "", "samples": 0, "predictions": 0, "commands": 0, "done": False, "error": None}

    def observe(item):
        with lock:
            if isinstance(item, EEGChunk):
                state["samples"] += item.data.shape[1]
            elif isinstance(item, Marker):
                state["phase"] = {"RUN_START": "预热", "TRIAL_START": "准备", "MI_START": "运动想象", "MI_END": "休息", "DISCONNECT": "连接中断"}.get(item.kind, item.kind)
                if item.trial_id:
                    state["trial"] = item.trial_id
                if item.kind in {"TRIAL_START", "MI_START"}:
                    state["command"] = "准备下一次想象" if item.kind == "TRIAL_START" else "积累脑电证据"
                    state["p"] = (0.5, 0.5)
                    state["reason"] = ""
            elif isinstance(item, Prediction):
                state["predictions"] += 1
                if item.probabilities is not None:
                    state["p"] = item.probabilities
                state["reason"] = item.quality
            elif isinstance(item, Command):
                state["reason"] = item.reason
                if item.direction != "NO_COMMAND":
                    state["command"] = "← 左手" if item.direction == "LEFT" else "右手 →"
                    state["commands"] += 1

    def worker():
        try:
            task(observe, stop)
        except Exception as exc:
            with lock:
                state["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            with lock:
                state["done"] = True

    fg, muted = "#eaf3ff", "#9bacc4"
    def label(text, size=12, color=fg, **kwargs):
        return tk.Label(root, text=text, font=("Microsoft YaHei UI", size), bg="#101a2b", fg=color, **kwargs)

    label("MOTOR IMAGERY", 12, "#64d6c3").pack(anchor="w", padx=38, pady=(28, 4))
    label("脑电解码 · 实时反馈", 26).pack(anchor="w", padx=36)
    label(scope, 11, muted).pack(anchor="w", padx=38, pady=(8, 20))
    phase = label("", 16)
    phase.pack(anchor="w", padx=38)
    trial = label("", 10, muted)
    trial.pack(anchor="w", padx=38, pady=(6, 14))
    canvas = tk.Canvas(root, width=780, height=120, bg="#101a2b", highlightthickness=0)
    canvas.pack(fill="x", padx=38)
    command = label("等待指令", 28, "#64d6c3")
    command.pack(pady=(12, 4))
    reason = label("", 10, muted, wraplength=760)
    reason.pack()
    statistics = label("", 10, muted)
    statistics.pack(pady=14)
    status = label("运行中 · 每个试次最多一次方向指令", 10, muted, wraplength=760)
    status.pack(pady=4)

    def close():
        stop.set()
        if state["done"]:
            root.destroy()
        else:
            status.config(text="正在停止并保存记录…")
            root.after(100, close)

    ttk.Button(root, text="停止并保存 / 关闭", command=close).pack(pady=10)
    root.protocol("WM_DELETE_WINDOW", close)

    def refresh():
        with lock:
            snapshot = dict(state)
        phase.config(text=snapshot["phase"])
        trial.config(text=f"试次  {snapshot['trial']}")
        canvas.delete("all")
        width = max(500, canvas.winfo_width())
        for row, (name, value, color) in enumerate(zip(("左手", "右手"), snapshot["p"], ("#64d6c3", "#75abff"))):
            y = row * 56 + 12
            canvas.create_text(0, y + 12, text=name, fill=fg, anchor="w", font=("Microsoft YaHei UI", 13))
            canvas.create_rectangle(66, y, width - 80, y + 25, fill="#24334c", outline="")
            canvas.create_rectangle(66, y, 66 + (width - 146) * value, y + 25, fill=color, outline="")
            canvas.create_text(width - 5, y + 12, text=f"{value:.1%}", fill=fg, anchor="e", font=("Consolas", 13))
        command.config(text=snapshot["command"])
        reason.config(text=f"质量 / 决策：{REASON_TEXT.get(snapshot['reason'], snapshot['reason'] or '等待完整窗口')}")
        statistics.config(text=f"采样点 {snapshot['samples']:,}    窗口 {snapshot['predictions']}    已发指令 {snapshot['commands']}")
        if snapshot["done"]:
            status.config(text=snapshot["error"] or "运行已结束 · 记录已保存", fg="#ffa59a" if snapshot["error"] else muted)
        root.after(80, refresh)

    threading.Thread(target=worker, name="mi-decoder", daemon=True).start()
    refresh()
    root.mainloop()
    if state["error"]:
        raise RuntimeError(state["error"])


def run_paradigm(output, stream_name="MI_Markers", trials=20, seed=42, runs=1, config=None, run_rest_seconds=15):
    """Present cue classes locally; publish neutral phase JSON to LSL.

    Marker timestamps follow Tk redraw submission. They are software onset
    estimates, not a photodiode measurement of physical monitor onset.
    """
    import json
    from pathlib import Path
    import random
    import pylsl
    import tkinter as tk
    from tkinter import ttk

    if trials < 2 or trials % 2:
        raise ValueError("Use a positive even number of trials")
    if runs < 1 or not np.isfinite(run_rest_seconds) or run_rest_seconds < 0:
        raise ValueError("Runs must be positive and between-run rest must be nonnegative")
    config = config or Config()
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    order = []
    for _ in range(runs):
        block = [1, 2] * (trials // 2)
        rng.shuffle(block)
        order.extend(block)
    experiment_id = uuid4().hex[:12]
    # Closing the context covers stream/Tk construction failures as well as
    # callback errors; dropping the outlet releases its native LSL resource.
    with target.open("x", encoding="utf-8") as log:
        outlet = pylsl.StreamOutlet(pylsl.StreamInfo(stream_name, "Markers", 1, 0, "string", f"mi-paradigm-{time.time_ns()}"))
        root = None
        errors = []
        try:
            root = tk.Tk()
            root.title("MI 提示范式 · 类别仅写入独立标签日志")
            root.geometry("700x480")
            root.configure(bg="#101a2b")
            cue = tk.Label(root, text="等待连接", font=("Microsoft YaHei UI", 62), fg="white", bg="#101a2b")
            cue.pack(expand=True)
            note = tk.Label(root, text=f"共 {runs} 组，每组 {trials} 次；准备 2 秒 → 想象 {config.imagery_seconds:g} 秒 → 休息 2 秒", font=("Microsoft YaHei UI", 13), fg="#9bacc4", bg="#101a2b", wraplength=660)
            note.pack(pady=24)
            current = {"index": 0, "id": None, "active": False, "closed": False, "finished": False, "begun": False}

            def emit(kind, class_label=None):
                root.update_idletasks()
                stamp = pylsl.local_clock()
                neutral = {"kind": kind, "trial_id": current["id"]}
                outlet.push_sample([json.dumps(neutral)], timestamp=stamp)
                log.write(json.dumps({**neutral, "timestamp": stamp, "original_class": class_label}) + "\n")
                log.flush()

            def prepare():
                if current["closed"]:
                    return
                i = current["index"]
                if i == len(order):
                    cue.config(text="完成")
                    current["id"] = None
                    emit("SESSION_END")
                    current["finished"] = True
                    note.config(text=f"{len(order)} 个试次已完成，采集将自动保存，可关闭窗口")
                    return
                if i % trials == 0:
                    current["id"] = None
                    emit("RUN_START")
                current["id"] = f"{experiment_id}-{i + 1:04d}"
                cue.config(text="+")
                note.config(text=f"第 {i // trials + 1}/{runs} 组 · 第 {i % trials + 1}/{trials} 次；手保持不动，想象相应手的运动感觉")
                emit("TRIAL_START")
                root.after(2000, imagine)

            def imagine():
                if current["closed"]:
                    return
                current["active"] = True
                klass = order[current["index"]]
                cue.config(text="←" if klass == 1 else "→")
                emit("MI_START", klass)
                root.after(round(config.imagery_seconds * 1000), rest)

            def rest():
                if current["closed"]:
                    return
                cue.config(text="休息")
                emit("MI_END")
                current["active"] = False
                current["index"] += 1
                between_runs = current["index"] < len(order) and current["index"] % trials == 0
                if between_runs:
                    note.config(text=f"组间休息 {run_rest_seconds:g} 秒；请保持设备和坐姿稳定")
                root.after(round((run_rest_seconds if between_runs else 2) * 1000), prepare)

            def begin():
                if current["closed"]:
                    return
                if not outlet.have_consumers():
                    note.config(text="还没有采集程序连接。请先启动 record 或 live，再点击开始。")
                    return
                start_button.destroy()
                current["begun"] = True
                prepare()

            def close():
                if current["closed"]:
                    return
                current["closed"] = True
                try:
                    if current["begun"] and not current["finished"]:
                        emit("DISCONNECT")
                finally:
                    current["active"] = False
                    root.destroy()

            def callback_error(exc_type, exc, traceback):
                errors.append(exc)
                try:
                    close()
                except Exception as cleanup_error:
                    errors.append(cleanup_error)

            root.report_callback_exception = callback_error
            start_button = ttk.Button(root, text="设备与解码器连接后，开始实验", command=begin)
            start_button.pack(pady=20)
            root.protocol("WM_DELETE_WINDOW", close)
            root.mainloop()
        finally:
            if root is not None:
                try:
                    root.destroy()
                except tk.TclError:
                    pass
            outlet = None
        if errors:
            raise RuntimeError("Paradigm stopped after a UI/LSL callback failed") from errors[0]


# ========================================================================
# 18. 命令行与主入口
# ========================================================================

def _read(path, labels=None, strict=True):
    if Path(path).suffix.lower() == ".gdf":
        return load_gdf(path, labels_path=labels, strict=strict)
    recording = load_recording(path)
    if labels is not None:
        if recording.session_id != "E":
            raise ValueError("External evaluation labels may only be attached to an E session")
        truth = load_labels(labels)
        if len(truth) != len(recording.trials):
            raise ValueError("Label count must match ALL original E trials before any class or artifact filtering")
        if any(t.original_class is not None and t.original_class != value for t, value in zip(recording.trials, truth)):
            raise ValueError("External labels conflict with the labels already stored in the recording")
        recording.trials = [replace(t, original_class=int(value)) for t, value in zip(recording.trials, truth)]
        recording.metadata = {**recording.metadata, "labels_file": Path(labels).name}
    return recording


def _timeline_digest(recording):
    """Identity of signal, timestamps and neutral cue alignment, excluding truth."""
    hasher = sha256()
    for array in (recording.data, recording.timestamps):
        hasher.update(np.ascontiguousarray(array).view(np.uint8))
    neutral = {"subject": recording.subject_id, "session": recording.session_id,
               "stream": asdict(recording.info), "run_starts": recording.run_starts,
               "trials": [{key: getattr(trial, key) for key in ("trial_id", "run_id", "cue_sample", "start_sample")}
                          for trial in recording.trials]}
    hasher.update(canonical_json(neutral).encode("utf-8"))
    return hasher.hexdigest()


def _config(path=None):
    return Config.from_dict(json.loads(Path(path).read_text(encoding="utf-8-sig"))) if path else Config()


def _brief(report):
    scores = report["all"]
    def percent(value):
        return "N/A" if value is None else f"{value:.1%}"
    print(f"Trial accuracy: {percent(scores['accuracy'])} | balanced: {percent(scores['balanced_accuracy'])}")
    print(f"Command accuracy: {percent(scores['command_accuracy'])} | coverage: {percent(scores['coverage'])} | correct / all: {percent(scores['correct_command_fraction'])}")


def _replay(recording, bundle, output, ui=False, realtime=False, speed=1.0):
    if recording.subject_id != bundle.metadata.get("training_subject"):
        raise ValueError("Replay subject does not match this subject-specific model's training subject")
    context = {"format": "mi-bci-replay-context-v1", "model_version": bundle.version,
               "config": bundle.config.to_dict(), "subject_id": recording.subject_id,
               "session_id": recording.session_id, "timeline_sha256": _timeline_digest(recording)}
    source = ReplaySource(recording, chunk_samples=bundle.config.chunk_samples,
                          realtime=realtime or ui, speed=speed, imagery_seconds=bundle.config.imagery_seconds)
    result = {}
    def task(observer=None, stop=None):
        predictions, commands = run_session(source, bundle, output, observer, stop)
        write_json(Path(output) / "replay_context.json", context)
        report = evaluate(recording.trials, predictions, commands, sfreq=bundle.config.sfreq, imagery_seconds=bundle.config.imagery_seconds)
        report["data_provenance"] = recording.metadata
        report["mode"] = "synthetic_replay" if recording.metadata.get("synthetic") else "continuous_replay"
        report["evaluation_role"] = "independent_E_session" if recording.session_id == "E" else "training_session_replay_not_independent"
        report["session_status"] = json.loads((Path(output) / "manifest.json").read_text(encoding="utf-8"))["status"]
        write_json(Path(output) / "evaluation.json", report)
        result["report"] = report
    if ui:
        synthetic = recording.metadata.get("synthetic", False)
        show_feedback(task, scope="合成数据回放 · 仅工程验证" if synthetic else "连续数据回放 · 非真实闭环实验")
    else:
        task()
    if "report" in result:
        _brief(result["report"])
    return result.get("report")


def cmd_demo(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    print("Creating synthetic T/E sessions. These scores are not human EEG results.", flush=True)
    train_data = make_synthetic_recording(session_id="T", seed=args.seed)
    test_data = make_synthetic_recording(session_id="E", seed=args.seed + 1000)
    save_recording(output / "train.npz", train_data)
    save_recording(output / "evaluation.npz", test_data)
    print("Fitting CSP + shrinkage LDA with run-grouped validation...", flush=True)
    bundle = train_recording(train_data, _config(args.config))
    bundle.save(output / "model")
    write_json(output / "training_report.json", bundle.report)
    print(f"Frozen decision policy: {bundle.report['selected_policy']}", flush=True)
    _replay(test_data, bundle, output / "session", args.ui, args.realtime, args.speed)
    print(f"Demo saved: {output.resolve()}")


def cmd_import(args):
    recording = load_gdf(args.gdf, labels_path=args.labels, strict=not args.relaxed)
    if Path(args.output).exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    save_recording(args.output, recording)
    print(json.dumps({"subject": recording.subject_id, "session": recording.session_id, "trials": len(recording.trials), "left": sum(t.original_class == 1 for t in recording.trials), "right": sum(t.original_class == 2 for t in recording.trials), "unknown": sum(t.original_class is None for t in recording.trials)}, indent=2))


def cmd_train(args):
    if Path(args.model).exists():
        raise FileExistsError(f"Model destination exists: {args.model}")
    recording = _read(args.data)
    config = (_config(args.config) if args.config else
              Config.from_dict(recording.metadata["config"]) if "config" in recording.metadata else Config())
    bundle = train_recording(recording, config)
    bundle.save(args.model)
    print(f"Saved frozen model {bundle.version}: {Path(args.model).resolve()}")
    print(json.dumps(bundle.report["selected_policy"], indent=2))


def cmd_replay(args):
    bundle = ModelBundle.load(args.model)
    recording = _read(args.data, args.labels)
    _replay(recording, bundle, args.output, args.ui, args.realtime, args.speed)


def cmd_evaluate(args):
    recording = _read(args.data, args.labels)
    context_path = Path(args.session) / "replay_context.json"
    context = json.loads(context_path.read_text(encoding="utf-8")) if context_path.exists() else None
    if context is not None and context.get("timeline_sha256") != _timeline_digest(recording):
        raise ValueError("Scoring recording differs from the replay signal or neutral trial timeline")
    config = Config.from_dict(context["config"]) if context else None
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
    report = evaluate(recording.trials, predictions, commands, sfreq=recording.info.sfreq,
                      imagery_seconds=config.imagery_seconds if config else 4.0)
    report["data_provenance"] = recording.metadata
    report["mode"] = "post_hoc_replay_scoring" if context else "post_hoc_scoring_without_replay_context"
    report["alignment_verified"] = context is not None
    manifest_path = Path(args.session) / "manifest.json"
    report["session_status"] = json.loads(manifest_path.read_text(encoding="utf-8")).get("status") if manifest_path.exists() else "unknown"
    write_json(args.output, report)
    _brief(report)


def cmd_batch(args):
    data_dir = Path(args.data_dir)
    labels_dir = Path(args.labels_dir) if args.labels_dir else data_dir
    paths, missing = [], []
    for subject in args.subjects:
        pair = []
        for session in ("T", "E"):
            candidates = [data_dir / f"{subject}{session}{extension}" for extension in (".npz", ".gdf")]
            found = next((p for p in candidates if p.exists()), None)
            if found is None:
                missing.append(f"{subject}{session}.npz or .gdf")
            pair.append(found)
        label = labels_dir / f"{subject}E.mat"
        if pair[1] is not None and pair[1].suffix == ".gdf" and not label.exists():
            missing.append(str(label))
        paths.append((subject, pair, label))
    if missing:
        raise FileNotFoundError("Missing batch inputs: " + ", ".join(missing))
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    subjects = []
    for subject, (train_path, evaluation_path), label in paths:
        print(f"Training and replaying {subject}...", flush=True)
        training = _read(train_path)
        testing = _read(evaluation_path, label if label.exists() else None)
        if any(t.original_class is None for t in testing.trials):
            raise ValueError(f"{subject}: independent E truth is required for batch scoring")
        if training.subject_id != testing.subject_id or training.subject_id != subject or testing.session_id != "E":
            raise ValueError("Batch subject/session pairing mismatch")
        bundle = train_recording(training, _config(args.config))
        bundle.save(root / subject / "model")
        report = _replay(testing, bundle, root / subject / "session")
        subjects.append({"subject": subject, **report["all"]})
        write_json(root / "summary.json", {"status": "running", "subjects": subjects})
    metrics = ("accuracy", "balanced_accuracy", "command_accuracy", "coverage", "correct_command_fraction")
    means = {key: sum(s[key] for s in subjects if s[key] is not None) / sum(s[key] is not None for s in subjects) if any(s[key] is not None for s in subjects) else None for key in metrics}
    write_json(root / "summary.json", {"status": "complete", "subjects": subjects, "macro_mean": means, "scope": "Each subject: T run-grouped selection then frozen E continuous replay"})
    print(f"Batch summary: {(root / 'summary.json').resolve()}")


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
    # Use the rate actually exported by the GUI; this command does not reprogram the amplifier.
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
        print("Connected. Click Start in the paradigm window. Acquisition saves automatically at SESSION_END.", flush=True)
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


def cmd_evaluate_live(args):
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
    config = Config.from_dict(context["config"])
    report = evaluate(recording.trials, predictions, commands, sfreq=config.sfreq, imagery_seconds=config.imagery_seconds)
    if report["stability"]["unknown_trial_predictions"] or report["stability"]["unknown_trial_direction_commands"]:
        raise ValueError("Predictions or commands reference trials outside this capture")
    report.update(mode="personal_online_test", alignment_verified=True, model_version=context["model_version"],
                  subject_id=recording.subject_id, session_status="complete", data_provenance=recording.metadata)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    _brief(report)


def cmd_live(args):
    if Path(args.output).exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    bundle = ModelBundle.load(args.model)
    if bundle.metadata.get("synthetic"):
        raise ValueError("Synthetic demonstration models cannot be used for a personal live experiment")
    if args.subject != bundle.metadata.get("training_subject"):
        raise ValueError("The live subject must match the model's calibration subject")
    source = LiveSource(bundle.config, args.eeg_name, args.marker_name, args.reference or bundle.config.reference, args.units)
    try:
        context = _capture_context(args, bundle.config, source, "live", bundle.version)
        print("Connected. Start the independent test in the paradigm window.", flush=True)
        def task(observer=None, stop=None):
            run_session(source, bundle, args.output, observer, stop, context=context)
        if args.ui:
            show_feedback(task, title="MI · LSL 实时采集", scope=f"设备 EEG · {args.subject} · 独立在线测试")
        else:
            task()
    finally:
        source.close()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        app_data = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / ".local" / "share"))
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
        output = app_data / "MI_BCI" / "runs" / run_name
        argv = ["demo", "--ui", "--output", str(output)]
    parser = argparse.ArgumentParser(prog="main.py", description="Causal cue-synchronous motor imagery research system")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Generate synthetic T/E, train, replay, record and score")
    demo.add_argument("--output", default="outputs/demo")
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--config")
    demo.set_defaults(func=cmd_demo)
    imp = commands.add_parser("import-gdf", help="Import complete BCI IV 2a timeline")
    imp.add_argument("gdf")
    imp.add_argument("--labels", help="Official classlabel MAT; attach before filtering")
    imp.add_argument("--output", required=True)
    imp.add_argument("--relaxed", action="store_true", help="Relax standard dataset counts, for fixtures only")
    imp.set_defaults(func=cmd_import)
    train = commands.add_parser("train", help="T-only run-grouped CSP/LDA fitting")
    train.add_argument("data")
    train.add_argument("--model", required=True)
    train.add_argument("--config")
    train.set_defaults(func=cmd_train)
    replay = commands.add_parser("replay", help="Replay all trials with neutral markers")
    replay.add_argument("data")
    replay.add_argument("--model", required=True)
    replay.add_argument("--labels")
    replay.add_argument("--output", required=True)
    replay.set_defaults(func=cmd_replay)
    for command in (demo, replay):
        command.add_argument("--ui", action="store_true", help="Show probability bars and command feedback; enables pacing")
        command.add_argument("--realtime", action="store_true")
        command.add_argument("--speed", type=float, default=1.0)
    evaluation = commands.add_parser("evaluate", help="Score previously recorded predictions separately")
    evaluation.add_argument("--data", required=True)
    evaluation.add_argument("--session", required=True)
    evaluation.add_argument("--labels")
    evaluation.add_argument("--output", required=True)
    evaluation.set_defaults(func=cmd_evaluate)
    batch = commands.add_parser("batch", help="Run paired subject-specific T/E evaluations")
    batch.add_argument("--data-dir", required=True)
    batch.add_argument("--labels-dir")
    batch.add_argument("--output", required=True)
    batch.add_argument("--subjects", nargs="+", default=[f"A{i:02d}" for i in range(1, 10)])
    batch.add_argument("--config")
    batch.set_defaults(func=cmd_batch)
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
    live.add_argument("--ui", action="store_true")
    live.set_defaults(func=cmd_live)
    paradigm = commands.add_parser("paradigm", help="Present actual cues; emit only neutral LSL events")
    paradigm.add_argument("--output", required=True, help="Separate cue truth JSONL")
    paradigm.add_argument("--marker-name", default="MI_Markers")
    paradigm.add_argument("--trials", type=int, default=20, help="Even number of trials PER run, balanced left/right")
    paradigm.add_argument("--runs", type=int, default=1)
    paradigm.add_argument("--run-rest", type=float, default=15, help="Seconds of rest between runs")
    paradigm.add_argument("--seed", type=int, help="Optional reproducible randomization; otherwise fresh order")
    paradigm_config = paradigm.add_mutually_exclusive_group()
    paradigm_config.add_argument("--config", help="Use the same device config as calibration capture")
    paradigm_config.add_argument("--model", help="Use the frozen model's timing for an online test")
    def paradigm_run(args):
        config = ModelBundle.load(args.model).config if args.model else _config(args.config)
        run_paradigm(args.output, args.marker_name, args.trials, args.seed, args.runs, config, args.run_rest)
    paradigm.set_defaults(func=paradigm_run)
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("Stopped; session records were finalized.", file=sys.stderr)
        return 130
    except ImportError as exc:
        parser.exit(1, f"Missing dependency: {exc}. Install with python -m pip install -e .[live]\n")
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
