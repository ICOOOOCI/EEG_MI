"""EEG 领域对象与个人会话 NPZ 存储。

这些对象是算法、采集适配器和 UI 之间唯一共享的数据契约。
"""

from dataclasses import asdict, dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any, Iterator, Protocol
import json

import numpy as np

from .constants import CLASS_IDS, TRUTH_IDS
from .config import Config


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
    probabilities: tuple[float, ...] | None  # 四列顺序：LEFT、RIGHT、FEET、TONGUE
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
    direction: str  # LEFT、RIGHT、FEET、TONGUE 或 NO_COMMAND
    sample: int
    timestamp: float
    reason: str
    processing_seconds: float = 0.0
    emitted_at: float | None = None  # 在线发布时的 LSL 时刻；离线判决可为空
    cue_to_output_seconds: float | None = None
    result_age_seconds: float | None = None


class WindowModel(Protocol):
    # 模型输入为“窗口数 × 通道数 × 采样点数”，输出四列类别概率。
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
    original_class: int  # 真值：1=LEFT，2=RIGHT，3=FEET，4=TONGUE；训练时减 1
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
               or t.original_class not in TRUTH_IDS for t in self.trials):
            raise ValueError("Personal truth labels must be 1=LEFT, 2=RIGHT, 3=FEET or 4=TONGUE")
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

__all__ = [
    "StreamInfo", "EEGChunk", "Marker", "Prediction", "Command",
    "WindowModel", "Source", "Trial", "Recording", "save_recording",
    "load_recording",
]
