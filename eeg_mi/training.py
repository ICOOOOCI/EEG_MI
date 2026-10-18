"""个人会话导入、模型包、评估与训练服务。

这些函数构成业务层，CLI 只负责参数解析和调用，不再承载模型/数据持久化细节。
"""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, fields, replace
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version as package_version
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
import json
import platform
import warnings

import joblib
import mne
import numpy as np
from scipy.signal import butter
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from .config import Config
from .constants import (ALGORITHMS, CLASS_IDS, CLASS_MAPPING, CLASS_NAMES,
                        N_CLASSES, TRUTH_IDS, TRUTH_MAPPING)
from .domain import (Command, Prediction, Recording, StreamInfo, Trial,
                     load_recording, save_recording)
from .models import build_model, validate_fitted_model
from .processing import Runtime
from .quality import QualityThresholds
from .transport import ReplaySource


__version__ = "0.6.0"


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
    for key, expected in (("class_mapping", CLASS_MAPPING), ("truth_mapping", TRUTH_MAPPING)):
        if key in context and context[key] != expected:
            raise ValueError("Capture class mapping differs from the four-class label contract")
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
    truth_manifest_path = labels_path.parent / "manifest.json"
    if truth_manifest_path.is_file():
        truth_manifest = read_object(truth_manifest_path)
        if "truth_mapping" in truth_manifest and truth_manifest["truth_mapping"] != TRUTH_MAPPING:
            raise ValueError("Truth-log class mapping differs; FEET=3 and TONGUE=4 are required")
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
            if isinstance(original, bool) or not isinstance(original, int) or original not in TRUTH_IDS:
                raise ValueError("Personal truth labels must be 1=LEFT, 2=RIGHT, 3=FEET or 4=TONGUE")
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
                     metadata={"source": "personal_lsl", "class_mapping": CLASS_MAPPING,
                               "capture_directory": directory.name, "labels_file": labels_path.name,
                               "capture_context": context, "config": config.to_dict(),
                               "label_alignment": "Exact trial IDs and phase sequence; cue samples from received neutral markers",
                               "artifact_annotation": "Not manually reviewed; runtime signal-quality checks remain active"})





# ========================================================================
# 12. 冻结模型包与一致性校验
# ========================================================================
def canonical_json(value):
    # 固定键顺序和分隔符，使同一份元数据生成稳定的摘要。
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def dependency_versions():
    versions = {"python": platform.python_version(), **{
        name: package_version(name) for name in ("numpy", "scipy", "scikit-learn", "mne", "joblib")}}
    for name in ("pyriemann", "torch", "pylsl"):
        try:
            versions[name] = package_version(name)
        except PackageNotFoundError:
            pass
    return versions


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
        if not np.array_equal(classes, CLASS_IDS):
            raise ValueError("Four-class model required: probability columns must be [LEFT, RIGHT, FEET, TONGUE]; retrain old binary models")
        algorithm = self.metadata.get("algorithm")
        if algorithm is not None:
            validate_model_profile(self.config)
            validate_fitted_model(self.model, algorithm, len(self.config.channels),
                                  self.config.window_samples, self.config.sfreq)
        else:
            # 旧包只读兼容；新训练入口不再生成 CSP/LDA。
            steps = getattr(self.model, "named_steps", {})
            csp, lda = steps.get("csp"), steps.get("lda")
            if csp is None or lda is None:
                raise ValueError("Expected a fitted supported model and its algorithm metadata")
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
            raise ValueError("Bundle class mapping must be [LEFT, RIGHT, FEET, TONGUE]; retrain old binary models")
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
    balanced = float(balanced_accuracy_score(truth, pred)) if set(truth) == set(CLASS_IDS) else None
    return {
        "target_trials": len(rows), "classified_trials": len(classified),
        "missing_classifications": len(rows) - len(classified),
        "accuracy": float(accuracy_score(truth, pred)) if truth else None,
        "balanced_accuracy": balanced,
        "confusion_matrix": confusion_matrix(truth, pred, labels=CLASS_IDS).tolist() if truth else np.zeros((N_CLASSES, N_CLASSES), dtype=int).tolist(),
        "confusion_labels": list(CLASS_NAMES),
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


def evaluate(trials, predictions, commands, *, sfreq, imagery_seconds=5.0,
             window_seconds=3.0, step_seconds=0.2, cue_offset_seconds=0.5, max_age_seconds=0.5):
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
           or _value(t, "original_class") not in TRUTH_IDS for t in trials):
        raise ValueError("Scoring requires personal truth labels 1=LEFT, 2=RIGHT, 3=FEET or 4=TONGUE")
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
            if probs.shape != (N_CLASSES,) or not np.isfinite(probs).all() or np.any(probs < 0) or not np.isclose(probs.sum(), 1):
                raise ValueError("Invalid probability vector in evaluation")
            valid.append(probs)
        missing_windows = len(expected_starts - observed_starts)
        if missing_windows:
            invalid["MISSING_WINDOW"] += missing_windows
        mean = np.mean(valid, axis=0) if valid else None
        direction_commands = sorted(
            [c for c in cmd_by[trial_id] if _value(c, "direction") in CLASS_NAMES],
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
        command_class = CLASS_NAMES.index(_value(first, "direction")) if first else None
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
        "scope": "Personal four-class cue-synchronous trial scoring.",
        "aggregation": "Mean valid window probabilities per trial; first directional command per trial.",
        "probability_note": "Classifier probabilities are not calibrated correctness guarantees.",
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
            "unknown_trial_direction_commands": sum(_value(c, "trial_id") not in trial_ids and _value(c, "direction") in CLASS_NAMES for c in commands),
            "prediction_processing_seconds": _distribution([float(_value(p, "processing_seconds", 0.0)) for p in predictions]),
        },
        "per_trial": target_rows,
    }



# ========================================================================
# 14. Riemannian / EEGNet 独立训练及按实验组验证
# ========================================================================
class _WindowProbe:
    # 窗口提取用的占位模型返回等概率；有效窗口由观察回调收集。
    def predict_proba(self, windows):
        return np.full((len(windows), N_CLASSES), 1.0 / N_CLASSES)


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


def validate_model_profile(config):
    if len(config.channels) != 16 or not math.isclose(config.sfreq, 250.0, rel_tol=0, abs_tol=1e-8):
        raise ValueError("New models require 16 EEG channels at 250 Hz; no implicit resampling or channel selection")
    if not math.isclose(config.window_seconds, 3.0, rel_tol=0, abs_tol=1e-8):
        raise ValueError("New models require 3-second windows (16 x 750 samples)")


def make_pipeline(config, algorithm="hybrid", **options):
    validate_model_profile(config)
    return build_model(algorithm, len(config.channels), config.window_samples, config.sfreq, **options)


def _fit_model(config, x, y, *, groups=None, **options):
    # 每折重新学习切空间参考/标准化/网络权重；不读取留出组或 E 会话。
    if set(np.asarray(y).tolist()) != set(CLASS_IDS):
        raise ValueError("Every training fold needs valid LEFT, RIGHT, FEET and TONGUE windows")
    pipeline = make_pipeline(config, **options)
    with mne.use_log_level("ERROR"):
        if pipeline.algorithm == "hybrid":
            pipeline.fit(x, y, groups=groups)
        else:
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
                      max_rejected_trial_fraction=0.5, max_rejected_window_fraction=0.5,
                      min_runs=2):
    # 按独立试次审计采集数量和拒绝比例，重叠窗口不增加独立试次数。
    # 这些门限是工程最低要求，并不保证真人实验的识别效果。
    if isinstance(min_runs, bool) or not isinstance(min_runs, Integral) or min_runs < 2:
        raise ValueError("min_runs must be an integer >= 2")
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
        target = dict.fromkeys(CLASS_NAMES, 0)
        valid = dict(target)
        fit = dict(target)
        reasons = Counter()
        valid_windows = 0
        missing_windows = 0
        for trial in trials:
            label = CLASS_NAMES[trial.original_class - 1]
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
            issues.append(f"Run {run_id}: valid four-class trial imbalance exceeds {max_class_ratio:g}")
        if rejected_trial_fraction > max_rejected_trial_fraction:
            issues.append(f"Run {run_id}: rejected fit trials {rejected_trial_fraction:.1%} exceed {max_rejected_trial_fraction:.1%}")
        if rejected_window_fraction > max_rejected_window_fraction:
            issues.append(f"Run {run_id}: rejected/missing windows {rejected_window_fraction:.1%} exceed {max_rejected_window_fraction:.1%}")
    if len(rows) < min_runs:
        issues.append(f"At least {min_runs} complete acquisition runs are required; collect additional balanced runs")
    return {
        "status": "READY_TO_TRAIN" if not issues else "NEEDS_MORE_COLLECTION",
        "limits": {"min_runs": int(min_runs), "min_trials_per_class_per_run": int(min_trials_per_class_per_run),
                   "max_class_ratio": float(max_class_ratio),
                   "max_rejected_trial_fraction": float(max_rejected_trial_fraction),
                   "max_rejected_window_fraction": float(max_rejected_window_fraction)},
        "trial_count_note": "Independent trials, not overlapping windows. Minimum counts do not establish online readiness.",
        "window_rejection_note": "Quality-rejected and missing scheduled windows; expert flags separately exclude trials from fitting.",
        "runs": rows, "issues": issues,
    }


def train_recording(recording, config=None, *, thresholds=(0.4, 0.5, 0.6, 0.7, 0.8), confirmations=(2, 3), exclude_artifacts=True,
                    min_trials_per_class_per_run=10, max_class_ratio=2.0,
                    max_rejected_trial_fraction=0.5, max_rejected_window_fraction=0.5,
                    algorithm="hybrid", seed=42, eegnet_epochs=100,
                    eegnet_batch_size=32, eegnet_learning_rate=1e-3):
    # 仅用 T 会话拟合和选参；按整组 run 留出验证，避免重叠窗口跨折泄漏。
    # 人工伪迹标签只影响拟合样本选择，验证报告仍包含带伪迹标记的试次。
    acquisition_config = _recording_config(recording)
    config = config if config is not None else replace(acquisition_config, window_seconds=3.0)
    if recording.metadata.get("synthetic"):
        raise ValueError("Training requires recorded EEG; legacy synthetic recordings are not accepted")
    if recording.metadata.get("source") != "personal_lsl":
        raise ValueError("Training requires a personal LSL calibration recording")
    if recording.session_id != "T":
        raise ValueError("Training requires a T session; E data must never select or fit parameters")
    model_options = {"algorithm": algorithm, "seed": seed}
    if algorithm in {"eegnet", "hybrid"}:
        model_options.update(eegnet_epochs=eegnet_epochs, eegnet_batch_size=eegnet_batch_size,
                             eegnet_learning_rate=eegnet_learning_rate)
    # 在提取窗口前检查设备、输入形状、算法名与超参数。
    make_pipeline(config, **model_options)
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
           or t.original_class not in TRUTH_IDS for t in trials):
        raise ValueError("Training requires personal truth labels 1=LEFT, 2=RIGHT, 3=FEET or 4=TONGUE")
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
                              max_rejected_window_fraction=max_rejected_window_fraction,
                              min_runs=3 if algorithm == "hybrid" else 2)
    if audit["issues"]:
        raise CalibrationDataError(audit)
    valid_predictions = [extraction.predictions[i] for i in extraction.valid_prediction_indices]
    window_trials = [truth[p.trial_id] for p in valid_predictions]
    eligible = np.asarray([not exclude_artifacts or not t.artifact_flag
                           for t in window_trials], dtype=bool)
    if not eligible.any():
        raise ValueError("No quality-approved four-class training windows")
    # 原始标签 1–4 转为模型类别 0–3；同一试次的所有窗口保留所属 run。
    labels = np.asarray([t.original_class - 1 for t in window_trials])
    groups = np.asarray([t.run_id for t in window_trials])
    oof_predictions = list(extraction.predictions)
    fold_records = []
    # 外层整组留出；Hybrid 的内层交叉拟合只能看到外层训练组。
    for run_id in run_ids:
        fit_indices = np.flatnonzero(eligible & (groups != run_id))
        held_indices = np.flatnonzero(groups == run_id)
        model = _fit_model(config, extraction.windows[fit_indices], labels[fit_indices],
                           groups=groups[fit_indices], **model_options)
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
            "hybrid_inner_folds": getattr(model, "stacking_report_", None),
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
    final_model = _fit_model(frozen, extraction.windows[eligible], labels[eligible],
                             groups=groups[eligible], **model_options)
    source_hash = _recording_digest(recording)
    versions = dependency_versions()
    metadata = {
        "algorithm": algorithm, "model_parameters": model_options,
        "input_shape": [len(frozen.channels), frozen.window_samples],
        "software_version": __version__,
        "hybrid_training": getattr(final_model, "stacking_report_", None),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_subject": recording.subject_id, "training_session": recording.session_id,
        "recording_metadata": dict(recording.metadata),
        "recording_sha256": source_hash, "dependencies": versions,
        "class_mapping": CLASS_MAPPING, "original_class_mapping": {str(value): value - 1 for value in TRUTH_IDS},
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
        "algorithm": algorithm, "model_parameters": model_options,
        "input_shape": [len(frozen.channels), frozen.window_samples],
        "preprocessing": {"bandpass_hz": [frozen.low_hz, frozen.high_hz],
                          "filter_order": frozen.filter_order, "causal": True,
                          "window_seconds": frozen.window_seconds, "sfreq": frozen.sfreq},
        "fit_scope": "Each outer fold learns all preprocessing/model parameters on its training runs only. Hybrid neural training uses inner leave-one-run-out LR probabilities; the outer held-out run never enters either stage. Neural epochs are fixed without E-session selection.",
        "inference_flow": ("window -> Riemannian/TS/LR probabilities -> EEGNet residual fusion with the same EEG window -> four-class probabilities"
                           if algorithm == "hybrid" else algorithm),
        "hybrid_training": getattr(final_model, "stacking_report_", None),
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
    model_version = algorithm + "-" + digest({"recording": source_hash, "config": frozen.to_dict(), "dependencies": versions,
                                        "model_parameters": model_options, "software_version": __version__,
                                        "window_quality_limits": metadata["window_quality_limits"],
                                        "artifact_policy": metadata["training_artifact_policy"]})[:16]
    bundle = ModelBundle(frozen, final_model, sos, model_version, metadata, report)
    bundle.validate()
    return bundle



# ========================================================================


__all__ = [
    "load_personal_session", "canonical_json", "digest", "dependency_versions",
    "ModelBundle", "load_bundle", "evaluate", "ExtractedWindows",
    "extract_windows", "validate_model_profile", "make_pipeline",
    "CalibrationDataError", "audit_calibration", "train_recording",
]
