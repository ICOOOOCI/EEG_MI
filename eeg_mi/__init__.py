"""EEG 运动想象应用的分层包。

模块职责：

* :mod:`constants`：类别和单位契约；
* :mod:`config`：不可变运行配置；
* :mod:`domain`：采集数据对象与持久化；
* :mod:`models`：可替换分类器实现；
* :mod:`cli`：命令行业务编排；
* :mod:`gui`：Qt 采集界面。
"""

from .constants import *
from .config import Config
from .domain import (Command, EEGChunk, Marker, Prediction, Recording, Source,
                     StreamInfo, Trial, WindowModel, load_recording, save_recording)
from .models import (EEGNetClassifier, HybridClassifier, RiemannianClassifier,
                     build_model, validate_fitted_model)
from .processing import CausalPreprocessor, DecisionPolicy, RingBuffer, Runtime
from .transport import AsyncRecorder, ReplaySource, write_json
from .lsl import LiveSource
from .quality import QualityThresholds, channel_quality_report, timing_report
from .safety import LiveSafety, MarkerProtocol
from .training import (CalibrationDataError, ModelBundle, evaluate, extract_windows,
                       load_bundle, load_personal_session, make_pipeline,
                       train_recording)

__version__ = "0.6.0"

__all__ = [
    "Config", "Command", "EEGChunk", "Marker", "Prediction", "Recording",
    "Source", "StreamInfo", "Trial", "WindowModel", "load_recording",
    "save_recording", "EEGNetClassifier", "HybridClassifier",
    "RiemannianClassifier", "build_model", "validate_fitted_model",
    "CausalPreprocessor", "DecisionPolicy", "RingBuffer", "Runtime",
    "AsyncRecorder", "ReplaySource", "write_json", "LiveSource",
    "QualityThresholds", "channel_quality_report", "timing_report",
    "LiveSafety", "MarkerProtocol",
    "CalibrationDataError", "ModelBundle", "evaluate", "extract_windows",
    "load_bundle", "load_personal_session", "make_pipeline", "train_recording",
]
