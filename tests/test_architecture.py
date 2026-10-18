"""不依赖真实 LSL/硬件的分层契约冒烟测试。"""

import numpy as np

from eeg_mi.config import Config
from eeg_mi.domain import EEGChunk, Marker, StreamInfo
from eeg_mi.processing import Runtime
from eeg_mi.lsl import parse_neutral_marker


class UniformModel:
    def predict_proba(self, windows):
        return np.tile([[0.25, 0.25, 0.25, 0.25]], (len(windows), 1))


def test_configuration_and_marker_contract_roundtrip():
    config = Config(sfreq=250, channels=("C1", "C2", "C3", "C4"), reference="REF")
    assert Config.from_dict(config.to_dict()) == config
    assert parse_neutral_marker('{"kind":"MI_START","trial_id":"trial-1"}') == {
        "kind": "MI_START", "trial_id": "trial-1"
    }


def test_runtime_uses_shared_domain_contract():
    config = Config(sfreq=250, channels=("C1", "C2", "C3", "C4"), reference="REF")
    info = StreamInfo(config.sfreq, config.channels, config.reference)
    runtime = Runtime(config, UniformModel())
    runtime.process(Marker("RUN_START", 0))
    runtime.process(Marker("TRIAL_START", 1, "trial-1"))
    runtime.process(Marker("MI_START", 2, "trial-1", timestamp=0.008))
    data = np.random.default_rng(7).normal(0, 2e-6, (4, 1600))
    timestamps = np.arange(1600) / 250 + 0.012
    outputs = runtime.process(EEGChunk(data, timestamps, 2, info))
    assert sum(type(item).__name__ == "Prediction" for item in outputs) == 8
