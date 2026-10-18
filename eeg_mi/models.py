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
from collections import OrderedDict
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

from .constants import (ALGORITHMS, CLASS_CUES, CLASS_IDS, CLASS_MAPPING, CLASS_NAMES,
                       N_CLASSES, TRUTH_IDS, TRUTH_MAPPING)


__version__ = "0.6.0"



# ========================================================================
# 00. 串联模型 Hybrid，以及 Riemannian / EEGNet 对照模型
# ========================================================================
# 输入为以 V 为单位的 (窗口数, 通道数, 采样点数)，由下方 Runtime 统一滤波。
# EEGNet 结构参考：https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py
# 切空间参考只在 fit 学习，在线不更新：
# https://pyriemann.readthedocs.io/en/latest/generated/pyriemann.tangentspace.TangentSpace.html




def _positive_integer(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _check_parameters(model):
    _positive_integer("n_channels", model.n_channels, 2)
    _positive_integer("n_samples", model.n_samples, 32)
    _positive_integer("seed", model.seed, 0)
    if model.seed > 2**32 - 1:
        raise ValueError("seed must be between 0 and 2**32 - 1")
    if (isinstance(model.sfreq, bool) or not isinstance(model.sfreq, Real)
            or not np.isfinite(model.sfreq) or model.sfreq <= 0):
        raise ValueError("sfreq must be a positive finite number")
    if model.algorithm in {"eegnet", "hybrid"}:
        _positive_integer("epochs", model.epochs)
        _positive_integer("batch_size", model.batch_size)
        if (isinstance(model.learning_rate, bool) or not isinstance(model.learning_rate, Real)
                or not np.isfinite(model.learning_rate) or model.learning_rate <= 0):
            raise ValueError("learning_rate must be positive and finite")


def _windows(model, windows):
    x = np.asarray(windows, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (model.n_channels, model.n_samples):
        raise ValueError(f"EEG windows must have shape (n_windows, {model.n_channels}, {model.n_samples})")
    if not np.isfinite(x).all():
        raise ValueError("EEG windows must contain finite values in volts")
    return x


def _training_data(model, windows, labels):
    _check_parameters(model)
    x = _windows(model, windows)
    y = np.asarray(labels)
    if y.ndim != 1 or len(y) != len(x) or y.dtype.kind not in "iu" or set(y.tolist()) != set(CLASS_IDS):
        raise ValueError("Training labels must contain all four integer classes: 0=LEFT, 1=RIGHT, 2=FEET, 3=TONGUE")
    if np.any(np.sum(np.var(x, axis=2), axis=1) <= 0):
        raise ValueError("Training windows must contain nonconstant EEG")
    return x, y.astype(np.int64, copy=False)


def _mark_fitted(model):
    model.classes_ = np.array(CLASS_IDS, dtype=np.int64)
    model.n_channels_ = model.n_channels
    model.n_samples_ = model.n_samples
    model.sfreq_ = model.sfreq


def _require_fitted(model):
    if not hasattr(model, "classes_"):
        raise NotFittedError("Call fit with calibration EEG before predicting")
    if not np.array_equal(model.classes_, CLASS_IDS):
        raise ValueError("This is not a four-class model; retrain with LEFT, RIGHT, FEET and TONGUE data")


class RiemannianClassifier(ClassifierMixin, BaseEstimator):
    """OAS 协方差 → Riemannian 切空间 → 逻辑回归。"""

    algorithm = "riemannian"

    def __init__(self, n_channels, n_samples, sfreq, *, seed=42):
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.sfreq = sfreq
        self.seed = seed

    def fit(self, windows, labels):
        x, y = _training_data(self, windows, labels)
        try:
            from pyriemann.estimation import Covariances
            from pyriemann.tangentspace import TangentSpace
        except ImportError as exc:
            raise ImportError("Riemannian training requires pyriemann; install requirements.lock.txt") from exc
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        # OAS 收缩保证常见的平均参考/相关通道不会造成奇异协方差。
        # 固定 V→μV 换算只改善数值尺度，不从训练或测试数据估计参数。
        pipeline = Pipeline([
            ("covariances", Covariances(estimator="oas")),
            ("tangent_space", TangentSpace(metric="riemann", tsupdate=False)),
            ("logistic_regression", LogisticRegression(
                C=1.0, max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=self.seed)),
        ])
        pipeline.fit(x * 1e6, y)
        self.pipeline_ = pipeline
        _mark_fitted(self)
        return self

    def predict_proba(self, windows):
        _require_fitted(self)
        x = _windows(self, windows)
        if len(x) == 0:
            return np.empty((0, N_CLASSES), dtype=float)
        if np.any(np.sum(np.var(x, axis=2), axis=1) <= 0):
            raise ValueError("Prediction windows must contain nonconstant EEG")
        return self.pipeline_.predict_proba(x * 1e6)

    def predict(self, windows):
        return self.classes_[self.predict_proba(windows).argmax(axis=1)]


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("EEGNet requires PyTorch; install requirements.lock.txt") from exc
    return torch


class EEGNetClassifier(ClassifierMixin, BaseEstimator):
    """CPU EEGNet-8,2；固定轮数训练，保留仅由训练集估计的通道均值/标准差。

    第一层时间核约 0.5 秒；第二层可分离时间核在 4 倍降采样后覆盖
    约 0.5 秒。两次平均池化为 4 和 8，普通 Dropout 为 0.5。
    状态字典以 NumPy 数组保存，因此 joblib 不依赖局部类或 GPU 设备。
    """

    algorithm = "eegnet"

    def __init__(self, n_channels, n_samples, sfreq, *, seed=42, epochs=100,
                 batch_size=32, learning_rate=1e-3):
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.sfreq = sfreq
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def _architecture(self):
        return {"f1": 8, "depth_multiplier": 2, "f2": 16,
                "temporal_kernel": max(1, round(self.sfreq / 2)),
                "separable_kernel": max(1, round(self.sfreq / 8)),
                "pool1": 4, "pool2": 8, "dropout": 0.5, "n_classes": N_CLASSES}

    def _new_network(self):
        torch = _torch()
        nn = torch.nn
        spec = self._architecture()
        k1, k2 = spec["temporal_kernel"], spec["separable_kernel"]
        # 显式不对称 padding 保持时间长度，也支持偶数核。
        return nn.Sequential(OrderedDict([
            ("temporal_pad", nn.ZeroPad2d(((k1 - 1) // 2, k1 // 2, 0, 0))),
            ("temporal", nn.Conv2d(1, 8, (1, k1), bias=False)),
            ("temporal_bn", nn.BatchNorm2d(8, eps=1e-3, momentum=0.01)),
            ("spatial", nn.Conv2d(8, 16, (self.n_channels, 1), groups=8, bias=False)),
            ("spatial_bn", nn.BatchNorm2d(16, eps=1e-3, momentum=0.01)),
            ("spatial_elu", nn.ELU()),
            ("pool1", nn.AvgPool2d((1, 4))),
            ("dropout1", nn.Dropout(0.5)),
            ("separable_pad", nn.ZeroPad2d(((k2 - 1) // 2, k2 // 2, 0, 0))),
            ("separable_depthwise", nn.Conv2d(16, 16, (1, k2), groups=16, bias=False)),
            ("separable_pointwise", nn.Conv2d(16, 16, (1, 1), bias=False)),
            ("separable_bn", nn.BatchNorm2d(16, eps=1e-3, momentum=0.01)),
            ("separable_elu", nn.ELU()),
            ("pool2", nn.AvgPool2d((1, 8))),
            ("dropout2", nn.Dropout(0.5)),
            ("flatten", nn.Flatten()),
            ("classifier", nn.Linear(16 * (self.n_samples // 4 // 8), N_CLASSES)),
        ]))

    @staticmethod
    def _constrain_weights(network, torch):
        with torch.no_grad():
            for weight, limit in ((network.spatial.weight, 1.0), (network.classifier.weight, 0.25)):
                norms = torch.linalg.vector_norm(weight, dim=tuple(range(1, weight.ndim)), keepdim=True)
                weight.mul_(torch.clamp(limit / norms.clamp_min(1e-12), max=1.0))

    def _normalized(self, x):
        return np.ascontiguousarray((x - self.mean_) / self.scale_, dtype=np.float32)

    @staticmethod
    def _forward(network, samples, priors=None):
        return network(samples)

    @staticmethod
    def _initialize_network(network, torch):
        for layer in network.modules():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)

    def _prior_probabilities(self, windows):
        return None

    def fit(self, windows, labels):
        x, y = _training_data(self, windows, labels)
        return self._fit_neural(x, y)

    def _fit_neural(self, x, y, priors=None):
        torch = _torch()
        # 每个 fit 重建网络、优化器和归一化参数；外部验证窗口不参与任何统计。
        self.mean_ = x.mean(axis=(0, 2), keepdims=True)
        self.scale_ = np.maximum(x.std(axis=(0, 2), keepdims=True), 1e-12)
        samples = torch.from_numpy(self._normalized(x)[:, None, :, :])
        targets = torch.from_numpy(y)
        prior_tensor = (torch.from_numpy(np.ascontiguousarray(priors, dtype=np.float32))
                        if priors is not None else None)
        class_counts = np.bincount(y, minlength=N_CLASSES)
        class_weights = torch.tensor(len(y) / (N_CLASSES * class_counts), dtype=torch.float32)
        # 保存/恢复外部随机数状态，避免训练污染其他程序的随机序列。
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            network = self._new_network().cpu()
            self._initialize_network(network, torch)
            optimizer = torch.optim.Adam(network.parameters(), lr=self.learning_rate)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
            self.loss_history_ = []
            network.train()
            for _ in range(self.epochs):
                order = torch.randperm(len(samples))
                total_loss = 0.0
                for indices in order.split(self.batch_size):
                    optimizer.zero_grad(set_to_none=True)
                    logits = self._forward(network, samples[indices],
                                           prior_tensor[indices] if prior_tensor is not None else None)
                    loss = criterion(logits, targets[indices])
                    if not torch.isfinite(loss):
                        raise ValueError("EEGNet training loss became nonfinite; inspect calibration EEG")
                    loss.backward()
                    optimizer.step()
                    self._constrain_weights(network, torch)
                    total_loss += float(loss.detach()) * len(indices)
                self.loss_history_.append(total_loss / len(samples))
        network.eval()
        self._network = network
        self.state_dict_ = {name: value.detach().cpu().numpy().copy()
                            for name, value in network.state_dict().items()}
        self.architecture_ = self._architecture()
        self.n_epochs_ = self.epochs
        _mark_fitted(self)
        return self

    def _get_network(self):
        if getattr(self, "_network", None) is None:
            torch = _torch()
            with torch.random.fork_rng(devices=[]):
                network = self._new_network().cpu()
            network.load_state_dict({name: torch.from_numpy(np.asarray(value))
                                     for name, value in self.state_dict_.items()}, strict=True)
            network.eval()
            self._network = network
        return self._network

    def predict_proba(self, windows):
        _require_fitted(self)
        x = _windows(self, windows)
        if len(x) == 0:
            return np.empty((0, N_CLASSES), dtype=float)
        # Hybrid 必须先执行前级机器学习，再进入后级网络。
        priors = self._prior_probabilities(x)
        torch = _torch()
        network = self._get_network()
        network.eval()
        result = []
        with torch.inference_mode():
            for start in range(0, len(x), self.batch_size):
                batch = torch.from_numpy(self._normalized(x[start:start + self.batch_size])[:, None, :, :])
                prior_batch = (torch.from_numpy(np.ascontiguousarray(
                    priors[start:start + self.batch_size], dtype=np.float32)) if priors is not None else None)
                result.append(torch.softmax(self._forward(network, batch, prior_batch), dim=1).cpu().numpy())
        return np.concatenate(result).astype(np.float64)

    def predict(self, windows):
        return self.classes_[self.predict_proba(windows).argmax(axis=1)]

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_network", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._network = None


class HybridClassifier(EEGNetClassifier):
    """先 Riemannian/TS/LR，再由 EEGNet 利用原窗口和前级概率修正分类。

    后级只用按完整 run 交叉拟合得到的前级概率训练。最终前级再拟合全部
    训练 run，部署时顺序执行前级和后级；E 会话不参与任何拟合。
    """

    algorithm = "hybrid"

    def _architecture(self):
        return {**super()._architecture(), "fusion_hidden": 64,
                "prior_features": N_CLASSES, "fusion_dropout": 0.25,
                "fusion": "log_prior_plus_eegnet_residual_v1"}

    def _new_network(self):
        nn = _torch().nn
        encoder = super()._new_network()
        encoder = nn.Sequential(OrderedDict(
            (name, layer) for name, layer in encoder.named_children() if name != "classifier"))
        features = 16 * (self.n_samples // 4 // 8)
        return nn.ModuleDict({
            "encoder": encoder,
            "fusion": nn.Sequential(nn.Linear(features + N_CLASSES, 64), nn.ELU(),
                                    nn.Dropout(0.25), nn.Linear(64, N_CLASSES)),
        })

    @staticmethod
    def _initialize_network(network, torch):
        EEGNetClassifier._initialize_network(network, torch)
        # 初始修正为零，网络从前级概率开始学习残差，不随机覆盖前级判断。
        torch.nn.init.zeros_(network["fusion"][-1].weight)
        torch.nn.init.zeros_(network["fusion"][-1].bias)

    @staticmethod
    def _forward(network, samples, priors=None):
        torch = _torch()
        if priors is None or priors.ndim != 2 or priors.shape != (len(samples), N_CLASSES):
            raise ValueError("Hybrid EEGNet requires four probabilities from its first stage")
        features = network["encoder"](samples)
        residual = network["fusion"](torch.cat((features, priors), dim=1))
        return torch.log(priors.clamp_min(1e-6)) + residual

    @staticmethod
    def _constrain_weights(network, torch):
        with torch.no_grad():
            for weight, limit in ((network["encoder"].spatial.weight, 1.0),
                                  (network["fusion"][-1].weight, 0.25)):
                norms = torch.linalg.vector_norm(weight, dim=tuple(range(1, weight.ndim)), keepdim=True)
                weight.mul_(torch.clamp(limit / norms.clamp_min(1e-12), max=1.0))

    def fit(self, windows, labels, groups=None):
        x, y = _training_data(self, windows, labels)
        group_ids = np.asarray(groups)
        if (group_ids.ndim != 1 or len(group_ids) != len(x)
                or group_ids.dtype.kind not in "iu" or np.any(group_ids < 0)):
            raise ValueError("Hybrid fit requires integer acquisition-run groups for every window")
        unique_groups = np.unique(group_ids)
        if len(unique_groups) < 2:
            raise ValueError("Hybrid fit needs at least two runs for out-of-run first-stage predictions")
        for held in unique_groups:
            if set(y[group_ids != held].tolist()) != set(CLASS_IDS):
                raise ValueError("Every hybrid first-stage training fold must contain all four classes")
        oof = np.empty((len(x), N_CLASSES), dtype=float)
        covered = np.zeros(len(x), dtype=bool)
        folds = []
        for held in unique_groups:
            train = group_ids != held
            validation = ~train
            stage = RiemannianClassifier(self.n_channels, self.n_samples, self.sfreq, seed=self.seed)
            stage.fit(x[train], y[train])
            oof[validation] = stage.predict_proba(x[validation])
            covered[validation] = True
            folds.append({"held_out_run": int(held),
                          "training_runs": [int(g) for g in unique_groups if g != held],
                          "training_windows": int(train.sum()), "held_out_windows": int(validation.sum())})
        if (not covered.all() or not np.isfinite(oof).all() or np.any(oof < 0)
                or np.any(oof > 1) or not np.allclose(oof.sum(axis=1), 1)):
            raise ValueError("Hybrid first-stage out-of-run probabilities are incomplete or invalid")
        self.stage1_ = RiemannianClassifier(self.n_channels, self.n_samples, self.sfreq, seed=self.seed)
        self.stage1_.fit(x, y)
        self.stacking_report_ = {
            "method": "leave-one-acquisition-run-out first-stage predictions; EEGNet trained on OOF priors only",
            "groups": [int(g) for g in unique_groups], "folds": folds,
            "windows": len(x), "oof_probabilities_sha256": sha256(oof.astype("<f8").tobytes()).hexdigest(),
            "first_stage_final_fit": "all training runs only", "independent_test_used": False,
        }
        return self._fit_neural(x, y, priors=oof)

    def _prior_probabilities(self, windows):
        prior = np.asarray(self.stage1_.predict_proba(windows), dtype=float)
        if (prior.shape != (len(windows), N_CLASSES) or not np.isfinite(prior).all()
                or np.any(prior < 0) or np.any(prior > 1) or not np.allclose(prior.sum(axis=1), 1)):
            raise ValueError("Hybrid first-stage probabilities violate the four-class contract")
        return prior


def build_model(algorithm, n_channels, n_samples, sfreq, *, seed=42, eegnet_epochs=100,
                eegnet_batch_size=32, eegnet_learning_rate=1e-3):
    if algorithm == "riemannian":
        model = RiemannianClassifier(n_channels, n_samples, sfreq, seed=seed)
    elif algorithm in {"eegnet", "hybrid"}:
        classifier = HybridClassifier if algorithm == "hybrid" else EEGNetClassifier
        model = classifier(n_channels, n_samples, sfreq, seed=seed,
                                 epochs=eegnet_epochs, batch_size=eegnet_batch_size,
                                 learning_rate=eegnet_learning_rate)
    else:
        raise ValueError(f"Unsupported algorithm {algorithm!r}; choose from {ALGORITHMS}")
    _check_parameters(model)
    return model


def validate_fitted_model(model, algorithm, n_channels, n_samples, sfreq):
    """保存/加载时验证路线、学习状态与冻结的输入契约。"""
    expected = {"riemannian": RiemannianClassifier, "eegnet": EEGNetClassifier,
                "hybrid": HybridClassifier}.get(algorithm)
    if expected is None or not isinstance(model, expected) or model.algorithm != algorithm:
        raise ValueError("Model type differs from the frozen algorithm")
    _require_fitted(model)
    if not np.array_equal(model.classes_, CLASS_IDS):
        raise ValueError("Four-class model required: probability columns must be [LEFT, RIGHT, FEET, TONGUE]; retrain old binary models")
    for key, value in (("n_channels", n_channels), ("n_samples", n_samples), ("sfreq", sfreq)):
        if getattr(model, key, None) != value or getattr(model, f"{key}_", None) != value:
            raise ValueError(f"Model {key} differs from the frozen input contract")
    if algorithm == "riemannian":
        steps = getattr(getattr(model, "pipeline_", None), "named_steps", {})
        tangent = steps.get("tangent_space")
        covariance = steps.get("covariances")
        lr = steps.get("logistic_regression")
        reference = np.asarray(getattr(tangent, "reference_", []))
        features = n_channels * (n_channels + 1) // 2
        if (getattr(covariance, "estimator", None) != "oas"
                or getattr(tangent, "metric", None) != "riemann"
                or getattr(tangent, "tsupdate", None) is not False
                or reference.shape != (n_channels, n_channels) or not np.isfinite(reference).all()
                or not np.allclose(reference, reference.T) or np.min(np.linalg.eigvalsh(reference)) <= 0
                or getattr(lr, "n_features_in_", None) != features
                or not np.array_equal(getattr(lr, "classes_", []), CLASS_IDS)):
            raise ValueError("Invalid fitted Riemannian / Tangent Space / Logistic Regression state")
        if not np.isfinite(lr.coef_).all() or not np.isfinite(lr.intercept_).all():
            raise ValueError("Logistic Regression contains nonfinite parameters")
    else:
        if algorithm == "hybrid":
            validate_fitted_model(getattr(model, "stage1_", None), "riemannian", n_channels, n_samples, sfreq)
            report = getattr(model, "stacking_report_", {})
            groups = report.get("groups", [])
            folds = report.get("folds", [])
            if (len(groups) < 2 or len(set(groups)) != len(groups) or len(folds) != len(groups)
                    or {fold.get("held_out_run") for fold in folds} != set(groups)
                    or report.get("independent_test_used") is not False
                    or any(set(fold.get("training_runs", [])) != set(groups) - {fold.get("held_out_run")}
                           for fold in folds)):
                raise ValueError("Hybrid first-stage cross-fitting provenance is invalid")
        for key in ("mean_", "scale_"):
            value = np.asarray(getattr(model, key, []))
            if value.shape != (1, n_channels, 1) or not np.isfinite(value).all():
                raise ValueError("EEGNet normalization differs from the input channel contract")
        if (np.any(model.scale_ <= 0) or getattr(model, "architecture_", None) != model._architecture()
                or getattr(model, "n_epochs_", None) != model.epochs):
            raise ValueError("EEGNet fitted parameters differ from its training configuration")
        state = getattr(model, "state_dict_", {})
        if not state or any(not np.isfinite(value).all() for value in state.values()):
            raise ValueError("EEGNet weights are missing or nonfinite")
        torch = _torch()
        with torch.random.fork_rng(devices=[]):
            expected_state = model._new_network().state_dict()
        if set(state) != set(expected_state) or any(
                np.asarray(state[key]).shape != tuple(value.shape) for key, value in expected_state.items()):
            raise ValueError("EEGNet weights differ from the frozen architecture")


__all__ = ["RiemannianClassifier", "EEGNetClassifier", "HybridClassifier", "build_model", "validate_fitted_model"]
