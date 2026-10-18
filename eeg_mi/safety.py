"""实时采集安全状态与阶段标记协议。"""

from numbers import Real
import math
import threading

# ========================================================================
# 16. 采集线程的实时状态与标记协议
# ========================================================================
class MarkerProtocol:

    # 验证组与试次阶段的顺序，拒绝缺失、重复、乱序和复用的试次 ID。
    def __init__(self):
        self.run = False
        self.trial = None
        self.phase = "WAITING"
        self.seen = set()
        self.last_stamp = None

    def observe(self, kind, trial_id, timestamp):
        if (isinstance(timestamp, bool) or not isinstance(timestamp, Real) or not math.isfinite(timestamp)
                or (self.last_stamp is not None and timestamp < self.last_stamp)):
            raise ValueError("Marker timestamps are nonfinite or out of order")
        if self.phase == "ENDED":
            raise ValueError("Marker arrived after session end")
        if kind == "DISCONNECT":
            self.phase, self.trial = "ENDED", None
        elif kind == "RUN_START":
            if self.trial is not None or self.phase == "RUN_START":
                raise ValueError("RUN_START interrupts a trial or repeats an empty run")
            self.run, self.phase = True, "RUN_START"
        elif kind == "TRIAL_START":
            if not self.run or self.trial is not None or not trial_id or trial_id in self.seen:
                raise ValueError("Missing phase or duplicate trial ID before TRIAL_START")
            self.seen.add(trial_id)
            self.trial, self.phase = trial_id, kind
        elif kind == "MI_START":
            if self.phase != "TRIAL_START" or trial_id != self.trial:
                raise ValueError("MI_START requires its unique matching TRIAL_START")
            self.phase = kind
        elif kind == "MI_END":
            if self.phase != "MI_START" or trial_id != self.trial:
                raise ValueError("MI_END requires its unique matching MI_START")
            self.trial, self.phase = None, kind
        elif kind == "SESSION_END":
            if self.trial is not None or self.phase != "MI_END":
                raise ValueError("SESSION_END requires complete trials and a nonempty final run")
            self.phase = "ENDED"
        else:
            raise ValueError("Unrecognized neutral phase")
        self.last_stamp = timestamp


class LiveSafety:
    # 保存采集线程直接更新的状态；解码及发布阶段在锁内检查其有效性。
    def __init__(self, config, clock, stop_event):
        self.config, self.clock, self.stop = config, clock, stop_event
        self.lock = threading.RLock()
        self.protocol = MarkerProtocol()
        self.latest_eeg = None
        self.cue_timestamp = None
        self.failure = None

    def eeg(self, last_timestamp):
        with self.lock:
            self.latest_eeg = float(last_timestamp)

    def marker(self, kind, trial_id, timestamp):
        with self.lock:
            try:
                self.protocol.observe(kind, trial_id, timestamp)
            except ValueError:
                self.failure = "MARKER_PROTOCOL_ERROR"
                raise
            if kind == "MI_START":
                self.cue_timestamp = timestamp

    def fail(self, reason):
        with self.lock:
            self.failure = reason

    def reason(self, trial_id, window_timestamp):
        with self.lock:
            now = self.clock()
            if self.failure:
                return self.failure
            if self.stop.is_set():
                return "STOPPED"
            if self.protocol.phase == "ENDED":
                return "SESSION_ENDED"
            if self.latest_eeg is None or now - self.latest_eeg > self.config.max_age_seconds:
                return "SIGNAL_LOST"
            if now - window_timestamp > self.config.max_age_seconds:
                return "STALE_RESULT"
            if window_timestamp - now > self.config.timestamp_tolerance_seconds:
                return "CLOCK_MISMATCH"
            if self.protocol.phase != "MI_START" or self.protocol.trial != trial_id:
                return "OUTSIDE_MI_PHASE"
            if self.cue_timestamp is None or now >= self.cue_timestamp + self.config.imagery_seconds:
                return "MI_DEADLINE_EXCEEDED"
            return None


SCHEMA = "mi-results-v2"




__all__ = ["MarkerProtocol", "LiveSafety"]
