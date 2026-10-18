"""稳定的领域常量与单位契约。

所有算法、采集器和界面都从这里读取类别顺序，避免在不同模块重复硬编码。
"""

ALGORITHMS = ("hybrid", "riemannian", "eegnet")
CLASS_NAMES = ("LEFT", "RIGHT", "FEET", "TONGUE")
N_CLASSES = len(CLASS_NAMES)
CLASS_IDS = tuple(range(N_CLASSES))
TRUTH_IDS = tuple(range(1, N_CLASSES + 1))
CLASS_MAPPING = {str(index): name for index, name in enumerate(CLASS_NAMES)}
TRUTH_MAPPING = {str(index + 1): name for index, name in enumerate(CLASS_NAMES)}
CLASS_CUES = {1: "左手 ←", 2: "右手 →", 3: "脚（双脚）", 4: "舌头"}

# 外部 LSL 单位统一换算到核心管线使用的伏特。
UNIT_SCALE = {"V": 1.0, "mV": 1e-3, "uV": 1e-6, "µV": 1e-6, "μV": 1e-6}

__all__ = [
    "ALGORITHMS", "CLASS_NAMES", "N_CLASSES", "CLASS_IDS", "TRUTH_IDS",
    "CLASS_MAPPING", "TRUTH_MAPPING", "CLASS_CUES", "UNIT_SCALE",
]
