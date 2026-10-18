"""向后兼容的 GUI 启动器；Qt 实现位于 ``eeg_mi.gui``。"""

from eeg_mi.gui import *  # noqa: F401,F403
from eeg_mi.gui import launch


if __name__ == "__main__":
    launch()
