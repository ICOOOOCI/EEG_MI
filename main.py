"""向后兼容的命令行启动器。

真实业务实现位于 :mod:`eeg_mi.cli`；保留本文件是为了兼容既有脚本、模型
文件以及用户原来的 ``python main.py`` 使用方式。
"""

from eeg_mi.cli import *  # noqa: F401,F403 - 兼容旧代码的公开 API
from eeg_mi import cli as _cli
import sys

# 兼容历史 joblib 模型中记录的 ``main``/``mi_models`` 类路径。
sys.modules.setdefault("mi_models", sys.modules[__name__])
__version__ = _cli.__version__
for _name in dir(_cli):
    if not _name.startswith("__"):
        globals().setdefault(_name, getattr(_cli, _name))

# 旧代码可能从 main 模块读取以下私有辅助函数；显式保留这些兼容别名。
_lsl_description = _cli._lsl_description
_lsl_header = _cli._lsl_header


if __name__ == "__main__":
    raise SystemExit(_cli.main())
