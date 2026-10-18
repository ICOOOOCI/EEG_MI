# EEG-MI

这是一个四分类运动想象 EEG 工程。核心处理统一使用伏特（V），EEG 数组统一为“通道 × 采样点”，类别顺序固定为 `LEFT / RIGHT / FEET / TONGUE`。

## 目录职责

```text
eeg_mi/
├── constants.py   类别、单位和算法契约
├── config.py      不可变采集/解码配置
├── domain.py      EEGChunk、Marker、Recording 等领域对象及 NPZ 存储
├── models.py      Riemannian、EEGNet、Hybrid 可替换分类器
├── processing.py  因果滤波、环形缓存、窗口推断和判决状态机
├── transport.py   回放源、异步原始记录器和原子 JSON 写入
├── lsl.py         LSL 元数据与实时 EEG/Marker 适配器
├── safety.py      MarkerProtocol、LiveSafety 和结果 schema
├── quality.py     设备预检、通道质量和时序审计
├── cli.py         命令行业务编排
└── gui.py         PyQt6/OpenBCI 采集界面
```

`main.py` 和 `collector_gui.py` 仅保留兼容启动职责，原有命令不需要改写。推荐安装项目依赖后使用：

```bash
python main.py --help
python -m eeg_mi.cli --help
python collector_gui.py
```

GUI 也可通过 `eeg-mi-gui` 启动。`requirements.lock.txt` 是当前 macOS arm64 环境的可复现实验依赖快照，`requirements-gui.txt` 仅补充 Qt 组件。

## 可插拔边界

算法只需实现 `predict_proba(windows)` 即可接入 `Runtime`；采集源只需产出 `EEGChunk | Marker` 即可复用同一解码流程。离线 `ReplaySource`、真实 `LiveSource` 和 GUI 记录器都通过这些领域接口通信，不直接共享 UI 状态或文件细节。
