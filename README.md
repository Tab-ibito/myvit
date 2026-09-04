# myvit

基于 PyTorch 手写 ViT-Tiny 的学习项目，包含 Imagenette 分类训练、checkpoint 和指标可视化。

## 环境与安装

使用 **Python 3.11.9（64 位）**。CPU 可运行测试和合成数据调试，真实图片训练支持 NVIDIA CUDA。

在项目根目录执行以下 PowerShell 命令，先确认版本输出为 `3.11.9`：

```powershell
py -3.11 --version
py -3.11 -m venv .venv
$projectRoot = (Get-Location).Path
$projectPython = (Resolve-Path '.\.venv\Scripts\python.exe').Path
```

需要 GPU 时，先按 [PyTorch 官方指引](https://pytorch.org/get-started/locally/) 在此环境中
安装匹配驱动的 `torch` / `torchvision`，再安装项目：

```powershell
& $projectPython -m pip install -e ".[test,visualization]" fonttools
& $projectPython -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
& $projectPython -m pytest -q
```

原设备可直接把 `$projectPython` 设为
`C:\Users\Fwk\AppData\Local\Programs\Python\Python311\python.exe`，无需重建环境。
其他设备需重新选择 IDE 解释器，不要复制旧 `.venv`。

Linux 可用以下方式安装；后续将 `& $projectPython` 换成 `"$projectPython"`：

```bash
python3.11 -m venv .venv  # 先确认解释器为 3.11.9
projectRoot="$PWD"
projectPython="$projectRoot/.venv/bin/python"
# 需要 CUDA 时同样先安装对应的 PyTorch。
"$projectPython" -m pip install -e ".[test,visualization]" fonttools
```

当前 `src/myvit/data/` 被 Git 忽略。克隆后若缺少其中的 `__init__.py` 和 `imagenette.py`，需从原设备补齐这两个源码文件。

## 快速调试

无需下载数据，在项目根目录运行：

```powershell
& $projectPython -m myvit.debug_training --epochs 15 --output-dir outputs/debug-new
```

## Imagenette 训练

下载 [Imagenette 320 px](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz)，
解压到项目根目录的 `data/`，确保存在
`data/imagenette2-320/train/` 和 `val/`。
数据共 10 类，训练图片 9,469 张、验证图片 3,925 张。

```powershell
& $projectPython -m myvit.inspect_data data/imagenette2-320
```

模型配置在 [tiny_training.py](src/myvit/tiny_training.py)：224 输入、16 Patch、
192 维、12 层、3 个注意力头。**当前文件仍为 `num_classes=1000`，
运行 Imagenette 前需改成 `10` 并保存。**

训练入口是 `debug_training.py --tiny`，不是直接运行 `tiny_training.py`。
由于当前数据路径相对工作目录解析，必须从 `src/myvit` 启动：

```powershell
Push-Location (Join-Path $projectRoot 'src/myvit')
try {
    & $projectPython -m myvit.debug_training --tiny --epochs 50 --batch-size 16 --learning-rate 1e-4 --output-dir (Join-Path $projectRoot 'outputs/imagenette-new')
} finally {
    Pop-Location
}
```

Linux 对应命令：

```bash
(
    cd "$projectRoot/src/myvit" || exit 1
    "$projectPython" -m myvit.debug_training --tiny --epochs 50 --batch-size 16 --learning-rate 1e-4 --output-dir "$projectRoot/outputs/imagenette-new"
)
```

当前使用 AdamW + `StepLR(step_size=epochs)`：计划内训练保持初始学习率，
最后一轮结束后才衰减。显存不足时降低 batch size。

## 查看结果与恢复训练

回到项目根目录执行：

```powershell
& $projectPython -m myvit.inspect_checkpoint outputs/imagenette-new/best.pt
& $projectPython -m myvit.visualize_training outputs/imagenette-new
```

- `metrics.jsonl`：每轮的 Loss、Top-1/Top-5、学习率和梯度范数。
- `best.pt`：验证 Top-1 最佳模型；打平时保留首次达到的版本。
- `last.pt`：最新训练状态；原训练命令增加 `--resume <last.pt路径>` 可恢复。
- `training_curves.png`：可视化程序生成的曲线图；加 `--show` 可弹窗。

每次新实验使用新输出目录，避免覆盖旧记录。恢复时保持原模型、调度器和总
`--epochs`，保留原日志；`--tiny` 仍需从 `src/myvit` 启动。
更换分类头后不能直接完整恢复旧 checkpoint。

模型实现见 [models/vit.py](src/myvit/models/vit.py)，训练逻辑见
[engine](src/myvit/engine)。详细说明见 [指标与 checkpoint](docs/training-metrics-and-checkpoints.md)
和 [实验记录](docs/experiments.md)（其中余弦衰减描述属于旧实验）。
