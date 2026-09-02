# myvit

从基础 PyTorch 模块搭建并训练 Vision Transformer 的学习项目。

当前已经包含：

- ViT-Tiny/16 模型结构；
- 分类训练与验证循环；
- Loss、Top-1、Top-5、学习率和梯度范数统计；
- `last.pt` / `best.pt` checkpoint 保存、检查与恢复；
- 不依赖外部数据集的快速训练闭环；
- 模型、指标、训练和 checkpoint 单元测试。

## 项目 Python

项目统一使用本地 Python 路径。

VS Code 已在 `.vscode/settings.json` 中配置该解释器。

## 运行测试

在 PowerShell 中执行：

```powershell
& $projectPython -m pytest -q
```

## 运行调试训练

```powershell
& $projectPython -m myvit.debug_training --epochs 15 --output-dir outputs/debug
```

查看 checkpoint：

```powershell
& $projectPython -m myvit.inspect_checkpoint outputs/debug/best.pt
& $projectPython -m myvit.inspect_checkpoint outputs/debug/last.pt
```

## 可视化训练曲线与 checkpoint

首次使用时安装绘图依赖：

```powershell
& $projectPython -m pip install -e ".[visualization]"
```

读取一个实验目录，生成 Loss、Top-1/Top-5、学习率和梯度范数曲线，同时在
终端打印 `best.pt`、`last.pt` 等 checkpoint 的摘要：

```powershell
& $projectPython -m myvit.visualize_training outputs/debug
```

默认图片保存为 `outputs/debug/training_curves.png`。如果需要绘图窗口，可以增加
`--show`；也可以使用 `--output` 指定其他 PNG 路径。

## Imagenette 小规模真实图片测试

项目使用 fastai 官方的 `imagenette2-320`，解压路径为：

```text
F:\myvit\data\imagenette2-320
```

数据规模：

```text
训练集：9,469 张
验证集：3,925 张
类别数：10
```

检查数据目录和第一个预处理 batch：

```powershell
$projectPython = 'C:\Users\Fwk\AppData\Local\Programs\Python\Python311\python.exe'
$env:PYTHONPATH = 'F:\myvit\src'
& $projectPython -m myvit.inspect_data `
  F:\myvit\data\imagenette2-320 `
  --batch-size 16 `
  --image-size 224
```

预期图片 batch 形状为 `[16, 3, 224, 224]`。数据接入代码位于
`src/myvit/data/imagenette.py`。

训练指标和 checkpoint 的详细解释见
[`docs/training-metrics-and-checkpoints.md`](docs/training-metrics-and-checkpoints.md)。
