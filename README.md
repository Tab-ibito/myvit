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

训练指标和 checkpoint 的详细解释见
[`docs/training-metrics-and-checkpoints.md`](docs/training-metrics-and-checkpoints.md)。

