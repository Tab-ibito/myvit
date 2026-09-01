# 如何阅读训练指标和 checkpoint

## 一行训练日志

调试脚本每个 epoch 会输出类似：

```text
epoch=004 train_loss=0.3013 train_top1=100.00% val_loss=0.2594
val_top1=100.00% lr=0.002250 grad_norm=2.457 best_val_top1=100.00%
```

不要只看某一个数字，重点观察多轮训练形成的趋势。

## Loss

`train_loss` 是训练集平均交叉熵，`val_loss` 是验证集平均交叉熵。通常越低越好。

在模型接近随机猜测、预测概率近似均匀时，交叉熵约为 `ln(类别数)`：

- 4 类调试任务约为 `ln(4) = 1.386`；
- ImageNet-1K 约为 `ln(1000) = 6.908`。

常见现象：

- train/val loss 一起下降：训练正常；
- train loss 下降、val loss 上升：开始过拟合；
- loss 几乎不动：检查学习率、数据标签、梯度和模型输出；
- loss 突然变得很大或出现 NaN：检查学习率、AMP、输入归一化和梯度爆炸。

验证 loss 有时会在准确率下降前提前上升，因为模型可能仍然分类正确，
但对错误样本变得越来越自信。

## Top-1 与 Top-5

- Top-1：模型概率最高的类别等于真实标签；
- Top-5：概率最高的五个类别中包含真实标签。

ImageNet 通常同时报告 Top-1 和 Top-5。项目调试数据只有 4 类，Top-5 会自动退化成
Top-4，所以它会一直是 100%，没有分析价值；此时主要看 Top-1。

常见组合：

- train/val Top-1 一起上升：模型正在学到可泛化规律；
- train Top-1 很高、val Top-1 明显低：过拟合或训练/验证预处理不一致；
- 两者都很低：欠拟合、训练不足或实现/数据存在问题；
- Top-5 高而 Top-1 低：模型大致知道候选类别，但区分相似类别的能力不足。

## Learning Rate

`lr` 是本轮实际用于参数更新的学习率。调试训练使用余弦衰减，所以它会从
`0.003` 逐渐下降到接近 0。

观察重点：

- loss 大幅震荡甚至发散：学习率可能过大；
- loss 下降极慢且梯度正常：学习率可能过小；
- 从 checkpoint 恢复后 lr 突然跳回初始值：scheduler 状态没有正确恢复；
- 正式 ImageNet 训练初期不稳定：通常需要 warmup，而不应直接使用峰值学习率。

## Gradient Norm

`grad_norm` 是裁剪前所有参数梯度合并后的 L2 范数。它没有统一的“合格数值”，
主要用于观察异常和相对变化。

- 偶尔波动：通常正常；
- 突然增大几个数量级：可能出现梯度爆炸或异常 batch；
- 长期接近 0：可能梯度消失、参数被冻结或学习信号中断；
- NaN/Inf：应立即停止并定位数值问题。

当前训练循环使用 `max_gradient_norm=1.0` 时，日志仍记录裁剪前的范数，
所以日志中看到大于 1 是正常的；真正参与更新的梯度已被裁剪。

## best_val_top1

它记录从训练开始到当前为止最高的验证 Top-1，而不是当前 epoch 的准确率。
该数值用于决定是否更新 `best.pt`。

项目使用严格的 `>` 判断：如果多个 epoch 都达到相同最高准确率，`best.pt` 保留
第一次达到该准确率的版本。如果准确率已经饱和，可以改用更低的 `val_loss` 作为
第二选择条件。

## last.pt 与 best.pt

### last.pt

每个 epoch 都会覆盖，表示最近完成的一轮。用途是：

- 训练意外中断后继续；
- 保留最新优化器和学习率调度状态；
- 排查最后几轮的异常。

### best.pt

仅在验证 Top-1 刷新纪录时覆盖。用途是：

- 最终验证和推理；
- 导出部署模型；
- 作为后续微调的起点。

训练跑完后通常使用 `best.pt` 做最终报告，而不是默认使用 `last.pt`。

## checkpoint 保存了什么

当前 checkpoint 包含：

- `model_state`：模型参数；
- `optimizer_state`：AdamW 动量等内部状态；
- `scheduler_state`：学习率调度进度；
- `scaler_state`：AMP 动态缩放状态，未使用 AMP 时为空；
- `epoch`：最近完成的 epoch；
- `metrics`：该 epoch 的训练和验证指标；
- `best_metric`：历史最佳验证 Top-1；
- `extra`：随机种子等实验信息。

只恢复 `model_state` 适合推理；继续训练时应同时恢复 optimizer、scheduler 和 scaler。
恢复后下一轮从 `checkpoint_epoch + 1` 开始。

## metrics.jsonl

该文件每行保存一个 epoch 的 JSON 指标，适合：

- 用 Python、Excel 或可视化工具画训练曲线；
- 比较不同学习率和模型配置；
- 即使 checkpoint 被覆盖，也保留完整实验历史。

重点曲线通常是：

1. train loss 与 val loss；
2. train Top-1 与 val Top-1；
3. learning rate；
4. gradient norm。

