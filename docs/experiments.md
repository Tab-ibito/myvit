# myvit 实验文件

主要使用 ChatGPT Sol-5.6 完成代码生成和实验理解。

## 1. 模型代码 2026.9.1

`src/myvit/vit.py` 主要涉及 Transformer Block / MLP 等模块的实现。以此复习了注意力机制的原理，熟悉 PyTorch 的代码实现方式。

## 2. 训练代码 2026.9.1

学习了该如何关注训练代码的指标。包括

- Epoch 轮次
- Top1 & Top5
- Accuracy
- 梯度 L2 norm
- loss

等。

一段训练代码的核心逻辑在

```Python
for epoch in range(num_epochs):
    for images, targets in train_loader:
        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

    validation_metrics = evaluate(model, val_loader)
    scheduler.step()
    save_checkpoint()
```

通过 debug 工具观察上述过程中部分梯度矩阵，张量形状等变化。

