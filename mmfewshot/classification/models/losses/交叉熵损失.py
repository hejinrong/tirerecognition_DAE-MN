
# 当面临小样本学习问题时，除了MSE和NLL损失函数之外，还有一种常用的损失函数是交叉熵损失（Cross-Entropy Loss）。
#
# 交叉熵损失函数：交叉熵损失函数广泛应用于分类任务中。它可以有效地处理多分类问题，并在训练过程中鼓励模型更好地区分不同的类别。对于小样本分类问题，交叉熵损失函数通常比MSE和NLL损失函数更合适。交叉熵损失函数将预测值与目标值之间的差异量化为损失，并通过最大化正确类别的概率来优化分类模型。
#
# 在PyTorch中，可以使用torch.nn.CrossEntropyLoss来实现交叉熵损失函数。它接受预测值和目标值作为输入，并自动进行softmax操作和负对数似然计算。可以根据需要选择不同的参数，如权重加权和减少方法等。
#
# 需要注意的是，在小样本学习中，还可以考虑一些特定的损失函数或技术，如Focal Loss、Label Smoothing等，以进一步改善模型训练效果。这些方法可以针对小样本问题的特点进行调整和优化，使得模型更好地适应小样本数据。
#
# 综上所述，交叉熵损失函数是适用于小样本学习的常用损失函数之一，可以作为MSE和NLL损失函数的替代选择。


import torch.nn.functional as F

def cross_entropy_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Wrapper of cross entropy loss."""
    return F.cross_entropy(pred, target)
