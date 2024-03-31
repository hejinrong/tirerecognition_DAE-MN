# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import torch.nn as nn
import torch.nn.functional as F
from mmcls.models.builder import LOSSES
from mmcls.models.losses.utils import weighted_loss
from torch import Tensor
from typing_extensions import Literal


@weighted_loss
def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Wrapper of mse loss."""
    return F.mse_loss(pred, target, reduction='none')


@LOSSES.register_module()
class MSELoss(nn.Module):
    """MSELoss.

    Args:
        reduction (str): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum". Default: 'mean'.
        loss_weight (float): The weight of the loss. Default: 1.0.
    """

    def __init__(self,
                 reduction: Literal['none', 'mean', 'sum'] = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[Union[float, int]] = None,
                reduction_override: str = None) -> Tensor:
        """Forward function of loss.

        Args:
            pred (Tensor): The prediction with shape (N, *), where * means
                any number of additional dimensions.
            target (Tensor): The learning target of the prediction
                with shape (N, *) same as the input.
            weight (Tensor | None): Weight of the loss for each
                prediction. Default: None.
            avg_factor (float | int | None): Average factor that is used to
                average the loss. Default: None.
            reduction_override (str | None): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum". Default: None.

        Returns:
            Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * mse_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss
# 这个代码定义了一个均方误差（Mean Squared Error，MSE）损失函数。损失函数接受预测值pred和目标值target作为输入，并计算它们之间的均方误差。
#
# 具体来说，该损失函数使用F.mse_loss()函数计算预测值pred和目标值target之间的均方误差。可以选择使用reduction参数来指定如何将损失降至标量。可选的选项包括"none"、"mean"和"sum"，分别表示不降低损失、求均值和求和。默认情况下，采用"mean"方法将损失降低到标量。
#
# 此外，可以使用weight参数为每个预测值指定权重，以便在计算损失时进行加权。还可以使用avg_factor参数指定平均因子，用于对损失进行平均。如果reduction_override参数不为空，则使用其覆盖原始的减少方法。
#
# 最后，损失函数的输出是计算得到的损失值。
#
# 总结起来，这个代码定义的损失函数是均方误差（MSE）损失函数，可根据需要进行配置和使用。
