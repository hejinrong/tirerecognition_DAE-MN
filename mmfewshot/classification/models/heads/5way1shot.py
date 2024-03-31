# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, List

import torch
import torch.nn.functional as F
from mmcls.models.builder import HEADS
from torch import Tensor
import torch.nn as nn
from mmfewshot.classification.datasets import label_wrapper
from .base_head import BaseFewShotHead

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SMyBidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SMyBidirectionalLSTM, self).__init__()

        # Define forward and backward LSTM layers
        self.forward_lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=False)
        self.backward_lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=False)

        # Bidirectional LSTM layer (summing the outputs)
        self.bidirectional_layer = nn.LSTM(input_size=hidden_size * 2, hidden_size=hidden_size, batch_first=True,
                                           bidirectional=True)

    def forward(self, x):
        # Forward LSTM
        #         print(x.shape)
        forward_output, _ = self.forward_lstm(x)

        # Backward LSTM
        backward_output, _ = self.backward_lstm(torch.flip(x, [1]))  # Reverse the sequence

        # Concatenate forward and backward outputs
        bidirectional_input = torch.cat((forward_output, torch.flip(backward_output, [1])), dim=-1)

        # Bidirectional LSTM layer
        bidirectional_output, _ = self.bidirectional_layer(bidirectional_input)
        #         print(x , bidirectional_output)
        #         print(x.shape , bidirectional_output.shape)
        # Sum the input and outputs
        result = x + bidirectional_output

        return result


# input_size, hidden_size =
# Initialize the custom Bidirectional LSTM model
s = SMyBidirectionalLSTM(512, 512 // 2).to(device)


@HEADS.register_module()
class MatchingHead(BaseFewShotHead):
    """Classification head for `MatchingNet.

    <https://arxiv.org/abs/1606.04080>`_.

    Note that this implementation is without FCE(Full Context Embeddings).

    Args:
        temperature (float): The scale factor of `cls_score`.
        loss (dict): Config of training loss.
    """

    def __init__(self,
                 temperature: float = 100,
                 loss: Dict = dict(type='NLLLoss', loss_weight=1.0),
                 *args,
                 **kwargs) -> None:
        super().__init__(loss=loss, *args, **kwargs)
        self.temperature = temperature

        # used in meta testing
        self.support_feats_list = []
        self.support_labels_list = []
        self.support_feats = None
        self.support_labels = None
        self.class_ids = None

    def forward_train(self, support_feats: Tensor, support_labels: Tensor,
                      query_feats: Tensor, query_labels: Tensor,
                      **kwargs) -> Dict:
        """Forward training data.

        Args:
            support_feats (Tensor): Features of support data with shape (N, C).
            support_labels (Tensor): Labels of support data with shape (N).
            query_feats (Tensor): Features of query data with shape (N, C).
            query_labels (Tensor): Labels of query data with shape (N).

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        #         print("第一处：")
        #         print("support_feats:",support_feats.shape)
        #         print("query_feats:",query_feats.shape)

        support_feats = support_feats.view(25, 1, 512)
        #         print(support_feats.shape)
        support_feats = s(support_feats)
        support_feats = support_feats.view(25, 512)

        query_feats = query_feats.view(80, 1, 512)
        #         print(query_feats.shape)
        query_feats = s(query_feats)
        query_feats = query_feats.view(80, 512)

        class_ids = torch.unique(support_labels).cpu().tolist()
        cosine_distance = torch.mm(
            F.normalize(query_feats),
            F.normalize(support_feats).transpose(0, 1))
        scores = F.softmax(cosine_distance * self.temperature, dim=-1)
        scores = torch.cat([
            scores[:, support_labels == class_id].mean(1, keepdim=True)
            for class_id in class_ids
        ],
            dim=1).log()
        query_labels = label_wrapper(query_labels, class_ids)
        losses = self.loss(scores, query_labels)
        return losses

    def forward_support(self, x: Tensor, gt_label: Tensor, **kwargs) -> None:
        """Forward support data in meta testing."""

        self.support_feats_list.append(x)
        self.support_labels_list.append(gt_label)

    def forward_query(self, x: Tensor, **kwargs) -> List:
        """Forward query data in meta testing."""
        if x != None:
            #             print(1)

            #             print("空的时候执行",x.shape)
            x = x.view(75, 1, 512)
            #             print(x.shape)
            #             print("&&&&")
            x = s(x)
            x = x.view(75, 512)

        if self.support_feats != None:
            print(2)
            print(self.support_feats.shape)
            self.support_feats = self.support_feats.view(5, 1, 512)
            #             print(self.support_feats.shape)
            self.support_feats = self.support_feats
            self.support_feats = self.support_feats.view(5, 512)

        cosine_distance = torch.mm(
            F.normalize(x),
            F.normalize(self.support_feats).transpose(0, 1))
        scores = F.softmax(cosine_distance * self.temperature, dim=-1)
        scores = torch.cat([
            scores[:, self.support_labels == class_id].mean(1, keepdim=True)
            for class_id in self.class_ids
        ],
            dim=1)
        pred = F.softmax(scores, dim=1)
        pred = list(pred.detach().cpu().numpy())
        return pred

    def before_forward_support(self) -> None:
        """Used in meta testing.

        This function will be called before model forward support data during
        meta testing.
        """
        # reset saved features for testing new task
        self.support_feats_list.clear()
        self.support_labels_list.clear()
        self.support_feats = None
        self.support_labels = None
        self.class_ids = None

    def before_forward_query(self) -> None:
        """Used in meta testing.

        This function will be called before model forward query data during
        meta testing.
        """
        self.support_feats = torch.cat(self.support_feats_list, dim=0)
        self.support_labels = torch.cat(self.support_labels_list, dim=0)
        self.class_ids, _ = torch.unique(self.support_labels).sort()
        if max(self.class_ids) + 1 != len(self.class_ids):
            warnings.warn(
                f'the max class id is {max(self.class_ids)}, while '
                f'the number of different number of classes is '
                f'{len(self.class_ids)}, it will cause label '
                f'mismatching problem.', UserWarning)
