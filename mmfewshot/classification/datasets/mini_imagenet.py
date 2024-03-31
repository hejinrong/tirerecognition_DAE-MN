# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from typing import List, Optional, Sequence, Union

import mmcv
import numpy as np
from mmcls.datasets.builder import DATASETS
from typing_extensions import Literal

from .base import BaseFewShotDataset

TRAIN_CLASSES = [
"n0000001","n0000002","n0000003","n0000004","n0000005","n0000006","n0000007","n0000008","n0000009","n0000010","n0000011","n0000012","n0000013","n0000014","n0000015","n0000016","n0000017","n0000018","n0000019","n0000020","n0000021","n0000022","n0000023","n0000024","n0000025","n0000026","n0000027","n0000028","n0000029","n0000030"

]
VAL_CLASSES = [
       "n0000031","n0000032","n0000033","n0000034","n0000035","n0000036","n0000037"
]
TEST_CLASSES = [
    "n0000038","n0000039","n0000040","n0000041","n0000042","n0000043","n0000044","n0000045","n0000046"
]


@DATASETS.register_module()
class MiniImageNetDataset(BaseFewShotDataset):
    """MiniImageNet dataset for few shot classification.

    Args:
        subset (str| list[str]): The classes of whole dataset are split into
            three disjoint subset: train, val and test. If subset is a string,
            only one subset data will be loaded. If subset is a list of
            string, then all data of subset in list will be loaded.
            Options: ['train', 'val', 'test']. Default: 'train'.
        file_format (str): The file format of the image. Default: 'JPEG'
    """

    resource = 'https://github.com/twitter/meta-learning-lstm/tree/master/data/miniImagenet'  # noqa

    TRAIN_CLASSES = TRAIN_CLASSES
    VAL_CLASSES = VAL_CLASSES
    TEST_CLASSES = TEST_CLASSES

    def __init__(self,
                 subset: Literal['train', 'test', 'val'] = 'train',
                 file_format: str = 'JPEG',
                 *args,
                 **kwargs):
        if isinstance(subset, str):
            subset = [subset]
        for subset_ in subset:
            assert subset_ in ['train', 'test', 'val']
        self.subset = subset
        self.file_format = file_format
        super().__init__(*args, **kwargs)

    def get_classes(
            self,
            classes: Optional[Union[Sequence[str],
                                    str]] = None) -> Sequence[str]:
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): Three types of input
                will correspond to different processing logics:

                - If `classes` is a tuple or list, it will override the
                  CLASSES predefined in the dataset.
                - If `classes` is None, we directly use pre-defined CLASSES
                  will be used by the dataset.
                - If `classes` is a string, it is the path of a classes file
                  that contains the name of all classes. Each line of the file
                  contains a single class name.

        Returns:
            tuple[str] or list[str]: Names of categories of the dataset.
        """
        if classes is None:
            class_names = []
            for subset_ in self.subset:
                if subset_ == 'train':
                    class_names += self.TRAIN_CLASSES
                elif subset_ == 'val':
                    class_names += self.VAL_CLASSES
                elif subset_ == 'test':
                    class_names += self.TEST_CLASSES
                else:
                    raise ValueError(f'invalid subset {subset_} only '
                                     f'support train, val or test.')
        elif isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')
        return class_names

    def load_annotations(self) -> List:
        """Load annotation according to the classes subset."""
        img_file_list = {
            class_name: sorted(
                os.listdir(osp.join(self.data_prefix, 'images', class_name)),
                key=lambda x: int(x.split('.')[0].split('_')[-1]))
            for class_name in self.CLASSES
        }
        data_infos = []
        for subset_ in self.subset:
            ann_file = osp.join(self.data_prefix, f'{subset_}.csv')
            assert osp.exists(ann_file), \
                f'Please download ann_file through {self.resource}.'
            with open(ann_file) as f:
                for i, line in enumerate(f):
                    # skip file head
                    if i == 0:
                        continue
                    filename, class_name = line.strip().split(',')
                    filename = img_file_list[class_name][
                        int(filename.split('.')[0][-5:]) - 1]
                    gt_label = self.class_to_idx[class_name]
                    info = {
                        'img_prefix':
                        osp.join(self.data_prefix, 'images', class_name),
                        'img_info': {
                            'filename': filename
                        },
                        'gt_label':
                        np.array(gt_label, dtype=np.int64)
                    }
                    data_infos.append(info)
        return data_infos
