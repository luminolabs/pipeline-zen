from typing import Callable

from segmentation_models_pytorch.losses import FocalLoss
from torch.nn import CrossEntropyLoss


def loss_factory(loss_func_name: str, **kwargs) -> Callable:
    if 'focal' == loss_func_name:
        return FocalLoss(**kwargs)
    elif 'cross_entropy' == loss_func_name:
        return CrossEntropyLoss(**kwargs)
