from typing import Callable

from segmentation_models_pytorch.losses import FocalLoss
from torch.nn import CrossEntropyLoss


def loss_factory(loss_func_name: str, **kwargs) -> Callable:
    """
    Factory function for creating loss functions.

    :param loss_func_name: Loss function name
    :param kwargs: Keyword arguments to pass to the loss function
    :return: Instantiated loss function
    """
    print(f'Using `{loss_func_name}` loss function')
    if 'FocalLoss' == loss_func_name:
        return FocalLoss(**kwargs)
    elif 'CrossEntropyLoss' == loss_func_name:
        return CrossEntropyLoss(**kwargs)
    else:
        raise TypeError(f'loss_func_name: {loss_func_name} is not a valid option')
