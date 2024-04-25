import pytest
from segmentation_models_pytorch.losses import FocalLoss

from common.loss.utils import loss_factory


def test_loss_factory():
    # Invalid `loss_func_name` raises error
    with pytest.raises(TypeError):
        loss_factory(loss_func_name='foo', mode='binary')

    # Valid function arguments return a new object
    r = loss_factory(loss_func_name='focal', mode='binary')
    assert isinstance(r, FocalLoss)
