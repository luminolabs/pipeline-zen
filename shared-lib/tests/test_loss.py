import pytest
from segmentation_models_pytorch.losses import FocalLoss

from common.loss.utils import loss_factory


def test_loss_factory(logger):
    # Invalid `loss_func_name` raises error
    with pytest.raises(TypeError):
        loss_factory(loss_func_name='foo', mode='binary', logger=logger)

    # Valid function arguments return a new object
    r = loss_factory(loss_func_name='FocalLoss', mode='binary', logger=logger)
    assert isinstance(r, FocalLoss)
