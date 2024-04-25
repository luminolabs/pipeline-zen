from unittest.mock import patch

import pytest
from transformers import ResNetPreTrainedModel

from common.model.utils import model_factory


@patch('common.model.utils.image.resnet')
def test_model_factory(resnet):
    # Invalid `loss_func_name` raises error
    with pytest.raises(TypeError):
        model_factory(model_kind='foo', model_base='resnet')

    # Valid function arguments return a new object
    # We don't want to actually fetch the model here, which is
    # what the real `resnet()` function will do, so instead
    # we make it return a string value to compare to;
    # we are testing the `model_factory()` not the `resnet()` method
    resnet.return_value = 'resnet model...'
    r = model_factory(model_kind='single_label', model_base='resnet')
    assert r == resnet.return_value
