from unittest.mock import patch

import pytest
from transformers import ResNetPreTrainedModel

from common.model.utils import model_factory
from common.tokenizer.utils import tokenizer_factory


@patch('common.tokenizer.utils.nlp.auto')
def test_tokenizer_factory(nlp_auto):
    # Invalid `tokenizer_id` raises error
    with pytest.raises(TypeError):
        tokenizer_factory(tokenizer_id='foo')

    # Valid function arguments return a new object
    # We don't want to actually fetch the model here, which is
    # what the real `resnet()` function will do, so instead
    # we make it return a string value to compare to;
    # we are testing the `model_factory()` not the `resnet()` method
    nlp_auto.return_value = 'bert tokenizer...'
    r = tokenizer_factory(tokenizer_id='bert')
    assert r == nlp_auto.return_value
