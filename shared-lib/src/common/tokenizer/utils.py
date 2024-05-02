from logging import Logger

from transformers import PreTrainedTokenizerBase, TensorType
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.utils import PaddingStrategy

from common.tokenizer import nlp


def tokenizer_factory(tokenizer_id: str, logger: Logger) -> PreTrainedTokenizerBase:
    """
    Factory method for tokenizers.

    :param tokenizer_id: The tokenizer name
    :param logger: The logger instance
    :return: Tokenizer instance
    """
    logger.info(f'Using `{tokenizer_id}` tokenizer')
    if any(x in tokenizer_id for x in ('bert', 't5')):
        return nlp.auto(tokenizer_id)
    else:
        raise TypeError(f'tokenizer_id: {tokenizer_id} is not a valid option')


def tokenize_inputs(inputs, tokenizer: PreTrainedTokenizerBase, model_args: dict,
                    device: str, **kwargs):
    """
    Tokenize inputs using the tokenizer, and place attention masks on device.

    :param inputs: The inputs to be tokenized
    :param tokenizer: The tokenizer to use
    :param model_args: Additional model arguments
    :param device: The device to load the attentions masks to
    :return:
    """

    if not kwargs.get('max_length'):
        kwargs['max_length'] = tokenizer.model_max_length

    # Tensor data need to be of same length, so we need to
    # set max size and padding options
    tokenized_values = tokenizer(
        inputs,
        padding=PaddingStrategy.MAX_LENGTH,
        truncation=TruncationStrategy.ONLY_FIRST,
        return_tensors=TensorType.PYTORCH,
        **kwargs)
    # Replace original inputs with tokenized inputs
    inputs = tokenized_values.get('input_ids')
    # Load attention masks to device
    attention_masks = tokenized_values.get('attention_mask').to(device)
    model_args['attention_mask'] = attention_masks
    return inputs
