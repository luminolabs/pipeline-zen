from transformers import PreTrainedTokenizerBase, TensorType
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.utils import PaddingStrategy

from common.tokenizer import nlp


def tokenizer_factory(tokenizer_id: str) -> PreTrainedTokenizerBase:
    if 'bert' in tokenizer_id:
        return nlp.auto(tokenizer_id)


def tokenize_inputs(inputs, tokenizer: PreTrainedTokenizerBase, model_args: dict, device):
    # Tokenize batch of inputs
    # Tensor data need to be of same length, so we need to
    # set max size and padding options
    tokenized_values = tokenizer(
        inputs,
        padding=PaddingStrategy.MAX_LENGTH,
        truncation=TruncationStrategy.ONLY_FIRST,
        max_length=tokenizer.model_max_length,
        return_tensors=TensorType.PYTORCH)
    # Replace original inputs with tokenized inputs
    inputs = tokenized_values.get('input_ids')
    # Load attention masks to device
    attention_masks = tokenized_values.get('attention_mask').to(device)
    model_args['attention_mask'] = attention_masks
    return inputs
