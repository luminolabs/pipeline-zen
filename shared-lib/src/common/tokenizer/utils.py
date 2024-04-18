from transformers import PreTrainedTokenizerBase

from common.tokenizer import nlp


def tokenizer_factory(tokenizer_id: str) -> PreTrainedTokenizerBase:
    if 'bert' in tokenizer_id:
        return nlp.auto(tokenizer_id)
