from transformers import AutoTokenizer, PreTrainedTokenizerBase

"""
Collection of NLP tokenizers.
"""


def auto(tokenizer: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(tokenizer)
