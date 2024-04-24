from transformers import AutoTokenizer, PreTrainedTokenizerBase


def auto(tokenizer: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(tokenizer)
