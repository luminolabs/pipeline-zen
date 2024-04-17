from transformers import AutoTokenizer


def auto(tokenizer: str):
    return AutoTokenizer.from_pretrained(tokenizer)