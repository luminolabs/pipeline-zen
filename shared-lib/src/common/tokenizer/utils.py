from common.tokenizer import nlp


def tokenizer_factory(tokenizer: str):
    if 'bert' in tokenizer:
        return nlp.auto(tokenizer)
