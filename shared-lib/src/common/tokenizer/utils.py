from common.tokenizer import nlp


def tokenizer_factory(tokenizer_id: str):
    if 'bert' in tokenizer_id:
        return nlp.auto(tokenizer_id)
