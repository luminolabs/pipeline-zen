import re

import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")


def transforms_set_1(text: str) -> str:
    """
     - Lowercase the sentence
     - Change "'t" to "not"
     - Remove "@name"
     - Isolate and remove punctuations except "?"
     - Remove other special characters
     - Remove stop words except "not" and "can"
     - Remove trailing whitespace
     """
    text = text.lower()
    # Change 't to 'not'
    text = re.sub(r"\'t", " not", text)
    # Remove @name
    text = re.sub(r'(@.*?)[\s]', ' ', text)
    # Isolate and remove punctuations except '?'
    text = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r'  ', text)
    text = re.sub(r'[^\w\s\?]', ' ', text)
    # Remove some special characters
    text = re.sub(r'([\;\:\|•«\n])', ' ', text)
    # Remove stopwords except 'not' and 'can'
    text = " ".join([word for word in text.split()
                  if word not in stopwords.words('english')
                  or word in ['not', 'can']])
    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def transforms_set_2(text: str) -> str:
    pass
