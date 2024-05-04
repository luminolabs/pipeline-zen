"""
Collection of preprocessing functions for text processing.
"""


def strip(text: str) -> str:
    """
    :param text: Input text
    :return: Stripped text

    >>> strip('   aaa   ')
    'aaa'
    """
    return text.strip()
