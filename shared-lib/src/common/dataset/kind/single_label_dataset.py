from typing import Any, Tuple

from common.dataset.kind.base import BaseDatasetKind


class SingleLabelDataset(BaseDatasetKind):
    """
    This is an iterator that works with a single label dataset.
    For example, it will work with a dataset with a single `annotation` label column,
    but it won't work with a dataset with multiple columns, such as `annotation` and `category`.
    """

    def __init__(self, label_col: str, **kwargs):
        """
        :param label_col: The name of the label key (ex. `annotation` or `category`)
        """
        super().__init__(**kwargs)
        if not isinstance(label_col, str):
            raise TypeError('`label_col` must be of type `str`')
        self.label_col = label_col

    def _num_labels(self) -> int:
        """
        This iterator supports single label datasets
        """
        return 1

    def _getitem(self, item: int) -> Tuple[Any, Any]:
        """
        The responsibility of this iterator is to convert data from a `BaseDatasetProvider` to
        a `tuple` of `(input, label)`, that can be consumed from downstream iterators such
        as preprocessors and dataloaders.
        """
        return self.dataset[item][self.input_col], self.dataset[item][self.label_col]
