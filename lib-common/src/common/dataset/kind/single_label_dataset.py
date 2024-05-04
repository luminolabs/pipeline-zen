from typing import Any, Tuple, Optional

from common.dataset.kind.base import BaseDatasetKind


class SingleLabelDataset(BaseDatasetKind):
    """
    This is an iterator that works with a single label dataset.
    For example, it will work with a dataset with a single `annotation` label column,
    but it won't work with a dataset with multiple columns, such as `annotation` and `category`.
    """

    def __init__(self, label_col: str, master_col: Optional[str] = None, **kwargs):
        """
        :param label_col: The name of the label key (ex. `annotation` or `category`)
        :param master_col: Some datasets have a single column, which contains a dict with the inputs and labels;
        in this case, `input_col` and `label_col` will be under the `master_col`. For example:
        `{'translation': {'el': 'Some Greek text', 'en': 'The English translation'}}`
        In this case, `master_col` is `translation` and `label_col` is `en`.
        """
        super().__init__(**kwargs)
        if not isinstance(label_col, str):
            raise TypeError('`label_col` must be of type `str`')
        if master_col and not isinstance(master_col, str):
            raise TypeError('`master_col` must be of type `str`')
        self.label_col = label_col
        self.master_col = master_col

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
        item = self.dataset[item]
        if self.master_col:
            item = item[self.master_col]
        return item[self.input_col], item[self.label_col]
