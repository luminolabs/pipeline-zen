from torchtune.models import convert_weights


def update_convert_weights_from_hf():
    """
    mistralai/Mixtral-8x7B-Instruct-v0.1 specific change:
    Allows loading model weights for this model. This is a temporary fix (hack!)
    till torchtune adds proper support for this.

    :return: None
    """
    convert_weights._FROM_HF.update({
        'model.layers.{}.block_sparse_moe.experts.{}.w1.weight': 'foo.{}.bar',
        'model.layers.{}.block_sparse_moe.experts.{}.w2.weight': 'foo.{}.bar',
        'model.layers.{}.block_sparse_moe.experts.{}.w3.weight': 'foo.{}.bar',
        'model.layers.{}.block_sparse_moe.gate.weight': 'foo.{}.bar',
    })