from typing import List

from torchtune.models.mistral import lora_mistral
from torchtune.modules import TransformerDecoder
from torchtune.modules.peft import LORA_ATTN_MODULES


def lora_mistral_8x7b(
        lora_attn_modules: List[LORA_ATTN_MODULES],
        apply_lora_to_mlp: bool = False,
        apply_lora_to_output: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16,
        quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Builder for creating a Mistral 7B model with LoRA enabled.

    Args:
        lora_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        apply_lora_to_output (bool): whether to apply LoRA to the model's final output projection.
            Default: False
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        quantize_base (bool): Whether to quantize base model weights

    Returns:
        TransformerDecoder: Instantiation of Mistral 8x7B model with LoRA applied
    """
    return lora_mistral(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=32_000,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        intermediate_dim=14336,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=10_000,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        quantize_base=quantize_base,
    )