"""OLMoE model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation


class OlmoeConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`OlmoeModel`]. It is used to instantiate an OLMoE
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the [allenai/OLMoE-1B-7B-0824](https://huggingface.co/allenai/OLMoE-1B-7B-0824).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50304):
            Vocabulary size of the OLMoE model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`OlmoeModel`]
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 16):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 1):
            Padding token id.
        bos_token_id (`int`, *optional*):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 50279):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        clip_qkv (`float`, *optional*):
            If not `None`, elements of query, key and value attention states are clipped so that their
            absolute value does not exceed this value.
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Number of selected experts.
        num_experts (`int`, *optional*, defaults to 64):
            Number of routed experts.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether or not the router logits should be returned by the model. Enabeling this will also
            allow the model to output the auxiliary loss, including load balancing loss and router z-loss.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.01):
            The aux loss factor for the total loss.
        norm_topk_prob (`bool`, *optional*, defaults to `False`):
            Whether to normalize the topk probabilities.

    ```python
    >>> from transformers import OlmoeModel, OlmoeConfig

    >>> # Initializing a OLMoE 7B A1B style configuration
    >>> configuration = OlmoeConfig()

    >>> # Initializing a model from the OLMoE 7B A1B style configuration
    >>> model = OlmoeModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "olmoe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50304,
        hidden_size=2048,
        intermediate_size=2048,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=None,
        eos_token_id=50279,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        clip_qkv=None,
        num_experts_per_tok=8,
        num_experts=64,
        output_router_logits=False,
        router_aux_loss_coef=0.01,
        norm_topk_prob=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.clip_qkv = clip_qkv
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.norm_topk_prob = norm_topk_prob
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.cache_utils import Cache
import torch.nn.functional as F
from itertools import combinations
import logging

logger = logging.getLogger(__name__)

from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
)
# from my_configuration.my_configuration_olmoe import OlmoeConfig

class OlmoeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        """
        OlmoeRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed



# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

import math
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.utils import logging
from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock
import torch.nn.functional as F

# from modeling_olmoe import *


if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

logger = logging.get_logger(__name__)

class AttentionLoRALinear(nn.Module):
    """A module to apply Low-Rank Adaptation (LoRA) using nn.Linear for A and B."""
    
    def __init__(self, attention_bias, in_features: int, out_features: int, rank: int = 4):
        super().__init__()
        self.rank = rank
        
        # Use nn.Linear for low-rank matrices A and B.
        self.A = nn.Linear(in_features, rank, bias=attention_bias)
        self.B = nn.Linear(rank, out_features, bias=attention_bias)
        
    def forward(self, x):
        # Apply the low-rank adaptation directly using nn.Linear layers.
        return self.B(self.A(x))

class SVD_OlmoeAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, lora_rank, config: OlmoeConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        # self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        # self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        # self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)

        self.q_proj = AttentionLoRALinear(config.attention_bias, self.hidden_size, self.num_heads * self.head_dim, rank=lora_rank)
        self.k_proj = AttentionLoRALinear(config.attention_bias, self.hidden_size, self.num_key_value_heads * self.head_dim, rank=lora_rank)
        self.v_proj = AttentionLoRALinear(config.attention_bias, self.hidden_size, self.num_key_value_heads * self.head_dim, rank=lora_rank)
        self.o_proj = AttentionLoRALinear(config.attention_bias, self.hidden_size, self.hidden_size, rank=lora_rank)

        self.q_norm = OlmoeRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.k_norm = OlmoeRMSNorm(
            (self.hidden_size // self.num_heads) * self.num_key_value_heads, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_norm(self.q_proj(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)

        if self.config.clip_qkv is not None:
            query_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            key_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            value_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        print("========================================================")
        return attn_output, attn_weights, past_key_value


class SVD_OlmoeFlashAttention2(SVD_OlmoeAttention):
    """
    OLMoE flash attention module. This module inherits from `OlmoeAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_norm(self.q_proj(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)
        if self.config.clip_qkv is not None:
            query_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            key_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            value_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (OlmoeRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class SVD_OlmoeSdpaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper using SDPA API"""

    def __init__(self, lora_rank, config: OlmoeConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Initialize projections using LoRA
        self.q_proj = AttentionLoRALinear(config.attention_bias, self.hidden_size, self.num_heads * self.head_dim, rank=lora_rank)
        self.k_proj = AttentionLoRALinear(config.attention_bias, self.hidden_size, self.num_key_value_heads * self.head_dim, rank=lora_rank)
        self.v_proj = AttentionLoRALinear(config.attention_bias, self.hidden_size, self.num_key_value_heads * self.head_dim, rank=lora_rank)
        self.o_proj = AttentionLoRALinear(config.attention_bias, self.hidden_size, self.hidden_size, rank=lora_rank)

        self.q_norm = OlmoeRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.k_norm = OlmoeRMSNorm(
            (self.hidden_size // self.num_heads) * self.num_key_value_heads, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            logger.warning_once(
                "OlmoeModel is using OlmoeSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            # Since we are not implementing the fallback here, you would need to implement it similarly to how it was done in the parent class.
            return None  # Placeholder for fallback implementation

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_norm(self.q_proj(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)

        if self.config.clip_qkv is not None:
            query_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            key_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            value_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # Ensure inputs are contiguous for SDPA with memory-efficient backend
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

class SVD_OlmoeSdpaAttention_old(nn.Module):
    """
    OLMoE attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `OlmoeAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from OlmoeAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "OlmoeModel is using OlmoeSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_norm(self.q_proj(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)

        if self.config.clip_qkv is not None:
            query_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            key_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            value_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        # if attention_mask is not None and cache_position is not None:
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value



# class SVD_OlmoeMLP_weight:
#     def __init__(self, config, expert_group_index=0, ratio=1, rank=602):
#         self.hidden_size = config.hidden_size
#         self.intermediate_size = config.intermediate_size
#         self.ratio = ratio
#         self.lora_rank = rank
#         low_rank = int(self.intermediate_size * self.hidden_size * self.ratio / (self.intermediate_size + self.hidden_size))
#         # Initialize weights for low-rank projections
#         self.layers = {
#             'gate': {
#                 'A': nn.Linear(self.hidden_size, self.lora_rank, bias=False),
#                 'B': nn.Linear(self.lora_rank, self.intermediate_size, bias=False)
#             },
#             'up': {
#                 'A': nn.Linear(self.hidden_size, self.lora_rank, bias=False),
#                 'B': nn.Linear(self.lora_rank, self.intermediate_size, bias=False)
#             },
#             'down': {
#                 'A': nn.Linear(self.intermediate_size, self.lora_rank, bias=False),
#                 'B': nn.Linear(self.lora_rank, self.hidden_size, bias=False)
#             }
#         }

class SVD_OlmoeMLP_weight(nn.Module):
    def __init__(self, config, expert_group_index=0, ratio=1, rank=602):
        super().__init__()  # 继承 nn.Module 的初始化方法
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.ratio = ratio
        self.lora_rank = rank
        low_rank = int(self.intermediate_size * self.hidden_size * self.ratio / (self.intermediate_size + self.hidden_size))
        self.expert_group_index = int(expert_group_index)
        
        # Initialize weights for low-rank projections
        self.layers = nn.ModuleDict({
            'gate': nn.Sequential(
                nn.Linear(self.hidden_size, self.lora_rank, bias=False),
                nn.Linear(self.lora_rank, self.intermediate_size, bias=False)
            ),
            'up': nn.Sequential(
                nn.Linear(self.hidden_size, self.lora_rank, bias=False),
                nn.Linear(self.lora_rank, self.intermediate_size, bias=False)
            ),
            'down': nn.Sequential(
                nn.Linear(self.intermediate_size, self.lora_rank, bias=False),
                nn.Linear(self.lora_rank, self.hidden_size, bias=False)
            )
        })

class SVD_Basenet_weight(nn.Module):
    def __init__(self, config):
        super().__init__()  # 继承 nn.Module 的初始化方法
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.hidden_act = config.hidden_act
        
        # Initialize parameters as nn.Parameter objects and store them in a dictionary
        self.layers = nn.ModuleDict({
            'gate': nn.Linear(self.hidden_size, self.intermediate_size, bias=False),
            'up': nn.Linear(self.hidden_size, self.intermediate_size, bias=False),
            'down': nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        })


class SVD_OlmoeSparseMoeBlock(nn.Module):
    def __init__(self, config, num_group=1, ratio=1, rank=602):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.rank = rank
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)

        self.experts = nn.ModuleList([SVD_OlmoeMLP_weight(config, ratio=ratio, rank=rank) for _ in range(self.num_experts)])
        self.basenet = nn.ModuleList([SVD_Basenet_weight(config) for _ in range(num_group)])

    def basenet_forward(self, x, expert_index):
        #print(x.shape)   #[299,2048],[401.2048]
        basenet_index = self.experts[expert_index].expert_group_index

        gate_proj, up_proj, down_proj = self.basenet[basenet_index].layers['gate'], self.basenet[basenet_index].layers['up'], self.basenet[basenet_index].layers['down']
        #print(gate_proj.weight.shape, up_proj.weight.shape, down_proj.weight.shape)  #[1024, 2048]) torch.Size([1024, 2048]) torch.Size([2048, 1024]
        # Access the sub-modules within Sequential using indices
        gate_proj_lora_A, gate_proj_lora_B = self.experts[expert_index].layers['gate'][0], self.experts[expert_index].layers['gate'][1]
        up_proj_lora_A, up_proj_lora_B = self.experts[expert_index].layers['up'][0], self.experts[expert_index].layers['up'][1]
        down_proj_lora_A, down_proj_lora_B = self.experts[expert_index].layers['down'][0], self.experts[expert_index].layers['down'][1]

        act_fn = ACT2FN[self.basenet[basenet_index].config.hidden_act]

        gate_output = gate_proj(x)
        up_output = up_proj(x)
        #print(gate_output.shape) #299, 1024]) [401, 1024])
        gate_output_lora = gate_proj_lora_B(gate_proj_lora_A(x))
        up_output_lora = up_proj_lora_B(up_proj_lora_A(x))

        adjusted_gate_output = gate_output + gate_output_lora
        adjusted_up_output = up_output + up_output_lora

        down_output = down_proj(act_fn(adjusted_gate_output) * adjusted_up_output)
        down_output_lora = down_proj_lora_B(down_proj_lora_A(act_fn(adjusted_gate_output) * adjusted_up_output))
        # print('down', down_output.shape) #299, 1024]) [401, 1024])
        return down_output + down_output_lora

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        selected_experts_set = set(selected_experts.flatten().tolist())

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in selected_experts_set:
            idx, top_x = torch.where(expert_mask[expert_idx])

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            
            current_hidden_states = self.basenet_forward(current_state, expert_idx) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        
        return final_hidden_states, router_logits

class SVD_OlmoeSparseMoeBlock_No_Basenet(nn.Module):
    def __init__(self, config, ratio=1, rank = 602):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.rank = rank
        self.hidden_act = config.hidden_act
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([SVD_OlmoeMLP_weight(config, ratio=ratio, rank=rank) for _ in range(self.num_experts)])

    def basenet_forward(self, x, expert_index):
        gate_proj_lora_A, gate_proj_lora_B = self.experts[expert_index].layers['gate'][0], self.experts[expert_index].layers['gate'][1]
        up_proj_lora_A, up_proj_lora_B = self.experts[expert_index].layers['up'][0], self.experts[expert_index].layers['up'][1]
        down_proj_lora_A, down_proj_lora_B = self.experts[expert_index].layers['down'][0], self.experts[expert_index].layers['down'][1]
        act_fn = ACT2FN[self.hidden_act]

        gate_output_lora = gate_proj_lora_B(gate_proj_lora_A(x))
        up_output_lora = up_proj_lora_B(up_proj_lora_A(x))

        down_output_lora = down_proj_lora_B(down_proj_lora_A(act_fn(gate_output_lora) * up_output_lora))

        return down_output_lora
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        selected_experts_set = set(selected_experts.flatten().tolist())

        #final_hidden_states = torch.zeros_like(hidden_states)
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in selected_experts_set:
            idx, top_x = torch.where(expert_mask[expert_idx])

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)

            current_hidden_states = self.basenet_forward(current_state, expert_idx) * routing_weights[top_x, idx, None]
            # current_hidden_states = expert_with_index(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
     
        return final_hidden_states, router_logits  



class MoEModelWrapper(torch.nn.Module):
    def __init__(self, model, device):
        super(MoEModelWrapper, self).__init__()
        self.model = model.to(device)  # 确保模型也在device上
        self.device = device
        self.expert_counts = {}          # 激活次数统计
        self.expert_weights_sum = {}     # 路由权重累加值
        self.expert_input_l2_sum = {}    # 输入L2范数总和
        self.expert_input_count = {}     # 输入样本计数
        self.expert_output_l2_sum = {}   # 输出L2范数总和
        self.hooks = []
        self._register_hooks()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, OlmoeSparseMoeBlock):  # 替换成实际的MoE层类型
                num_experts = module.num_experts
                self.expert_counts[name] = torch.zeros(num_experts, device=self.device)
                self.expert_weights_sum[name] = torch.zeros(num_experts, device=self.device)
                self.expert_input_l2_sum[name] = torch.zeros(num_experts, device=self.device)
                self.expert_input_count[name] = torch.zeros(num_experts, device=self.device)
                self.expert_output_l2_sum[name] = torch.zeros(num_experts, device=self.device)
                
                moe_hook = module.register_forward_hook(self._moe_layer_hook(name))
                self.hooks.append(moe_hook)
                
                for expert_idx, expert in enumerate(module.experts):
                    expert_hook = expert.register_forward_hook(
                        self._expert_forward_hook(name, expert_idx))
                    self.hooks.append(expert_hook)

    def _moe_layer_hook(self, layer_name):
        def hook(module, inputs, outputs):
            final_hidden_states, router_logits = outputs
            router_weights = torch.softmax(router_logits, dim=-1).to(self.device)
            selected_experts = torch.argmax(router_weights, dim=-1)
            
            unique, counts = torch.unique(selected_experts, return_counts=True)
            self.expert_counts[layer_name][unique] += counts
            
            selected_weights = router_weights[torch.arange(router_weights.size(0)), selected_experts]
            for expert_idx in unique:
                mask = (selected_experts == expert_idx)
                self.expert_weights_sum[layer_name][expert_idx] += selected_weights[mask].sum()
                
        return hook

    def _expert_forward_hook(self, layer_name, expert_idx):
        def hook(module, inputs, outputs):
            input_tensor = inputs[0].detach().to(self.device)
            output_tensor = outputs.detach() if isinstance(outputs, torch.Tensor) else outputs[0].detach().to(self.device)
            
            input_l2 = torch.norm(input_tensor, p=2, dim=-1).mean()
            output_l2 = torch.norm(output_tensor, p=2, dim=-1).mean()
            
            # 防止NaN
            if not torch.isnan(input_l2) and not torch.isinf(input_l2):
                self.expert_input_l2_sum[layer_name][expert_idx] += input_l2
            if not torch.isnan(output_l2) and not torch.isinf(output_l2):
                self.expert_output_l2_sum[layer_name][expert_idx] += output_l2
            self.expert_input_count[layer_name][expert_idx] += 1
            
        return hook

    def reset_counts(self):
        for stats_dict in [self.expert_counts, self.expert_weights_sum, 
                           self.expert_input_l2_sum, self.expert_output_l2_sum, 
                           self.expert_input_count]:
            for tensor in stats_dict.values():
                tensor.zero_()

    def get_expert_stats(self):
        stats = {
            'activation_frequency': {},
            'sum_routing_weights': {},
            'input_l2_avg': {},
            'output_l2_avg': {}
        }
        
        for layer in self.expert_counts:
            total_activations = self.expert_counts[layer].sum()
            if total_activations > 0:
                stats['activation_frequency'][layer] = (
                    (self.expert_counts[layer] / total_activations).tolist()
                )
            else:
                stats['activation_frequency'][layer] = [0] * len(self.expert_counts[layer])
            
            stats['sum_routing_weights'][layer] = self.expert_weights_sum[layer].tolist()
            
            input_l2_avg = []
            output_l2_avg = []
            for idx in range(len(self.expert_counts[layer])):
                count = self.expert_input_count[layer][idx]
                if count > 0:
                    avg_input_l2 = self.expert_input_l2_sum[layer][idx] / count
                    avg_output_l2 = self.expert_output_l2_sum[layer][idx] / count
                    # 防止NaN
                    if not torch.isnan(avg_input_l2) and not torch.isinf(avg_input_l2):
                        input_l2_avg.append(avg_input_l2.item())
                    else:
                        input_l2_avg.append(0.0)
                        
                    if not torch.isnan(avg_output_l2) and not torch.isinf(avg_output_l2):
                        output_l2_avg.append(avg_output_l2.item())
                    else:
                        output_l2_avg.append(0.0)
                else:
                    input_l2_avg.append(0.0)
                    output_l2_avg.append(0.0)
                    
            stats['input_l2_avg'][layer] = input_l2_avg
            stats['output_l2_avg'][layer] = output_l2_avg
            
        return stats

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
class MoEModelWrapperWithLogits(nn.Module):
    def __init__(self, model, device):
        super(MoEModelWrapperWithLogits, self).__init__()
        self.model = model
        self.device = device
        self.expert_counts = {}
        self.expert_weights_sum = {}
        # 累加每个批次计算得到的 router_logits 相似性和专家输出相似性
        self.router_logits_similarity_sum = {}
        self.router_logits_similarity_count = {}
        self.expert_output_similarity_sum = {}
        self.expert_output_similarity_count = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            # 仅对 MoE 层注册钩子（请替换为实际的 MoE 层类型）
            if isinstance(module, OlmoeSparseMoeBlock):
                hook = module.register_forward_hook(self._expert_layer_forward_hook(name))
                self.hooks.append(hook)

    def _expert_layer_forward_hook(self, layer_name):
        def hook(module, input, output):
            final_hidden_states, router_logits = output
            batch_size, seq_len, hidden_dim = final_hidden_states.shape
            total_samples = batch_size * seq_len
            num_experts = module.num_experts
            # reshape router_logits为 (total_samples, num_experts)
            router_logits = router_logits.view(-1, num_experts)

            # 对 router_logits 做 softmax，然后取 top_k 专家
            routing_weights_all = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights_all, module.top_k, dim=-1)
            if module.norm_topk_prob:
                routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
            routing_weights = routing_weights.to(final_hidden_states.dtype)
            selected_experts = selected_experts.to(self.device)  # shape: (total_samples, top_k)
            # 初始化本层统计变量（仅第一次调用时）
            if layer_name not in self.expert_counts:
                self.expert_counts[layer_name] = torch.zeros(num_experts, device=self.device)
                self.expert_weights_sum[layer_name] = torch.zeros(num_experts, device=self.device)
                self.router_logits_similarity_sum[layer_name] = {
                    'cosine': torch.zeros((num_experts, num_experts), device=self.device),
                    'l2': torch.zeros((num_experts, num_experts), device=self.device)
                }
                self.router_logits_similarity_count[layer_name] = 0
                self.expert_output_similarity_sum[layer_name] = {
                    'cosine': torch.zeros((num_experts, num_experts), device=self.device),
                    'l2': torch.zeros((num_experts, num_experts), device=self.device)
                }
                self.expert_output_similarity_count[layer_name] = 0

            # 更新专家激活计数：每个样本中如果某个专家出现在 top_k 内，就记一次
            for i in range(total_samples):
                unique_experts = torch.unique(selected_experts[i])
                for expert in unique_experts:
                    self.expert_counts[layer_name][expert] += 1

            # 累加 routing weights：对每个样本，在 top_k 内，每个专家贡献其权重（可能多个位置出现时求和）
            for i in range(total_samples):
                for k in range(routing_weights.shape[1]):
                    expert = selected_experts[i, k]
                    self.expert_weights_sum[layer_name][expert] += routing_weights[i, k].item()
                    
            # 计算当前批次的 router_logits 相似性
            cosine_matrix_logits = torch.zeros((num_experts, num_experts), device=self.device)
            l2_matrix_logits = torch.zeros((num_experts, num_experts), device=self.device)
            for i in range(num_experts):
                for j in range(i, num_experts):
                    cos_sim_logits = F.cosine_similarity(router_logits[:, i], router_logits[:, j], dim=0)
                    cosine_matrix_logits[i, j] = cos_sim_logits
                    cosine_matrix_logits[j, i] = cos_sim_logits
                    l2_dist_logits = torch.norm(router_logits[:, i] - router_logits[:, j], p=2)
                    l2_matrix_logits[i, j] = l2_dist_logits
                    l2_matrix_logits[j, i] = l2_dist_logits
            self.router_logits_similarity_sum[layer_name]['cosine'] += cosine_matrix_logits
            self.router_logits_similarity_sum[layer_name]['l2'] += l2_matrix_logits
            self.router_logits_similarity_count[layer_name] += 1
            
            hidden_states = input[0].view(-1, hidden_dim)
            expert_outputs = []
            for expert_idx in range(num_experts):
                expert_out = torch.zeros(total_samples, hidden_dim, device=self.device, dtype=final_hidden_states.dtype)
                for i in range(total_samples):
                    mask = (selected_experts[i] == expert_idx)
                    if mask.sum() > 0:
                        # 对于样本 i，该专家的贡献为该样本隐藏状态乘以该专家在 top_k 中所有位置的 routing weight之和
                        agg_weight = routing_weights[i][mask].sum()
                        # 模拟专家输出
                        expert_layer = module.experts[expert_idx]
                        current_state = hidden_states[i].unsqueeze(0)
                        current_output = expert_layer(current_state)
                        expert_out[i] = current_output.squeeze(0) * agg_weight
                expert_outputs.append(expert_out.view(-1))  # flatten为向量

            # 对专家输出进行中心化和归一化
            expert_outputs = torch.stack(expert_outputs)
            # 中心化
            expert_outputs = expert_outputs - expert_outputs.mean(dim=1, keepdim=True)
            # 归一化
            expert_outputs = F.normalize(expert_outputs, p=2, dim=1)

            # 计算当前批次的专家输出相似性
            cosine_matrix_output = torch.zeros((num_experts, num_experts), device=self.device)
            l2_matrix_output = torch.zeros((num_experts, num_experts), device=self.device)
            for i in range(num_experts):
                for j in range(i, num_experts):
                    vec_i = expert_outputs[i]
                    vec_j = expert_outputs[j]
                    cos_sim_output = F.cosine_similarity(vec_i.unsqueeze(0), vec_j.unsqueeze(0), dim=1).item()
                    cosine_matrix_output[i, j] = cos_sim_output
                    cosine_matrix_output[j, i] = cos_sim_output
                    l2_dist_output = torch.norm(vec_i - vec_j, p=2)
                    l2_matrix_output[i, j] = l2_dist_output
                    l2_matrix_output[j, i] = l2_dist_output

            self.expert_output_similarity_sum[layer_name]['cosine'] += cosine_matrix_output
            self.expert_output_similarity_sum[layer_name]['l2'] += l2_matrix_output
            self.expert_output_similarity_count[layer_name] += 1

        return hook

    def reset_counts(self):
        self.expert_counts = {}
        self.expert_weights_sum = {}
        self.router_logits_similarity_sum = {}
        self.router_logits_similarity_count = {}
        self.expert_output_similarity_sum = {}
        self.expert_output_similarity_count = {}

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def compute_similarity(self):
        router_logits_similarity = {}
        expert_output_similarity = {}
        for layer in self.router_logits_similarity_sum.keys():
            count_logits = self.router_logits_similarity_count[layer]
            cosine_matrix_logits = (self.router_logits_similarity_sum[layer]['cosine'] / count_logits).tolist()
            l2_matrix_logits = (self.router_logits_similarity_sum[layer]['l2'] / count_logits).tolist()
            router_logits_similarity[layer] = {'cosine': cosine_matrix_logits, 'l2': l2_matrix_logits}

            count_output = self.expert_output_similarity_count[layer]
            cosine_matrix_output = (self.expert_output_similarity_sum[layer]['cosine'] / count_output).tolist()
            l2_matrix_output = (self.expert_output_similarity_sum[layer]['l2'] / count_output).tolist()
            expert_output_similarity[layer] = {'cosine': cosine_matrix_output, 'l2': l2_matrix_output}
        return router_logits_similarity, expert_output_similarity

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()



class RouterLogitsSimilarityWrapper(torch.nn.Module):
    def __init__(self, model, device):
        super(RouterLogitsSimilarityWrapper, self).__init__()
        self.model = model
        self.device = device
        self.router_logits_similarity_sum = {}
        self.router_logits_similarity_count = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, OlmoeSparseMoeBlock):
                hook = module.register_forward_hook(self._expert_layer_forward_hook(name))
                self.hooks.append(hook)

    def _expert_layer_forward_hook(self, layer_name):
        def hook(module, input, output):
            _, router_logits = output
            num_experts = router_logits.shape[-1]
            router_logits = router_logits.view(-1, num_experts)

            cosine_matrix = torch.zeros((num_experts, num_experts), device=self.device)
            l2_matrix = torch.zeros((num_experts, num_experts), device=self.device)

            for i in range(num_experts):
                for j in range(i, num_experts):
                    cos_sim = torch.nn.functional.cosine_similarity(router_logits[:, i], router_logits[:, j], dim=0)
                    cosine_matrix[i, j] = cos_sim
                    cosine_matrix[j, i] = cos_sim
                    l2_dist = torch.norm(router_logits[:, i] - router_logits[:, j], p=2)
                    l2_matrix[i, j] = l2_dist
                    l2_matrix[j, i] = l2_dist

            if layer_name not in self.router_logits_similarity_sum:
                self.router_logits_similarity_sum[layer_name] = {
                    'cosine': cosine_matrix,
                    'l2': l2_matrix
                }
                self.router_logits_similarity_count[layer_name] = 1
            else:
                self.router_logits_similarity_sum[layer_name]['cosine'] += cosine_matrix
                self.router_logits_similarity_sum[layer_name]['l2'] += l2_matrix
                self.router_logits_similarity_count[layer_name] += 1

        return hook

    def reset_counts(self):
        self.router_logits_similarity_sum = {}
        self.router_logits_similarity_count = {}

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def compute_router_logits_similarity(self):
        similarity = {}
        for layer in self.router_logits_similarity_sum.keys():
            count = self.router_logits_similarity_count[layer]
            cosine_matrix = self.router_logits_similarity_sum[layer]['cosine'] / count
            l2_matrix = self.router_logits_similarity_sum[layer]['l2'] / count
            similarity[layer] = {
                'cosine': cosine_matrix.tolist(),
                'l2': l2_matrix.tolist()
            }
        return similarity

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class OutputLogitsSimilarityWrapper(torch.nn.Module):
    def __init__(self, model, device):
        super(OutputLogitsSimilarityWrapper, self).__init__()
        self.model = model
        self.device = device
        self.expert_output_similarity_sum = {}
        self.expert_output_similarity_count = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, OlmoeSparseMoeBlock):
                hook = module.register_forward_hook(self._expert_layer_forward_hook(name))
                self.hooks.append(hook)

    def _expert_layer_forward_hook(self, layer_name):
        def hook(module, input, output):
            final_hidden_states, _ = output
            num_experts = final_hidden_states.shape[-1]
            hidden_dim = final_hidden_states.shape[-2]
            final_hidden_states = final_hidden_states.view(-1, hidden_dim, num_experts)

            cosine_matrix = torch.zeros((num_experts, num_experts), device=self.device)
            l2_matrix = torch.zeros((num_experts, num_experts), device=self.device)

            for i in range(num_experts):
                for j in range(i, num_experts):
                    expert_i_output = final_hidden_states[..., i].view(-1)
                    expert_j_output = final_hidden_states[..., j].view(-1)
                    cos_sim = torch.nn.functional.cosine_similarity(expert_i_output, expert_j_output, dim=0)
                    cosine_matrix[i, j] = cos_sim
                    cosine_matrix[j, i] = cos_sim
                    l2_dist = torch.norm(expert_i_output - expert_j_output, p=2)
                    l2_matrix[i, j] = l2_dist
                    l2_matrix[j, i] = l2_dist

            if layer_name not in self.expert_output_similarity_sum:
                self.expert_output_similarity_sum[layer_name] = {
                    'cosine': cosine_matrix,
                    'l2': l2_matrix
                }
                self.expert_output_similarity_count[layer_name] = 1
            else:
                self.expert_output_similarity_sum[layer_name]['cosine'] += cosine_matrix
                self.expert_output_similarity_sum[layer_name]['l2'] += l2_matrix
                self.expert_output_similarity_count[layer_name] += 1

        return hook

    def reset_counts(self):
        self.expert_output_similarity_sum = {}
        self.expert_output_similarity_count = {}

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def compute_expert_similarity(self):
        similarity = {}
        for layer in self.expert_output_similarity_sum.keys():
            count = self.expert_output_similarity_count[layer]
            cosine_matrix = self.expert_output_similarity_sum[layer]['cosine'] / count
            l2_matrix = self.expert_output_similarity_sum[layer]['l2'] / count
            similarity[layer] = {
                'cosine': cosine_matrix.tolist(),
                'l2': l2_matrix.tolist()
            }
        return similarity

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()



class CacheDataset:
    def __init__(self):
        self.Xs = []
        self.Zs = []

    def append(self, X=None, Z=None):
        if X is not None:
            self.Xs.append(X)
        if Z is not None:
            self.Zs.append(Z)

class PrunableOlmoeSparseMoeBlockWrapper(nn.Module):
    def __init__(self, model, r: Optional[int] = None):
        super().__init__()
        if isinstance(model, OlmoeSparseMoeBlock):
            self.model = model
        else:
            self.model = model.model
        self.r = r
        
        self.experts_to_drop = None
        self.cache_space = CacheDataset()
        self.cache_logits = False
        self.cache_X = False
        self.cache_Z = False

        # 新增属性，用于记录剪枝后保留的原始专家序号
        self.original_expert_indices = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.model.gate(hidden_states)

        print('========router_logits.shape', router_logits.shape)
        print('========self.experts_to_drop', self.experts_to_drop)
        if self.experts_to_drop is not None:
            for e in self.experts_to_drop:
                router_logits[:, e] = -float('inf')

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        
        # Ensure top_k does not exceed the number of experts
        top_k = min(self.model.top_k, routing_weights.size(-1))
        routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        if self.model.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.model.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.model.num_experts):
            expert_layer = self.model.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        if self.experts_to_drop is not None and (self.cache_logits or self.cache_X or self.cache_Z):
            logger.warn(f'Already dropped {self.experts_to_drop} but still storing activations.')
        self.cache_space.append(X=(hidden_states if self.cache_X else None), 
                                Z=(final_hidden_states if self.cache_Z else None))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

    @torch.no_grad()
    def prune_by_threshold(self, experts_importance, threshold):
        # 根据阈值剪枝：丢弃重要性低于阈值的专家
        self.experts_to_drop = [i for i, importance in enumerate(experts_importance) if importance < threshold]
        print('===self.experts_to_drop===', self.experts_to_drop)
        # 保留重要性高于等于阈值的专家
        experts_to_keep = [i for i, importance in enumerate(experts_importance) if importance >= threshold]
        
        if len(experts_to_keep) == 0:
            logger.warning("No experts meet the threshold. Keeping all experts.")
            experts_to_keep = list(range(len(experts_importance)))
        
        # 记录下原始保留的专家序号
        self.original_expert_indices = experts_to_keep
        
        gate_new = nn.Linear(in_features=self.model.gate.in_features,
                             out_features=len(experts_to_keep), bias=False, device='cpu', dtype=torch.bfloat16)
        gate_new.weight.data = self.model.gate.weight.data[list(experts_to_keep)]
        self.model.gate = gate_new

        self.model.experts = nn.ModuleList([self.model.experts[i] for i in experts_to_keep])
        self.model.num_experts = len(experts_to_keep)

        # Adjust top_k if necessary
        self.model.top_k = min(self.model.top_k, self.model.num_experts)

    @torch.no_grad()
    def prune_by_importance(self, experts_importance, num_experts_remain):
        # 固定数量剪枝：根据专家重要性排序，保留 num_experts_remain 个专家
        num_experts_to_keep = num_experts_remain
        importance_scores = experts_importance.copy()
        
        # 按重要性从小到大排序，选出排名靠后的专家保留
        sorted_indices = sorted(range(len(importance_scores)), key=lambda k: importance_scores[k])
        experts_to_keep = sorted_indices[-num_experts_to_keep:]
        self.experts_to_drop = set(range(self.model.num_experts)) - set(experts_to_keep)
        
        # 记录下原始保留的专家序号
        self.original_expert_indices = experts_to_keep
        
        gate_new = nn.Linear(in_features=self.model.gate.in_features,
                             out_features=num_experts_remain, bias=False, device='cpu', dtype=torch.bfloat16)
        gate_new.weight.data = self.model.gate.weight.data[list(experts_to_keep)]
        print('===self.experts_to_drop===', self.experts_to_drop)
        self.model.gate = gate_new

        self.model.experts = nn.ModuleList([self.model.experts[i] for i in experts_to_keep])
        self.model.num_experts = num_experts_remain

class PrunableSVD_OlmoeSparseMoeBlockWrapper(nn.Module):
    def __init__(self, model: 'SVD_OlmoeSparseMoeBlock', r: Optional[int] = None):
        super().__init__()
        if isinstance(model, SVD_OlmoeSparseMoeBlock):
            self.model = model
        else:
            raise ValueError("Model must be an instance of SVD_OlmoeSparseMoeBlock")
        
        self.r = r
        
        self.experts_to_drop = None
        self.cache_space = CacheDataset()
        self.cache_logits = False
        self.cache_X = False
        self.cache_Z = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.model.gate(hidden_states)
        print('router_logits.shape', router_logits.shape)
        print('self.experts_to_drop', self.experts_to_drop)
        if self.experts_to_drop is not None:
            for e in self.experts_to_drop:
                router_logits[:, e] = -float('inf')

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        top_k = min(self.model.top_k, routing_weights.size(-1))
        routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        if self.model.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.model.num_experts).permute(2, 1, 0)
        print('self.model.num_experts', self.model.num_experts)
        for expert_idx in range(self.model.num_experts):
            print('expert_idx', expert_idx)
            if expert_idx in self.experts_to_drop:
                continue

            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = self.model.basenet_forward(current_state, expert_idx) * routing_weights[top_x, idx, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        if self.experts_to_drop is not None and (self.cache_logits or self.cache_X or self.cache_Z):
            logger.warn(f'Already dropped {self.experts_to_drop} but still storing activations.')
        self.cache_space.append(X=(hidden_states if self.cache_X else None), Z=(final_hidden_states if self.cache_Z else None))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

    @torch.no_grad()
    def prune_by_threshold(self, experts_importance, threshold: float, pruned_to_original_list):
        # Identify experts to drop based on the importance threshold
        self.experts_to_drop = [i for i, importance in enumerate(experts_importance) if importance < threshold]

        # Keep only the experts that are above the threshold
        experts_to_keep = [i for i, importance in enumerate(experts_importance) if importance >= threshold]
        
        if len(experts_to_keep) == 0:
            logger.warning("No experts meet the threshold. Keeping all experts.")
            experts_to_keep = list(range(len(experts_importance)))

        gate_new = nn.Linear(in_features=self.model.gate.in_features,
                             out_features=len(experts_to_keep), bias=False, device='cpu', dtype=torch.bfloat16)
        gate_new.weight.data = self.model.gate.weight.data[list(experts_to_keep)]
        self.model.gate = gate_new

        self.model.experts = nn.ModuleList([self.model.experts[i] for i in experts_to_keep])
        self.model.num_experts = len(experts_to_keep)

        # Adjust top_k if necessary
        self.model.top_k = min(self.model.top_k, self.model.num_experts)

        # 更新 pruned_to_original_list 列表
        pruned_to_original_list.clear()  # 清空列表以便重新填充
        for new_idx, old_idx in enumerate(experts_to_keep):
            pruned_to_original_list.append(old_idx)  # 新索引对应旧索引
# class PrunableSVD_OlmoeSparseMoeBlockWrapper(nn.Module):
#     def __init__(self, model: SVD_OlmoeSparseMoeBlock, r: Optional[int] = None):
#         super().__init__()
#         if isinstance(model, SVD_OlmoeSparseMoeBlock):
#             self.model = model
#         else:
#             raise ValueError("Model must be an instance of SVD_OlmoeSparseMoeBlock")
        
#         self.r = r
        
#         self.experts_to_drop = None
#         self.cache_space = CacheDataset()
#         self.cache_logits = False
#         self.cache_X = False
#         self.cache_Z = False

#     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         batch_size, sequence_length, hidden_dim = hidden_states.shape
#         hidden_states = hidden_states.view(-1, hidden_dim)
#         router_logits = self.model.gate(hidden_states)

#         if self.experts_to_drop is not None:
#             for e in self.experts_to_drop:
#                 router_logits[:, e] = -float('inf')

#         routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        
#         # Ensure top_k does not exceed the number of experts
#         top_k = min(self.model.top_k, routing_weights.size(-1))
#         routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
#         if self.model.norm_topk_prob:
#             routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
#         routing_weights = routing_weights.to(hidden_states.dtype)

#         final_hidden_states = torch.zeros(
#             (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
#         )

#         expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.model.num_experts).permute(2, 1, 0)

#         for expert_idx in range(self.model.num_experts):
#             if expert_idx in self.experts_to_drop:
#                 continue

#             idx, top_x = torch.where(expert_mask[expert_idx])

#             if top_x.shape[0] == 0:
#                 continue

#             current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
#             current_hidden_states = self.model.basenet_forward(current_state, expert_idx) * routing_weights[top_x, idx, None]

#             final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

#         if self.experts_to_drop is not None and (self.cache_logits or self.cache_X or self.cache_Z):
#             logger.warn(f'Already dropped {self.experts_to_drop} but still storing activations.')
#         self.cache_space.append(X=(hidden_states if self.cache_X else None), Z=(final_hidden_states if self.cache_Z else None))

#         final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
#         return final_hidden_states, router_logits

#     @torch.no_grad()
#     def prune_by_threshold(self, experts_importance, threshold: float, pruned_to_original_list):
#         # Identify experts to drop based on the importance threshold
#         self.experts_to_drop = [i for i, importance in enumerate(experts_importance) if importance < threshold]

#         # Keep only the experts that are above the threshold
#         experts_to_keep = [i for i, importance in enumerate(experts_importance) if importance >= threshold]
        
#         if len(experts_to_keep) == 0:
#             logger.warning("No experts meet the threshold. Keeping all experts.")
#             experts_to_keep = list(range(len(experts_importance)))

#         gate_new = nn.Linear(in_features=self.model.gate.in_features,
#                              out_features=len(experts_to_keep), bias=False, device='cpu', dtype=torch.bfloat16)
#         gate_new.weight.data = self.model.gate.weight.data[list(experts_to_keep)]
#         self.model.gate = gate_new

#         self.model.experts = nn.ModuleList([self.model.experts[i] for i in experts_to_keep])
#         self.model.num_experts = len(experts_to_keep)

#         # Adjust top_k if necessary
#         self.model.top_k = min(self.model.top_k, self.model.num_experts)

#         # 更新 pruned_to_original_list 列表
#         pruned_to_original_list.clear()  # 清空列表以便重新填充
#         for new_idx, old_idx in enumerate(experts_to_keep):
#             pruned_to_original_list.append(old_idx)  # 新索引对应旧索引



class OlmoeSparseMoeBlock_Router_logits(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([OlmoeMLP_Router_logits(config) for _ in range(self.num_experts)])
        # 用于保存每个 expert 的 routing 信息：映射 expert_idx -> (token_indices, routing_weights)
        self.expert_routing_info = {}

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)  # shape: (B*L, hidden_dim)
        # 计算路由 logits
        router_logits = self.gate(hidden_states)  # (B*L, num_experts)
        # softmax 得到所有专家的概率分布
        routing_all = F.softmax(router_logits, dim=1)
        # 取 top_k
        routing_topk, selected_experts = torch.topk(routing_all, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_topk = routing_topk / routing_topk.sum(dim=-1, keepdim=True)
        routing_topk = routing_topk.to(hidden_states.dtype)

        # 初始化输出
        final_hidden_states = torch.zeros_like(hidden_states)

        # 为后续保存 routing 信息先清空
        self.expert_routing_info = {}

        # 构造 one-hot mask，形状 (num_experts, total_tokens, top_k)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        # 遍历每个 expert
        for expert_idx in range(self.num_experts):
            # 找到该 expert 在 top_k 中被选中的 token 索引和对应在 top_k 内的位置
            # token_idx: 当前 expert 被分配到的 token 的索引，pos_in_topk: 在 topk 维度内的位置
            pos_in_topk, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue
            # 从 routing_topk 中提取对应 token 的权重，形状: (num_selected,)
            current_routing_weight = routing_topk[token_idx, pos_in_topk]  
            # 保存 routing 信息，用于后续 profiling
            self.expert_routing_info[expert_idx] = (token_idx, current_routing_weight)
            # 取出当前 expert 对应的输入 token，形状: (num_selected, hidden_dim)
            current_state = hidden_states[token_idx]
            # 调用 expert 时传递 routing_weight（注意 expert 内部的 forward 会将该值传递到各子模块）
            current_hidden_states = self.experts[expert_idx](current_state, routing_weight=current_routing_weight)
            # 累加到最终输出中（index_add_ 要求 token_idx 为一维索引）
            final_hidden_states.index_add_(0, token_idx, current_hidden_states)
        final_hidden_states = final_hidden_states.view(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class OlmoeMLP_Router_logits(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    # 修改 forward 接收 routing_weight 参数
    def forward(self, x, routing_weight=None):
        # 如果传入 routing_weight，则将其保存到各个子模块上，
        # 便于后续 hook 在计算统计时使用（注意：此 routing_weight 的 shape 应为 (N,) 对应当前 expert 的 token 数量）
        if routing_weight is not None:
            self.gate_proj.routing_weight = routing_weight
            self.up_proj.routing_weight = routing_weight
            self.down_proj.routing_weight = routing_weight
        # 标准计算流程
        hidden = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(hidden)
