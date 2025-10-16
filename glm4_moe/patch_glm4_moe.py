from sambaflow.samba.utils import parse_dtype
from sambanova_modelzoo.directives import named_subgraph
from sambanova_modelzoo.modules.cache_utils import SNStaticCache
from sambanova_modelzoo.modules.rms_norm import SNRMSNormLlamaLike
from sambanova_modelzoo.modules.rotary_embedding import SNRotaryEmbeddingLlamaLike
from sambanova_modelzoo.modules.attention import SNAttentionLlamaLike, AttentionType
from sambanova_modelzoo.modules.embedding import SNEmbeddingLlamaLike
from sambanova_modelzoo.modules.classifier import SNClassifierLlamaLike
from sambanova_modelzoo.modules.sparse_moe import SNSparseMoEBase

from sambanova_modelzoo.custom_ops import topk_streaming, sn_reduce, sn_zipmapreduce, sn_select, sn_imm, sn_iteridx, sn_embedding
from sambanova_modelzoo.modeling_patch_utils import MASK_MIN_VALUE, finfo_float32_min_patch
from sambanova_modelzoo.utils import named_tensor

from .glm4_moe_configuration import SNGlm4MoeConfig

from typing import Optional, Union, Unpack
import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.utils import TransformersKwargs
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

from .hyperfunction_glm4_moe import Glm4MoeHyperfunction

def sn_patch_module_add_hyperfunction(self, config, **kwargs):
    self.hyperfunction = Glm4MoeHyperfunction(config)
    
class SNGlm4MoeForCausalLMPatch:
   @staticmethod
   def patch_init(self, config):
       self.prepare_inputs_for_generation = self.sn_prepare_inputs_for_generation
   @staticmethod
   def forward(
          self,
          input_ids: Optional[torch.LongTensor] = None,
          attention_mask: Optional[torch.Tensor] = None,
          position_ids: Optional[torch.LongTensor] = None,
          past_key_values: Optional[Cache] = None,
          inputs_embeds: Optional[torch.FloatTensor] = None,
          labels: Optional[torch.LongTensor] = None,
          use_cache: Optional[bool] = None,
          cache_position: Optional[torch.LongTensor] = None,
          logits_to_keep: Union[int, torch.Tensor] = 0,
          **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
       outputs: BaseModelOutputWithPast = self.model(
          input_ids=input_ids,
          attention_mask=attention_mask,
          position_ids=position_ids,
          past_key_values=past_key_values,
          inputs_embeds=inputs_embeds,
          use_cache=use_cache,
          cache_position=cache_position,
          **kwargs,
       )
       
       hidden_states = outputs[0]
       # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
       # slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
       # logits = self.lm_head(hidden_states[:, slice_indices, :])

       # loss = None
       # if labels is not None:
       #    loss = self.loss_function(logits=logits, labels=labels, vocab_size
       # =self.config.vocab_size, **kwargs)
       batch_size, seq_length = input_ids.shape if input_ids is not None else inputs_embeds.shape[:2]
       consume_cache = use_cache and past_key_values is not None and any(past_key_values.is_updated)
       sn_classifier = SNClassifierLlamaLike(self.config, self.hyperfunction.classifier, seq_length, consume_cache, batch_size, self.training)
       logits, loss = sn_classifier.compute_logits_and_loss(hidden_states, self.lm_head, labels)
       output = (logits,) + outputs[1:]
       return (loss,) + output if loss is not None else output
     # return CausalLMOutputWithPast(
     #    loss=loss,
     #    logits=logits,
     #    past_key_values=outputs.past_key_values,
     #    hidden_states=outputs.hidden_states,
     #    attentions=outputs.attentions,
     # )
     
class Glm4MoePreTrainedModelPatch:
    @staticmethod
    def patch_init_weight_(self, module):
        pass

class SNGlm4MoeModelPatch:
    # we want path for norm
    @staticmethod
    def patch__init__(self, config : SNGlm4MoeConfig):
        self.embed_tokens = sn_embedding(config.vocab_size, config.hidden_size, self.padding_idx, off_chip=True)
        self.norm = SNRMSNormLlamaLike(config.hidden_size, eps=config.rms_norm_eps, fp32_ln = config.fp32_ln, config = config)
        self.rotary_emb = SNRotaryEmbeddingLlamaLike(config=config)

    @staticmethod
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # if inputs_embeds is None:
        #     inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = SNStaticCache(self.config)

        # if cache_position is None:
        #     past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        #     cache_position: torch.Tensor = torch.arange(
        #         past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        #     )

        # if position_ids is None:
        #     position_ids = cache_position.unsqueeze(0)

        # causal_mask = create_causal_mask(
        #     config=self.config,
        #     input_embeds=inputs_embeds,
        #     attention_mask=attention_mask,
        #     cache_position=cache_position,
        #     past_key_values=past_key_values,
        #     position_ids=position_ids,
        # )

        # hidden_states = inputs_embeds
        # position_embeddings = self.rotary_emb(hidden_states, position_ids)

        batch_size, seq_length = input_ids.shape if input_ids is not None else inputs_embeds.shape[:2]
        consume_cache = use_cache and past_key_values is not None and any(past_key_values.is_updated)
        sn_embedding_module = SNEmbeddingLlamaLike(self.config, self.hyperfunction.embedding, seq_length, consume_cache, batch_size, self.training, self.rotary_emb)
        hidden_states, attention_mask, positional_embeddings = sn_embedding_module.forward(cache_position, input_ids, self.embed_tokens, position_ids, inputs_embeds, attention_mask)
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                cache_position=cache_position,
                position_embeddings=positional_embeddings,
                use_cache = use_cache,
                **kwargs,
            )

        sn_classifier = SNClassifierLlamaLike(self.config, self.hyperfunction.classifier, seq_length, consume_cache, batch_size, self.training)
        hidden_states = sn_classifier.apply_normalization(hidden_states, cache_position, self.norm)
       # hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

class Glm4MoeDecoderLayerPatch:
    # we want patch for attention, mop, moe, norm
    @staticmethod
    def patch__init__(self, config : SNGlm4MoeConfig, layer_idx : int):
        qk_norm = None
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        if config.use_qk_norm:
            q_norm = SNRMSNormLlamaLike(head_dim, eps=config.rms_norm_eps, fp32_ln = config.fp32_ln, config = config)
            k_norm = SNRMSNormLlamaLike(head_dim, eps=config.rms_norm_eps, fp32_ln = config.fp32_ln, config = config)
            qk_norm = (q_norm, k_norm)

        self.self_attn = SNAttentionLlamaLike(config=config, layer_idx=layer_idx, custom_post_rope_qk_norm = qk_norm)
        self.input_layernorm = SNRMSNormLlamaLike(config.hidden_size, eps=config.rms_norm_eps, fp32_ln = config.fp32_ln, config = config)
        self.post_attention_layernorm = SNRMSNormLlamaLike(config.hidden_size, eps=config.rms_norm_eps, fp32_ln = config.fp32_ln, config = config)

    @staticmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        if self.config.fp32_skip_add:
            residual = residual.float()
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = hidden_states.to(parse_dtype(self.config.param_dtype))
        # Self Attention
        hidden_states, _, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            # position_ids=position_ids,
            past_key_values=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            attention_type = AttentionType.Full,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        if self.config.fp32_skip_add:
            residual = residual.float()
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states.to(parse_dtype(self.config.param_dtype))
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

        
class Glm4MoeExperts(SNSparseMoEBase):
   def __init__(self, config):
      super().__init__()
      self.config = config
      self.hidden_size = config.hidden_size
      self.intermediate_size = config.moe_intermediate_size
      self.num_experts = config.n_routed_experts
      self.expert_dim = self.intermediate_size
      self.gate_proj = nn.Parameter(torch.empty(self.num_experts, self.expert_dim, self.hidden_size))
      self.up_proj = nn.Parameter(torch.empty(self.num_experts, self.expert_dim, self.hidden_size))
      self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size,  self.expert_dim))
      self.num_experts_per_tok = config.num_experts_per_tok
      self.use_bias = False
      self.limit = getattr(config, 'limit', float('inf'))

   def _compute_activation(self, gate_output, up_output):
      act_fn = ACT2FN[self.config.hidden_act]
      return act_fn(gate_output) * up_output

   def forward(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor = None, topk_weights: torch.Tensor = None):
      batch_size, sequence_length, hidden_dim = hidden_states.shape
      if sequence_length == 1:
         next_states, mask = self._sparse_moe_token_gen(hidden_states, topk_weights, topk_indices)
      else:
         next_states, mask = self._sparse_moe_cache_gen(batch_size, sequence_length, hidden_states,
                                                        hidden_dim, topk_weights, topk_indices)
      return next_states
        
class Glm4MoeMoEPatch:
    # create extra dimenstion for experts using matmul,
    # glm4moetopkroute
   @staticmethod
   def patch__init__(self, config):
        self.experts = Glm4MoeExperts(config)
   @staticmethod
   def patch_forward(self, hidden_states):
       residuals = hidden_states
       orig_shape = hidden_states.shape
       hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
       topk_indices, topk_weights = self.gate(hidden_states)
       hidden_states = hidden_states.view(orig_shape)
       topk_indices = topk_indices.view(orig_shape[0], topk_indices.shape[0]//orig_shape[0], topk_indices.shape[1])
       topk_weights = topk_weights.view(orig_shape[0], topk_weights.shape[0]//orig_shape[0], topk_weights.shape[1])
       #hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
       #hidden_states = self.moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)
       hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
       hidden_states = hidden_states + self.shared_experts(residuals)
       return hidden_states

class Glm4MoeTopkRouterPatch:
    @staticmethod
    def patch__init__(self, config):
        self.e_score_correction_bias = nn.Parameter(torch.zeros((self.n_routed_experts), dtype=torch.float32))
