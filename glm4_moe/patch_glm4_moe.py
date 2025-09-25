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
import torch
from torch import nn
from transformers.activations import ACT2FN

class SNGlm4MoeForCausalLMPatch:
   @staticmethod
   def patch_init(self, config):
       self.prepare_inputs_for_generation = self.sn_prepare_inputs_for_generation

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
        #self.rotary_emb = SNRotaryEmbeddingLlamaLike(config=config)
        
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

        #self.self_attn = SNAttentionLlamaLike(config=config, layer_idx=layer_idx, custom_post_rope_qk_norm = qk_norm)
        self.input_layernorm = SNRMSNormLlamaLike(config.hidden_size, eps=config.rms_norm_eps, fp32_ln = config.fp32_ln, config = config)
        self.post_attention_layernorm = SNRMSNormLlamaLike(config.hidden_size, eps=config.rms_norm_eps, fp32_ln = config.fp32_ln, config = config)

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
       topk_indices, topk_weights = self.gate(hidden_states)
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
