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
        if config.use_qk_norm:
            self.q_norm = SNRMSNormLlamaLike(config.head_dim, eps=config.rms_norm_eps, fp32_ln = config.fp32_ln, config = config)
            self.k_norm = SNRMSNormLlamaLike(config.head_dim, eps=config.rms_norm_eps, fp32_ln = config.fp32_ln, config = config)
            qk_norm = (self.q_norm, self.k_norm)

        self.self_attn = SNAttentionLlamaLike(config=config, layer_idx=layer_idx, custom_post_rope_qk_norm = qk_norm)
        self.input_layernorm = SNRMSNormLlamaLike(config.hidden_size, eps=config.rms_norm_eps, fp32_ln = config.fp32_ln, config = config)
        self.post_attention_layernorm = SNRMSNormLlamaLike(config.hidden_size, eps=config.rms_norm_eps, fp32_ln = config.fp32_ln, config = config)

class Glm4MoeExperts(SNSparseMoEBase):
    def __init__(self, config):
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.num_experts = config.n_routed_experts
        self.expert_dim = self.intermediate_size
        self.gate_proj = nn.Parameter(torch.empty(self.num_experts, self.expert_dim, self.hidden_size))
        self.up_proj = nn.Parameter(torch.empty(self.num_experts, self.expert_dim, self.hidden_size))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size,  self.expert_dim))
        self.num_experts_per_tok = config.num_experts_per_tok
        self.use_bias = False
        self.limit = getattr(config, 'limit', float('inf'))
        
class Glm4MoeMoEPatch:
    # create extra dimenstion for experts using matmul,
    # glm4moetopkroute
    @staticmethod
    def patch__init__(self, config):
        self.experts = Glm4MoeExperts(config)


class Glm4MoeTopkRouterPatch:
    @staticmethod
    def patch__init__(self, config):
        self.e_score_correction_bias = nn.Parameter(torch.zeros((self.n_routed_experts), dtype=torch.float32))
