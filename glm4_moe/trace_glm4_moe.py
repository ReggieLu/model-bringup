from sambanova_modelzoo.config import SNPretrainedConfig
from sambanova_modelzoo.generation.clm_tracer import CachedInferenceTracer, PretrainTracer
from .modeling_glm4_moe import SNGlm4MoeForCausalLM

class Glm4MoeTracer(CachedInferenceTracer, model = SNGlm4MoeForCausalLM):

    def __init__(self, config, is_continuous_batching = False, token_gen_seq_length=1):
        super().__init__(config, is_continuous_batching, token_gen_seq_length)

class Glm4MoePretrainTracer(PretrainTracer, model = SNGlm4MoeForCausalLM):
    pass
