from transformers.models.glm4_moe.configuration_glm4_moe import Glm4MoeConfig
from sambanova_modelzoo.config import SNPretrainedConfig

class SNGlm4MoeConfig(Glm4MoeConfig, SNPretrainedConfig):
    model_type = "snglm4_moe"
    def __init__(self, **kwargs):
        SNPretrainedConfig.init_superclasses(subclass_self=self, kwargs_dict=kwargs)
        self.q_proj_attention_bias = self.attention_bias
        self.k_proj_attention_bias = self.attention_bias
        self.v_proj_attention_bias = self.attention_bias
