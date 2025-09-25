# Copyright © SambaNova Systems, Inc. Disclosure, reproduction,
# reverse engineering, or any other use made without the advance written
# permission of SambaNova Systems, Inc. is unauthorized and strictly
# prohibited. All rights of ownership and enforcement are reserved.

# Copyright © SambaNova Systems, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Type

from sambanova_modelzoo.config import SNPretrainedConfig
from sambanova_modelzoo.configuration_transformer import ConfigurationTransformerPlugin
from sambanova_modelzoo.generation.clm_runtime import CachedInferenceRuntime
from sambanova_modelzoo.model_loader import ModelLoaderPlugin
from glm4_moe.glm4_moe_configuration import SNGlm4MoeConfig
from glm4_moe.modeling_glm4_moe import SNGlm4MoeForCausalLM, SNGlm4MoeModel
from transformers import AutoModel, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from transformers.models.glm4_moe.configuration_glm4_moe import Glm4MoeConfig
from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeForCausalLM, Glm4MoeModel


class Glm4MoeConfigurationTransformer(ConfigurationTransformerPlugin):
    def get_source_conversion_type(self) -> Type[PretrainedConfig]:
        return Glm4MoeConfig

    def get_target_conversion_type(self) -> Type[SNPretrainedConfig]:
        return SNGlm4MoeConfig

    def get_architectures_transform_map(self) -> Dict[Type[PreTrainedModel], Type[PreTrainedModel]]:
        return {
            Glm4MoeForCausalLM: SNGlm4MoeForCausalLM,
            Glm4MoeModel: SNGlm4MoeModel,
        }


class Glm4MoeModelLoaderPlugin(ModelLoaderPlugin):
    def get_automodel_map(self) -> Dict[Type[PreTrainedModel], _BaseAutoModelClass]:
        return {
            SNGlm4MoeModel: AutoModel,
            SNGlm4MoeForCausalLM: AutoModelForCausalLM,
        }

    def get_config_type(self) -> Type[SNPretrainedConfig]:
        return SNGlm4MoeConfig


class Glm4MoeRuntime(CachedInferenceRuntime, model=SNGlm4MoeForCausalLM):
    pass
