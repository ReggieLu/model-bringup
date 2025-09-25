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

import os

import hydra
from accelerate import init_empty_weights
from accelerate.utils import set_seed
from sambanova_modelzoo.generation.cached_inference_compiler import CachedInferenceCompiler
from sambanova_modelzoo.generation.configuration_utils import get_config_overrides_for_generation
from sambanova_modelzoo.generation.fused_lm_head_compiler import FusedLMHeadCompiler
from sambanova_modelzoo.perfsdk.performance_heuristic import PerformanceHeuristic
from sambanova_modelzoo.schema.arguments import to_pydantic
from sambanova_modelzoo.schema.generation_test_schema import RDUGenerationAppConfig
from sambanova_modelzoo.utils import load_model_from_config

import sambaflow.samba as samba
from sambaflow.frameworks.quantized.quantized_linear import SNQConfig, quantize_model

# monkey patch this to sambanova_modelzoo.models.<model_name>
import sys
import types
from <model_name> import __init__ as <model_name>
from <model_name> import trace_<model_name>, plugin_heuristics_<model_name>, plugins_<model_name>
parent_module_name = 'sambanova_modelzoo.models'
target_package = f'{parent_module_name}.<model_name>'
parent_module = types.ModuleType(parent_module_name)
sys.modules[parent_module_name] = parent_module
sys.modules[target_package] = <model_name>
setattr(sys.modules[parent_module_name], '<model_name>', <model_name>)

# Register submodules
submodules = {
    'trace_<model_name>': trace_<model_name>,
    'plugin_heuristics_<model_name>': plugin_heuristics_<model_name>,
    'plugins_<model_name>': plugins_<model_name>,
}

for name, module in submodules.items():
    full_name = f'{target_package}.{name}'
    sys.modules[full_name] = module

# Optional: Set __path__ for package resolution
if hasattr(<model_name>, '__path__'):
    sys.modules[target_package].__path__ = <model_name>.__path__


def compile(cfg: RDUGenerationAppConfig) -> str:
    """
    Compile the model
    Args:
        cfg: Parsed Pydantic model from yaml file
    Returns:
        Compiled pef file name
    """
    original_config_overrides = get_config_overrides_for_generation()
    with init_empty_weights():
        sn_model = load_model_from_config(cfg.generation.model_name_or_path, cfg.model, original_config_overrides)

    qconfig = getattr(sn_model.config, 'quantization_config', {})
    if qconfig:
        quantization_config = SNQConfig(**qconfig)
        sn_model = quantize_model(sn_model, quantization_config)
    sn_model.eval()

    if cfg.generation.fuse_lm_head_with_postprocess:
        compiler = FusedLMHeadCompiler(
            sn_model,
            cfg.generation.batch_size,
            set(cfg.generation.static_seq_lengths),
            cfg.generation.seg_softmax_block_sizes,
            **cfg.cb.get_constructor_args(),
        )
    else:
        compiler = CachedInferenceCompiler(
            sn_model,
            cfg.generation.batch_size,
            set(cfg.generation.static_seq_lengths),
            cfg.generation.seg_softmax_block_sizes,
            cfg.generation.spec_decoding_k + 1,
            **cfg.cb.get_constructor_args(),
        )

    return compiler.compile(cfg=cfg.get_compile_args())


@hydra.main(config_path="configs", config_name="o0")
@to_pydantic(RDUGenerationAppConfig)
def main(cfg: RDUGenerationAppConfig):
    set_seed(cfg.generation.seed)

    if cfg.command == 'compile':
        # If using PerfSDK heuristic, do init here.
        if (heuristic := cfg.samba_compile.perfsdk_heuristic_name) is not None:
            PerformanceHeuristic.register(heuristic)

        samba.session.setup(cfg.samba_compile)
        compile(cfg)
    else:
        raise NotImplementedError("text_generation_compile.py only supports the 'compile' command.")


if __name__ == '__main__':
    main()
