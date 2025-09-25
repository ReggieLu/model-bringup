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

from typing import Dict

import pytest
import torch
from accelerate import init_empty_weights
from accelerate.utils import set_seed

from transformers.cache_utils import DynamicCache
from transformers.models.glm4_moe.configuration_glm4_moe import Glm4MoeConfig
from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeForCausalLM

from sambaflow import samba
from sambaflow.samba.nn.parameter import SambaParameter
from sambanova_modelzoo.generation.cached_inference_compiler import CachedInferenceCompiler
from sambanova_modelzoo.generation.configuration_utils import get_config_overrides_for_generation
from sambanova_modelzoo.testing.model_testing_utils import check_region_sharing, fp64_ops, get_errors
from sambanova_modelzoo.modules.cache_utils import SNStaticCache



# monkey patch this to sambanova_modelzoo.models.glm4_moe
import sys
import types
from glm4_moe import __init__ as glm4_moe
from glm4_moe import trace_glm4_moe, plugin_heuristics_glm4_moe, plugins_glm4_moe
parent_module_name = 'sambanova_modelzoo.models'
target_package = f'{parent_module_name}.glm4_moe'
parent_module = types.ModuleType(parent_module_name)
sys.modules[parent_module_name] = parent_module
sys.modules[target_package] = glm4_moe
setattr(sys.modules[parent_module_name], 'glm4_moe', glm4_moe)

# Register submodules
submodules = {
    'trace_glm4_moe': trace_glm4_moe,
    'plugin_heuristics_glm4_moe': plugin_heuristics_glm4_moe,
    'plugins_glm4_moe': plugins_glm4_moe,
}

for name, module in submodules.items():
    full_name = f'{target_package}.{name}'
    sys.modules[full_name] = module

# Optional: Set __path__ for package resolution
if hasattr(glm4_moe, '__path__'):
    sys.modules[target_package].__path__ = glm4_moe.__path__

from glm4_moe.glm4_moe_configuration import SNGlm4MoeConfig
from glm4_moe.modeling_glm4_moe import SNGlm4MoeForCausalLM
    
##################
# Util functions #
##################
def checkpoint_conversion_moe(state_dict: Dict[str, torch.Tensor], n_experts: int, expert_root: str):
    """For a single Qwen3 MoE block, applies state_dict conversion to pack the expert weights. The input state_dict
    will be modified with the packed weights

    Args:
        state_dict: the state dict
        n_experts: number of experts
        expert_root: root string for the Qwen3MoeSparseMoeBlock module. If not "", should contain the "." at the end

    Returns:
        the new state_dict with packed weights
    """
    packed_weights1 = []
    packed_weights2 = []
    packed_weights3 = []
    for i in range(n_experts):
        packed_weights1.append(state_dict[f"{expert_root}experts.{i}.gate_proj.weight"])
        packed_weights2.append(state_dict[f"{expert_root}experts.{i}.down_proj.weight"])
        packed_weights3.append(state_dict[f"{expert_root}experts.{i}.up_proj.weight"])
        del state_dict[f"{expert_root}experts.{i}.gate_proj.weight"]
        del state_dict[f"{expert_root}experts.{i}.down_proj.weight"]
        del state_dict[f"{expert_root}experts.{i}.up_proj.weight"]

    state_dict[f"{expert_root}experts.gate_proj"] = torch.stack(packed_weights1)
    state_dict[f"{expert_root}experts.down_proj"] = torch.stack(packed_weights2)
    state_dict[f"{expert_root}experts.up_proj"] = torch.stack(packed_weights3)
    

    return state_dict


def checkpoint_conversion(state_dict: Dict[str, torch.Tensor], config):
    """Converts a Qwen3MoeForCausalLM state_dict and adds the expert weight packing

    Args:
        state_dict: the state dict
        config: the config
    """
    for i in range(config.first_k_dense_replace, config.num_hidden_layers):
        state_dict = checkpoint_conversion_moe(state_dict, config.n_routed_experts, f"model.layers.{i}.mlp.")
    return state_dict


def module_init_random_param(module: torch.nn.Module):
    """
    Initialize the module with random parameters
    """
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(torch.rand_like(param))

######################
# Test SN Model Init #
######################
def test_model_init():
    config = Glm4MoeConfig(num_hidden_layers=2,
                           max_seq_length=256,
                           max_position_embeddings=2,
                           hidden_size=480,
                           vocab_size=1024)
    sn_default_config_overrides = get_config_overrides_for_generation()

    # [TODO]
    # Pass in config and sn_default_config_overrides to initialize the model here
    sn_config = SNGlm4MoeConfig.create(sn_args= sn_default_config_overrides, original_config = config)
    sn_model = SNGlm4MoeForCausalLM(sn_config)

    for p in sn_model.parameters():
        assert isinstance(p, torch.nn.Parameter)

###################
# Test simple e2e #
###################
@fp64_ops
def test_simple_e2e():
    torch.use_deterministic_algorithms(True)
    set_seed(256)
    config = Glm4MoeConfig(num_hidden_layers=2,
                           max_seq_length=256,
                           hidden_size=192,
                           vocab_size=1024,
                           intermediate_size=64,
                           use_cache=False,
                           use_qk_norm=True)
    ipts = torch.Tensor([[10, 100, 42, 12, 24, 6, 32, 72, 64, 10]]).long()  # sequence length 10
    hf_model = Glm4MoeForCausalLM(config)
    hf_model.eval()

    hf_out = hf_model(ipts)

    state_dict = hf_model.state_dict()
    sn_args = get_config_overrides_for_generation()
    sn_args.update({
        'fp32_ln': True,
        'fp32_logits': True,
        'max_seq_length': 256,
        'param_dtype': 'float32',
    })

    # [TODO]
    # Pass in config and sn_default_config_overrides to initialize the model here
    sn_config = SNGlm4MoeConfig.create(sn_args= sn_args, original_config = config)
    sn_model = SNGlm4MoeForCausalLM(sn_config)
    sn_model.eval()

    # sn_state_dict = checkpoint_conversion(state_dict, config)
    sn_state_dict = state_dict

    sn_model.load_state_dict(sn_state_dict)
    sn_out = sn_model(ipts)

    diff, rmse, rmse_ratio, max_ulp_diff = get_errors(sn_out[0], hf_out[0])

    assert rmse_ratio < 1e-6


# ################
# # Test tracing #
# ################
# def test_trace_cached_inference():
#     config = Glm4MoeConfig(num_hidden_layers=2,
#                            max_seq_length=256,
#                            hidden_size=192,
#                            vocab_size=1024,
#                            use_qk_norm=True)
#     batch_size = 2
#     sn_args = get_config_overrides_for_generation()

#     sn_args.update({
#         'mixedp_attn': True,
#         'fp32_logits': True,
#         'fp32_skip_add': True,
#         'lazy_init': True,
#         'max_seq_length': 256,
#         'param_dtype': 'bfloat16',
#     })

#     # [TODO]
#     # Pass in config and sn_default_config_overrides to initialize the model here
#     sn_config = None

#     with init_empty_weights():
#         sn_model = None

#     sn_model.eval()
#     cic = CachedInferenceCompiler(sn_model, batch_size)
#     cic.compile(arch='sn40', samba_only=True)
#     assert samba.session.output_dict

#     # For tensors that share memory, check that they share the same region name as well
#     check_region_sharing(cic)
#     samba.session.reset()

# ############################
# # Test numeric correctness #
# ############################

# @fp64_ops
# def test_cache_gen_graph():
#     torch.use_deterministic_algorithms(True)
#     set_seed(256)
    
#     config = Glm4MoeConfig(num_hidden_layers=2,
#                            max_seq_length=256,
#                            hidden_size=192,
#                            vocab_size=1024,
#                            use_qk_norm=True)
#     input_ids = torch.tensor([[1, 6] * 128])
#     hf_model = Glm4MoeForCausalLM(config)
#     hf_model.eval()
#     hf_out = hf_model(input_ids=input_ids, use_cache=True)
#     state_dict = hf_model.state_dict()

#     sn_args = get_config_overrides_for_generation()
#     sn_args.update({
#         'fp32_ln': True,
#         'fp32_logits': True,
#         'max_seq_length': 256,
#         'param_dtype': 'float32',
#     })

#     # [TODO]
#     # Pass in config and sn_default_config_overrides to initialize the model here
#     sn_config = None
#     sn_model = None

#     sn_state_dict = checkpoint_conversion(state_dict, config)
#     sn_model.load_state_dict(sn_state_dict)
#     sn_model.eval()

#     kv_cache = SNStaticCache(sn_config)
#     sn_out = sn_model(input_ids=input_ids, use_cache=True, past_key_values=kv_cache)
    
#     diff, rmse, rmse_ratio, max_ulp_diff = get_errors(hf_out[0].data, sn_out[0].data)

#     assert rmse_ratio < 1e-6

# @fp64_ops
# def test_token_gen_graph():
#     torch.use_deterministic_algorithms(True)
#     set_seed(256)
    
#     num_hidden_layers = 2
#     config = Glm4MoeConfig(num_hidden_layers=num_hidden_layers,
#                            max_seq_length=256,
#                            hidden_size=192,
#                            vocab_size=1024,
#                            use_qk_norm=True)
    
#     input_ids = torch.tensor([[10]])
#     key_values = torch.randn(1, 8, 1, 2, dtype=torch.float32)

#     complete_key_values = torch.cat([key_values, key_values], dim=-2)
#     attention_mask = torch.tensor([[1, 1]])
#     hf_model = Glm4MoeForCausalLM(config)
#     hf_model.eval()
#     hf_out = hf_model(input_ids=input_ids,
#                       attention_mask=attention_mask,
#                       past_key_values=DynamicCache.from_legacy_cache([[key_values, key_values], [key_values, key_values]]),
#                       use_cache=True)
#     state_dict = hf_model.state_dict()

#     sn_args = get_config_overrides_for_generation()
#     sn_args.update({
#         'fp32_ln': True,
#         'fp32_logits': True,
#         'max_seq_length': 256,
#         'param_dtype': 'float32',
#     })

#     # [TODO]
#     # Pass in config and sn_default_config_overrides to initialize the model here
#     sn_config = None
#     sn_model = None
    
#     sn_state_dict = checkpoint_conversion(state_dict, config)
#     sn_model.load_state_dict(sn_state_dict)
#     sn_model.eval()

#     kv_cache = SNStaticCache.from_legacy_cache(sn_config,
#                                                past_key_values=[[complete_key_values, complete_key_values]] *
#                                                num_hidden_layers)
    
#     sn_out = sn_model(
#         input_ids=input_ids,
#         attention_mask=attention_mask,
#         past_key_values=kv_cache,
#         use_cache=True,
#         cache_position=torch.tensor([[1]], dtype=torch.int32),
#     )

#     diff, rmse, rmse_ratio, max_ulp_diff = get_errors(hf_out[0].data, sn_out[0].data)
#     assert rmse_ratio < 1e-6
    
#     sn_k_cache = sn_out[1][0][0]
#     diff, rmse, rmse_ratio, max_ulp_diff = get_errors(hf_out[1][1][0].data, sn_k_cache)  # k_cache
#     assert rmse_ratio < 1e-6

#     sn_v_cache = sn_out[1][0][1]
#     diff, rmse, rmse_ratio, max_ulp_diff = get_errors(hf_out[1][1][1].data, sn_v_cache)  # v_cache
#     assert rmse_ratio < 1e-6
