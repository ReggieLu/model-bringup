#!/bin/bash
# Copyright © SambaNova Systems, Inc. Disclosure, reproduction,
# reverse engineering, or any other use made without the advance written
# permission of SambaNova Systems, Inc. is unauthorized and strictly
# prohibited. All rights of ownership and enforcement are reserved.

export PYTHONWARNINGS=ignore
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=2

python3 text_generation_compile.py command=compile generation.model_name_or_path=$PWD/configs/glm4-5.json samba_compile.pef_name=glm_4_5_o0 --config-name o0 --config-path $PWD/configs
