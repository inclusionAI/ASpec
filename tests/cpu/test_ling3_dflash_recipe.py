# Copyright 2026 Ant Group Co., Ltd.
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

import json
from pathlib import Path

from aspec.config.train_config import load_config
from aspec.models.draft import AutoDraftModelConfig
from aspec.models.draft.dflash import DFlashConfig
from aspec.planning import compile_execution_plan

ROOT = Path(__file__).parents[2]
RECIPE = ROOT / "configs/recipes/ling3_tiny_dflash.yaml"
CONFIG = ROOT / "aspec/config/dflash_ling3_tiny_draft_config.json"
DSPARK_CONFIG = ROOT / "aspec/config/dspark_ling3_tiny_draft_config.json"

SHARED_TARGET_BINDING = (
    "hidden_size",
    "intermediate_size",
    "head_dim",
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
    "rms_norm_eps",
    "max_position_embeddings",
    "rope_theta",
    "num_target_layers",
    "target_hidden_size",
    "target_num_hidden_layers",
    "target_layer_ids",
    "mask_token_id",
    "tie_word_embeddings",
)


def test_ling_tiny_dflash_recipe_resolves_explicit_feature_contract():
    plan = compile_execution_plan(load_config(str(RECIPE))).to_dict()
    assert plan["workload"]["models"]["target"]["family"] == "ling3"
    assert plan["workload"]["models"]["draft"]["family"] == "dflash"
    features = plan["workload"]["features"]
    assert features["target_hidden_size"] == 1536
    assert features["target_num_hidden_layers"] == 24
    assert features["aux_hidden_state_layers"] == [1, 6, 11, 16, 21]
    assert features["tensors"]["hidden_states"]["shape"] == ["sequence", 7680]
    assert features["tensors"]["last_hidden_states"]["value_stage"] == "post_norm"
    objective = plan["workload"]["objective"]
    assert objective["family"] == "dflash"
    assert objective["num_target_layers"] == 5
    assert objective["num_anchors"] == 512
    assert objective["block_size"] == 7
    assert objective["loss_objective"] == "decay"


def test_ling_tiny_dflash_config_binds_tiny_target():
    config = AutoDraftModelConfig.from_file(str(CONFIG))
    assert config.architectures == ["DFlashDraftModel"]
    assert config.model_type == "dflash"
    assert config.hidden_size == 1536
    assert config.intermediate_size == 4608
    assert config.num_hidden_layers == 5
    assert config.num_attention_heads == 12
    assert config.num_key_value_heads == 4
    assert config.head_dim == 128
    assert config.vocab_size == 157184
    assert config.rms_norm_eps == 1e-6
    assert config.max_position_embeddings == 131072
    assert config.rope_theta == 6000000.0
    assert config.tie_word_embeddings is False
    assert config.target_hidden_size == 1536
    assert config.target_num_hidden_layers == 24
    assert config.target_layer_ids == [1, 6, 11, 16, 21]
    assert config.num_target_layers == len(config.target_layer_ids) == 5
    assert max(config.target_layer_ids) < config.target_num_hidden_layers
    assert config.mask_token_id == 156894
    assert 0 <= config.mask_token_id < config.vocab_size

    trainer_config = DFlashConfig.from_pretrained(str(CONFIG))
    assert trainer_config.num_hidden_layers == config.num_hidden_layers
    assert trainer_config.mask_token_id == config.mask_token_id
    assert trainer_config.target_layer_ids == config.target_layer_ids


def test_ling_tiny_dflash_config_matches_dspark_target_binding():
    dflash = json.loads(CONFIG.read_text())
    dspark = json.loads(DSPARK_CONFIG.read_text())
    for key in SHARED_TARGET_BINDING:
        assert dflash[key] == dspark[key], key
