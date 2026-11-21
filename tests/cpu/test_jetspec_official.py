# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Map the published Qwen3 JetSpec config/weight layout into ASpec."""

import torch
from safetensors.torch import save_file

from aspec.models.draft import AutoDraftModelConfig, AutoEagle3DraftModel
from aspec.models.draft.jetspec import (
    normalize_official_jetspec_config,
    official_jetspec_config,
)
from aspec.models.draft.keymap import to_export_keys
from aspec.training.checkpoint import load_initial_draft_weights

_OFFICIAL = {
    "architectures": ["DFlashDraftModel"],
    "auto_map": {"AutoModel": "dflash.DFlashDraftModel"},
    "model_type": "qwen3",
    "dtype": "bfloat16",
    "vocab_size": 151936,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "block_size": 16,
    "num_target_layers": 36,
    "num_hidden_layers": 5,
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-06,
    "max_position_embeddings": 40960,
    "rope_theta": 1000000,
    "tie_word_embeddings": False,
    "layer_types": ["full_attention"] * 5,
    "max_window_layers": 5,
    "dflash_config": {
        "causal_head": True,
        "mask_token_id": 151669,
        "target_layer_ids": [1, 9, 17, 25, 33],
    },
}


def _raw():
    return {
        "architectures": ["JetSpecDraftModel"],
        "model_type": "jetspec",
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "vocab_size": 32,
        "target_hidden_size": 16,
        "num_target_layers": 2,
        "target_num_hidden_layers": 6,
        "target_layer_ids": [1, 4],
        "max_position_embeddings": 128,
        "mask_token_id": 31,
        "block_size": 5,
        "is_causal": True,
    }


def test_official_qwen3_jetspec_config_maps_to_flat_aspec_layout():
    config = AutoDraftModelConfig.from_dict(_OFFICIAL)
    assert config.architectures == ["JetSpecDraftModel"]
    assert config.model_type == "jetspec"
    assert config.is_causal is True
    assert config.block_size == 16
    assert config.num_target_layers == 5
    assert config.target_num_hidden_layers == 36
    assert config.target_layer_ids == [1, 9, 17, 25, 33]
    assert config.mask_token_id == 151669


def test_flat_config_roundtrips_through_published_layout():
    flat = _raw()
    published = official_jetspec_config(flat)
    assert published["architectures"] == ["DFlashDraftModel"]
    assert published["model_type"] == "qwen3"
    assert published["num_target_layers"] == flat["target_num_hidden_layers"]
    assert published["dflash_config"]["causal_head"] is True
    assert published["dflash_config"]["target_layer_ids"] == flat["target_layer_ids"]
    restored = normalize_official_jetspec_config(published)
    assert restored["architectures"] == ["JetSpecDraftModel"]
    assert restored["model_type"] == "jetspec"
    assert restored["is_causal"] is True
    assert restored["num_target_layers"] == flat["num_target_layers"]
    assert restored["target_num_hidden_layers"] == flat["target_num_hidden_layers"]
    for key in ("block_size", "target_layer_ids", "mask_token_id"):
        assert restored[key] == flat[key]


def test_official_weight_names_load_with_borrowed_embedding(tmp_path):
    raw = _raw()
    source = AutoEagle3DraftModel.from_config(AutoDraftModelConfig.from_dict(raw))
    official = to_export_keys(source.state_dict())
    official.pop("embed_tokens.weight")
    path = tmp_path / "model.safetensors"
    save_file(official, str(path))

    target = AutoEagle3DraftModel.from_config(AutoDraftModelConfig.from_dict(raw))
    load_initial_draft_weights(target, str(path), allow_target_embedding=True)
    loaded = target.state_dict()
    for key, value in source.state_dict().items():
        if key == "embed_tokens.weight":
            continue
        torch.testing.assert_close(loaded[key], value, atol=0, rtol=0)


def test_unknown_nested_head_is_not_rewritten():
    published = {key: value for key, value in _OFFICIAL.items() if key != "max_window_layers"}
    raw = {**published, "dflash_config": {**_OFFICIAL["dflash_config"],
                                          "causal_head": False}}
    config = AutoDraftModelConfig.from_dict(raw)
    assert config.architectures == ["DFlashDraftModel"]
    assert config.model_type == "qwen3"
