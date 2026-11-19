# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn

from aspec.config.utils import load_target_config
from aspec.data.renderers.glm5 import Glm5FlashRenderer
from aspec.models.dflash import _create_dflash_mask_mod
from aspec.models.draft.dspark import DSparkConfig
from aspec.models.dspark import DSparkModel
from aspec.planning import _resolve_target_family
from aspec.training.checkpoint import load_initial_draft_weights

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "aspec/config/dspark_glm5_3_flash_draft_config.json"


def test_glm_config_and_capture_coordinates(tmp_path):
    raw = {
        "model_type": "glm5_next",
        "architectures": ["Glm5NextForConditionalGeneration"],
        "text_config": {
            "model_type": "glm5_next_text",
            "num_hidden_layers": 45,
            "hidden_size": 4096,
            "vocab_size": 154880,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(raw))
    assert _resolve_target_family(str(tmp_path)) == "glm5_3_flash"
    outer = load_target_config(str(tmp_path))
    assert getattr(outer, "text_config", outer).vocab_size == 154880
    config = DSparkConfig(**json.loads(CONFIG.read_text()))
    assert [k + 1 for k in config.target_layer_ids] == [20, 28, 32, 36, 40, 44]
    draft = nn.Module()
    draft.config = config
    wrapper = DSparkModel(draft, block_size=8)
    options = wrapper._block_mask_options_for_layer(0)
    assert options == {"context_window": 2048, "is_causal": True}
    mask = _create_dflash_mask_mod(
        torch.tensor([[2050]]), torch.tensor([[True]]), 2100, 8, **options
    )
    ids = torch.tensor([1, 2, 2049, 2050, 2100, 2102, 2103])
    assert mask(0, 0, torch.tensor(2), ids).tolist() == [
        False,
        True,
        True,
        False,
        True,
        True,
        False,
    ]


@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"])
def test_draft_assets_and_exact_resume_compare_effective_block_attention(tmp_path, layer_type):
    from types import SimpleNamespace

    from safetensors.torch import save_file

    from aspec.commands.convert_to_hf import _checkpoint_training_contract
    from aspec.config.draft_attention import resolve_draft_attention
    from aspec.draft_artifact import _sha256, validate_native_asset
    from aspec.training.checkpoint import (
        _logical_training_topology,
        _validate_exact_resume_topology,
    )

    configured = json.loads(CONFIG.read_text())
    configured["layer_types"] = [layer_type] * configured["num_hidden_layers"]
    default = dict(configured)
    del default["sliding_window_non_causal"]
    assert resolve_draft_attention(default) == resolve_draft_attention({
        **default, "sliding_window_non_causal": True,
    })
    saved_attention = resolve_draft_attention(configured)

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(configured))
    weights_path = tmp_path / "model.safetensors"
    save_file({"embed_tokens.weight": torch.zeros(1, 1)}, weights_path)
    (tmp_path / "export_manifest.json").write_text(json.dumps({
        "storage_format": "aspec_native",
        "algorithm": "dspark",
        "config_sha256": _sha256(config_path),
        "model_sha256": _sha256(weights_path),
        "draft_structure": {"attention": saved_attention},
    }))
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    args = SimpleNamespace(
        draft_algorithm="dspark", draft_model_config_obj=configured,
        dflash_block_size=configured["block_size"], world_size=1,
    )
    metadata = {"world_size": 1, "training_topology": _logical_training_topology(args, 1)}
    (tmp_path / "meta.json").write_text(json.dumps(metadata))
    validate_native_asset(tmp_path, expected_config=configured, weights_only=True)
    _checkpoint_training_contract(model_dir, configured)
    _validate_exact_resume_topology(args, metadata)

    args.draft_model_config_obj = default
    if layer_type == "full_attention":
        validate_native_asset(tmp_path, expected_config=default, weights_only=True)
        _checkpoint_training_contract(model_dir, default)
        _validate_exact_resume_topology(args, metadata)
    else:
        with pytest.raises(ValueError, match="Native artifact Draft block attention"):
            validate_native_asset(tmp_path, expected_config=default, weights_only=True)
        with pytest.raises(ValueError, match="Export Draft block attention"):
            _checkpoint_training_contract(model_dir, default)
        with pytest.raises(RuntimeError, match="draft_attention_policy"):
            _validate_exact_resume_topology(args, metadata)


class Tokenizer:
    roles = {"<|system|>": 7, "<|user|>": 8, "<|assistant|>": 9}

    def encode(self, text, **kwargs):
        return [self.roles[text]]

    def convert_ids_to_tokens(self, token):
        return next(k for k, value in self.roles.items() if value == token)

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=True, add_generation_prompt=False, return_dict=False)
        return [7, 1, 8, 2, 9, 3, 4, 8, 5, 9, 6]


def test_native_role_masks():
    renderer = Glm5FlashRenderer(Tokenizer())
    messages = [{"role": "user", "content": "prompt"}]
    _, mask = renderer.render(messages, max_seq_length=20)
    assert mask == [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1]
    _, mask = renderer.render(messages, max_seq_length=7, last_turn_only=True)
    assert mask == [0] * 7
    with pytest.raises(ValueError, match="reserved role"):
        renderer.render([{"role": "user", "content": "<|assistant|>"}], max_seq_length=20)


@pytest.mark.parametrize("field", ["thinking", "thinking_content", "reasoning"])
@pytest.mark.parametrize("with_canonical", [False, True])
def test_native_reasoning_aliases_are_rejected(field, with_canonical):
    renderer = Glm5FlashRenderer(Tokenizer())
    message = {"role": "assistant", "content": "answer", field: "reasoning"}
    if with_canonical:
        message["reasoning_content"] = "canonical reasoning"
    with pytest.raises(ValueError, match="reasoning_content"):
        renderer.render([message], max_seq_length=20)


@pytest.mark.parametrize("field", ["thinking", "thinking_content", "reasoning"])
def test_sharegpt_reasoning_remains_canonical_before_rendering(field):
    from aspec.data.preprocessing import _normalize_conversation

    class ReasoningTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [
                {"role": "assistant", "content": "answer", "reasoning_content": "reasoning"}
            ]
            return [9, 10, 11]

    messages = _normalize_conversation([{"from": "gpt", "value": "answer", field: "reasoning"}])
    renderer = Glm5FlashRenderer(ReasoningTokenizer())
    ids, mask = renderer.render(messages, max_seq_length=20)
    assert ids == [9, 10, 11]
    assert mask == [0, 1, 1]


@pytest.fixture
def published_dspark(tmp_path):
    from safetensors.torch import save_file

    config = DSparkConfig(**json.loads(CONFIG.read_text()))
    draft = nn.Module()
    draft.config = config
    draft.embed_tokens = nn.Embedding(3, 2)
    published = {
        "speculators_model_type": "dspark",
        "aux_hidden_state_layer_ids": [20, 28, 32, 36, 40, 44],
        "mask_token_id": 154856,
        "block_size": 8,
        "sample_from_anchor": True,
        "sliding_window_non_causal": False,
        "transformer_layer_config": {
            "sliding_window": 2048,
            "layer_types": ["sliding_attention"] * 5,
            "rope_parameters": {"rope_theta": 10000},
            "rms_norm_eps": 1e-5,
            "num_attention_heads": 64,
            "num_key_value_heads": 64,
            "head_dim": 64,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(published))
    target = tmp_path / "target"
    target.mkdir()
    embedding = torch.randn(3, 2)
    head = torch.randn(3, 2)
    weights = {"embed_tokens.weight": embedding, "lm_head.weight": head}
    save_file(weights, tmp_path / "model.safetensors")
    save_file(weights, target / "model.safetensors")
    bindings = (str(target), "embed_tokens.weight", "lm_head.weight")
    return draft, published, embedding, head, target, bindings


def test_speculators_published_head_binding(tmp_path, published_dspark):
    from safetensors.torch import save_file

    draft, published, embedding, head, target, bindings = published_dspark
    load_initial_draft_weights(draft, str(tmp_path), target_bindings=bindings)
    torch.testing.assert_close(draft.embed_tokens.weight, embedding)
    save_file({"embed_tokens.weight": embedding, "lm_head.weight": -head},
              target / "model.safetensors")
    with pytest.raises(ValueError, match="lm_head.weight differs"):
        load_initial_draft_weights(draft, str(tmp_path), target_bindings=bindings)
    published["aux_hidden_state_layer_ids"][-1] -= 1
    (tmp_path / "config.json").write_text(json.dumps(published))
    with pytest.raises(ValueError, match="capture"):
        load_initial_draft_weights(draft, str(tmp_path), target_bindings=bindings)


def test_published_sliding_policy_uses_speculators_defaults(tmp_path, published_dspark):
    draft, published, embedding, _, _, bindings = published_dspark
    config_path = tmp_path / "config.json"
    del published["sliding_window_non_causal"]
    config_path.write_text(json.dumps(published))
    load_initial_draft_weights(draft, str(tmp_path), target_bindings=bindings)
    torch.testing.assert_close(draft.embed_tokens.weight, embedding)

    config_path.write_text(json.dumps({**published, "sliding_window_non_causal": True}))
    with pytest.raises(ValueError, match="attention policy"):
        load_initial_draft_weights(draft, str(tmp_path), target_bindings=bindings)
    del published["sample_from_anchor"]
    config_path.write_text(json.dumps(published))
    with pytest.raises(ValueError, match="capture"):
        load_initial_draft_weights(draft, str(tmp_path), target_bindings=bindings)


@pytest.mark.parametrize("config", [None, {"speculators_model_type": "dflash"}])
def test_bound_glm_warmstart_requires_speculators_config(tmp_path, published_dspark, config):
    from safetensors.torch import save_file

    draft, _, embedding, _, _, bindings = published_dspark
    config_path = tmp_path / "config.json"
    if config is None:
        config_path.unlink()
    else:
        config_path.write_text(json.dumps(config))

    weights_path = tmp_path / "model.safetensors"
    weights_path.write_bytes(b"unreadable checkpoint")
    with pytest.raises(ValueError, match=r"requires the original Speculators DSpark config.json"):
        load_initial_draft_weights(draft, str(weights_path), target_bindings=bindings,
                                   require_speculators_config=True)

    save_file({"embed_tokens.weight": embedding}, weights_path)
    load_initial_draft_weights(draft, str(weights_path), target_bindings=bindings)
    torch.testing.assert_close(draft.embed_tokens.weight, embedding)


@pytest.mark.parametrize("published_rope,configured_rope", [
    ({}, {"rope_type": "linear", "factor": 2.0}),
    ({"rope_type": "linear", "factor": 2.0}, {}),
    ({"rope_type": "linear", "factor": 2.0}, {"rope_type": "linear", "factor": 4.0}),
    ({"rope_theta": 20000}, {}),
    ({"rope_type": "yarn", "factor": 2.0, "beta_fast": 64},
     {"rope_type": "yarn", "factor": 2.0, "beta_fast": 32}),
])
def test_published_dspark_rejects_different_rope(tmp_path, published_dspark,
                                               published_rope, configured_rope):
    draft, published, _, _, _, bindings = published_dspark
    geometry = published["transformer_layer_config"]
    geometry["rope_parameters"] = {"rope_theta": 10000, **published_rope}
    geometry["max_position_embeddings"] = draft.config.max_position_embeddings
    draft.config.rope_parameters = {"rope_theta": 10000, **configured_rope}
    (tmp_path / "config.json").write_text(json.dumps(published))
    with pytest.raises(ValueError, match="RoPE|geometry"):
        load_initial_draft_weights(draft, str(tmp_path), target_bindings=bindings)


@pytest.mark.parametrize("published_updates,configured_updates", [
    ({"rope_parameters": None}, {}),
    ({"rope_theta": 10000, "rope_parameters": None}, {}),
    ({"rope_parameters": {"rope_type": "default", "factor": 1.0, "rope_theta": 10000,
                          "partial_rotary_factor": 1.0}}, {}),
    ({"rope_theta": 10000, "rope_parameters": None,
      "rope_scaling": {"type": "linear", "factor": 2.0}},
     {"rope_parameters": {"rope_type": "linear", "factor": 2.0, "rope_theta": 10000}}),
    ({"rope_parameters": {"rope_type": "yarn", "factor": 2.0, "rope_theta": 10000,
                          "beta_fast": 32, "beta_slow": 1}},
     {"rope_parameters": {"type": "yarn", "factor": 2.0, "rope_theta": 10000}}),
])
def test_published_dspark_accepts_equivalent_rope(tmp_path, published_dspark,
                                                published_updates, configured_updates):
    draft, published, embedding, _, _, bindings = published_dspark
    published["transformer_layer_config"].update(deepcopy(published_updates))
    published["transformer_layer_config"]["max_position_embeddings"] = draft.config.max_position_embeddings
    for key, value in configured_updates.items():
        setattr(draft.config, key, deepcopy(value))
    (tmp_path / "config.json").write_text(json.dumps(published))
    load_initial_draft_weights(draft, str(tmp_path), target_bindings=bindings)
    torch.testing.assert_close(draft.embed_tokens.weight, embedding)
