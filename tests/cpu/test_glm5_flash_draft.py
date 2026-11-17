# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import json
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


def test_speculators_published_head_binding(tmp_path):
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
    load_initial_draft_weights(draft, str(tmp_path), target_bindings=bindings)
    torch.testing.assert_close(draft.embed_tokens.weight, embedding)
    save_file({**weights, "lm_head.weight": -head}, target / "model.safetensors")
    with pytest.raises(ValueError, match="lm_head.weight differs"):
        load_initial_draft_weights(draft, str(tmp_path), target_bindings=bindings)
    published["aux_hidden_state_layer_ids"][-1] -= 1
    (tmp_path / "config.json").write_text(json.dumps(published))
    with pytest.raises(ValueError, match="capture"):
        load_initial_draft_weights(draft, str(tmp_path), target_bindings=bindings)
