# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""JetSpec native proposal keeps the causal block contract."""

import pytest
import torch
import torch.nn.functional as F

from aspec.draft_artifact import JetSpecConsumer
from aspec.models.draft import AutoDraftModelConfig, AutoEagle3DraftModel


def _raw(**overrides):
    raw = {
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
    raw.update(overrides)
    return raw


def _setup():
    torch.manual_seed(11)
    model = AutoEagle3DraftModel.from_config(
        AutoDraftModelConfig.from_dict(_raw())
    ).eval()
    ids = torch.tensor([[2, 3, 5, 7]])
    features = torch.randn(1, 4, 32)
    head = torch.randn(32, 16)
    return model, ids, features, head


def _block_hidden(consumer, ids, features):
    mask_token_id = consumer.config.mask_token_id
    padded = F.pad(ids, (0, consumer.block_size), value=mask_token_id)
    padded_features = F.pad(features, (0, 0, 0, consumer.block_size))
    anchor = torch.tensor([[ids.shape[1] - 1]])
    return consumer.backbone._draft_backbone(
        padded,
        padded_features,
        torch.ones((1, padded.shape[1])),
        injected_anchors=anchor,
        injected_keep_mask=torch.ones_like(anchor, dtype=torch.bool),
    )[0][0].unsqueeze(0)


def _reference(consumer, ids, features, head):
    hidden = _block_hidden(consumer, ids, features)
    return F.linear(hidden[:, 1:, :], head).argmax(-1)


def test_jetspec_consumer_matches_direct_block_forward():
    model, ids, features, head = _setup()
    consumer = JetSpecConsumer(model, block_size=model.config.block_size)

    actual = consumer.propose(ids, features, lm_head_weight=head)
    expected = _reference(consumer, ids, features, head)

    assert actual["algorithm"] == "jetspec"
    assert actual["token_ids"].shape == (1, model.config.block_size - 1)
    torch.testing.assert_close(actual["token_ids"], expected, atol=0, rtol=0)


def test_jetspec_consumer_ignores_later_block_positions(monkeypatch):
    model, ids, features, head = _setup()
    consumer = JetSpecConsumer(model, block_size=model.config.block_size)
    baseline = _block_hidden(consumer, ids, features)

    original = consumer.backbone._create_noise_embed

    def perturb_last_draft_position(input_ids, anchor_positions, block_keep_mask):
        embedding = original(input_ids, anchor_positions, block_keep_mask).clone()
        embedding[:, -1, :] += 10.0
        return embedding

    monkeypatch.setattr(
        consumer.backbone, "_create_noise_embed", perturb_last_draft_position
    )
    changed = _block_hidden(consumer, ids, features)

    torch.testing.assert_close(changed[:, :-1], baseline[:, :-1], atol=0, rtol=0)
    assert not torch.equal(changed[:, -1], baseline[:, -1])


def test_jetspec_anchor_target_feature_is_hidden_from_the_block():
    model, ids, features, head = _setup()
    consumer = JetSpecConsumer(model, block_size=model.config.block_size)

    baseline = _block_hidden(consumer, ids, features)
    changed_features = features.clone()
    changed_features[:, -1, :] += 10.0
    after = _block_hidden(consumer, ids, changed_features)
    torch.testing.assert_close(after, baseline, atol=0, rtol=0)

    baseline_tokens = consumer.propose(ids, features, lm_head_weight=head)["token_ids"]
    changed_tokens = consumer.propose(
        ids, changed_features, lm_head_weight=head
    )["token_ids"]
    torch.testing.assert_close(changed_tokens, baseline_tokens, atol=0, rtol=0)


def test_jetspec_consumer_rejects_other_block_algorithms():
    from aspec.models.draft.dflash import DFlashDraftModel

    raw = _raw()
    raw.update(architectures=["DFlashDraftModel"], model_type="dflash")
    config = AutoDraftModelConfig.from_dict(raw)
    with pytest.raises(ValueError, match="JetSpec"):
        JetSpecConsumer(DFlashDraftModel(config), block_size=5)


def test_jetspec_consumer_rejects_block_size_mismatch():
    model, _ids, _features, _head = _setup()
    with pytest.raises(ValueError, match="block size"):
        JetSpecConsumer(model, block_size=model.config.block_size + 1)
