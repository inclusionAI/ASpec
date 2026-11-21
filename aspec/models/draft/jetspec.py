# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# The JetSpec method is described in "JetSpec: Breaking the Scaling Ceiling of
# Speculative Decoding with Parallel Tree Drafting" (Hu et al., 2026,
# arXiv:2606.18394). This is an independent ASpec implementation: the retained
# DFlash block head already provides the selected-Target-layer feature fusion and
# dual-source KV injection; JetSpec fixes causal intra-block attention. No
# JetSpec source file is copied.
"""JetSpec causal-parallel Draft head over frozen Target features."""

from aspec.models.draft.dflash import DFlashConfig, DFlashDraftModel


class JetSpecConfig(DFlashConfig):
    """Causal-parallel Draft configuration with an explicit block contract."""

    model_type = "jetspec"

    def __init__(self, *, is_causal=True, block_size=16, **kwargs):
        if is_causal is not True:
            raise ValueError("JetSpec requires causal intra-block attention")
        if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 2:
            raise ValueError("JetSpec block_size must be an integer >= 2")
        architectures = kwargs.pop("architectures", ["JetSpecDraftModel"])
        if architectures != ["JetSpecDraftModel"]:
            raise ValueError("JetSpec uses exactly one JetSpecDraftModel architecture")
        kwargs["is_causal"] = True
        super().__init__(architectures=architectures, block_size=block_size, **kwargs)
        if self.ffn_type != "dense":
            raise ValueError("JetSpec currently supports dense Draft FFNs only")
        layer_types = list(getattr(self, "layer_types", []) or [])
        if any(kind != "full_attention" for kind in layer_types):
            raise ValueError("JetSpec currently supports full intra-block attention only")
        if getattr(self, "mtp_conditioning", None) is not None:
            raise ValueError("JetSpec does not use the native-MTP conditioning contract")


class JetSpecDraftModel(DFlashDraftModel):
    """DFlash block head with the JetSpec causal configuration."""

    config_class = JetSpecConfig

    def forward(
        self,
        draft_input_ids,
        context_feature,
        draft_position_ids,
        context_position_ids,
        block_mask=None,
        noise_embedding=None,
        draft_attention_mask=None,
        direct_cp_metadata=None,
        csa_metadata=None,
        qsa_metadata=None,
        dsa_metadata=None,
        return_router_logits=False,
    ):
        if block_mask is None and draft_attention_mask is None and direct_cp_metadata is None:
            raise ValueError(
                "JetSpecDraftModel.forward requires block_mask, draft_attention_mask "
                "or direct_cp_metadata so causal visibility is explicit"
            )
        for masks in (block_mask, draft_attention_mask, direct_cp_metadata):
            if isinstance(masks, (list, tuple)) and (
                len(masks) != self.num_layers or any(mask is None for mask in masks)
            ):
                raise ValueError(
                    "JetSpec requires an effective causal mask for every Draft layer"
                )
        metadata_entries = (
            direct_cp_metadata
            if isinstance(direct_cp_metadata, (list, tuple))
            else ([direct_cp_metadata] if direct_cp_metadata is not None else [])
        )
        if any(getattr(entry, "draft_is_causal", False) is not True for entry in metadata_entries):
            raise ValueError("JetSpec requires causal Direct-CP metadata")
        return super().forward(
            draft_input_ids=draft_input_ids,
            context_feature=context_feature,
            draft_position_ids=draft_position_ids,
            context_position_ids=context_position_ids,
            block_mask=block_mask,
            noise_embedding=noise_embedding,
            draft_attention_mask=draft_attention_mask,
            direct_cp_metadata=direct_cp_metadata,
            csa_metadata=csa_metadata,
            qsa_metadata=qsa_metadata,
            dsa_metadata=dsa_metadata,
            return_router_logits=return_router_logits,
        )


def normalize_official_jetspec_config(raw: dict) -> dict:
    """Translate the published JetSpec config layout into ASpec's flat layout.

    The published draft head uses ``architectures=["DFlashDraftModel"]``,
    ``model_type="qwen3"`` and nests the JetSpec fields under ``dflash_config``
    with ``causal_head=true``. In that layout ``num_target_layers`` is the
    Target decoder depth, while ASpec's flat config uses it as the number of
    captured feature layers.
    """
    raw = dict(raw)
    nested = raw.pop("dflash_config", None)
    if not isinstance(nested, dict) or nested.get("causal_head") is not True:
        raise ValueError("Not an official JetSpec draft config")
    layer_ids = nested.get("target_layer_ids")
    if not isinstance(layer_ids, list) or not layer_ids:
        raise ValueError("Official JetSpec config requires target_layer_ids")
    target_depth = raw.get("num_target_layers")
    if isinstance(target_depth, bool) or not isinstance(target_depth, int) or target_depth < 1:
        raise ValueError("Official JetSpec config requires the Target decoder depth")
    raw["architectures"] = ["JetSpecDraftModel"]
    raw["model_type"] = "jetspec"
    raw["is_causal"] = True
    for key in ("mask_token_id", "target_layer_ids", "target_hidden_size"):
        if key in nested:
            raw[key] = nested[key]
    raw["num_target_layers"] = len(layer_ids)
    raw["target_num_hidden_layers"] = target_depth
    raw.pop("auto_map", None)
    raw.pop("max_window_layers", None)
    return raw


def official_jetspec_config(raw: dict) -> dict:
    """Translate ASpec's flat JetSpec config into the published HF layout."""
    from aspec.config.draft_algorithm import resolve_draft_algorithm

    raw = dict(raw)
    if resolve_draft_algorithm(raw) != "jetspec":
        raise ValueError("Not an ASpec JetSpec draft config")
    target_depth = raw.get("target_num_hidden_layers")
    layer_ids = raw.get("target_layer_ids")
    if isinstance(target_depth, bool) or not isinstance(target_depth, int) or target_depth < 1:
        raise ValueError("JetSpec export requires the Target decoder depth")
    if not isinstance(layer_ids, list) or not layer_ids:
        raise ValueError("JetSpec export requires target_layer_ids")
    nested = {
        key: raw[key]
        for key in ("mask_token_id", "target_layer_ids", "target_hidden_size")
        if key in raw
    }
    nested["causal_head"] = True
    raw["architectures"] = ["DFlashDraftModel"]
    raw["model_type"] = "qwen3"
    raw["num_target_layers"] = target_depth
    raw["dflash_config"] = nested
    raw.pop("auto_map", None)
    return raw

