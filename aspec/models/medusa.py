# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Adapted from FasterDecoding/Medusa e2a5d20c048a9b0a4092e6933c34313687422518
# (medusa/train/train_legacy.py compute_loss, Apache-2.0). The per-head t+2+i
# supervision offset and assistant-only CE follow the reference; loss masking,
# chunked computation and the metric tuple use ASpec's native MTP contract.
# The source file is based on tatsu-lab/stanford_alpaca and adapts FastChat code.
# Retained notice: Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang,
# Yann Dubois, Xuechen Li (Apache-2.0).
"""Parallel heads-only Medusa objective. No recurrent state or TTT suffix."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from aspec.models.ops.medusa_ce import medusa_head_ce, use_fused_medusa_ce

# Retaining BF16 logits removes one projection pass in backward.  Bound the
# all-head retained bytes so gradient checkpointing still protects long
# sequences; 4K/Qwen3-8B/K=5 (about 5.8 GiB) uses the faster saved mode.
SAVE_LOGITS_BUDGET_BYTES = 8 * 1024**3


def should_save_fused_logits(tokens, vocab, num_heads, gradient_checkpointing):
    if not gradient_checkpointing:
        return True
    retained = tokens * vocab * 2 * num_heads
    return retained <= SAVE_LOGITS_BUDGET_BYTES


class MedusaModel(nn.Module):
    def __init__(self, draft_model, gradient_checkpointing=False):
        super().__init__()
        # The reference path retains each chunk's FP32 logits until backward.
        # The fused path instead uses the budget below: small selections retain
        # BF16 logits and skip one projection pass; long selections recompute.
        self.draft_model = draft_model
        self.num_heads = draft_model.medusa_num_heads
        self.gradient_checkpointing = gradient_checkpointing
        self.vocab_pruning = False

    def _save_fused_logits(self, tokens, vocab):
        return should_save_fused_logits(
            tokens, vocab, self.num_heads, self.gradient_checkpointing
        )

    def _reference_head_loss(
        self,
        head,
        features,
        input_ids_row,
        feature_positions,
        label_positions,
        selected,
    ):
        """Reference per-head CE at the t+2+i offset, one head."""
        total = features.new_zeros((), dtype=torch.float32)
        hits = features.new_zeros((), dtype=torch.float32)
        for indices in selected.split(128):
            targets = input_ids_row.index_select(
                0, label_positions.index_select(0, indices))
            chunk_features = features.index_select(
                0, feature_positions.index_select(0, indices))

            def loss_chunk(chunk_features, targets, head=head):
                logits = head(chunk_features)
                return (F.cross_entropy(logits.float(), targets, reduction="sum"),
                        (logits.argmax(-1) == targets).float().sum())

            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                loss, chunk_hits = checkpoint(loss_chunk, chunk_features, targets,
                                              use_reentrant=False)
            else:
                loss, chunk_hits = loss_chunk(chunk_features, targets)
            total = total + loss
            hits = hits + chunk_hits.detach()
        return total, hits

    def forward(self, input_ids, hidden_states, loss_mask, attention_mask, thd_layout=None):
        if thd_layout is not None:
            raise NotImplementedError("Medusa currently supports padded CP1 execution only")
        trainable = [p for p in self.parameters() if p.requires_grad]
        zero = (sum(p.reshape(-1)[0] for p in trainable) * 0.0 if trainable
                else hidden_states.new_zeros(()))
        sums = [zero for _ in range(self.num_heads)]
        # Exact token counts are dtype-independent; bf16 cannot represent every
        # integer above 256.
        counts = [torch.zeros((), dtype=torch.long, device=zero.device)
                  for _ in range(self.num_heads)]
        correct = [zero.detach() for _ in range(self.num_heads)]
        for row in range(attention_mask.shape[0]):
            valid_positions = attention_mask[row].bool().nonzero().flatten()
            length = valid_positions.numel()
            if 1 < length and int(valid_positions[-1]) - int(valid_positions[0]) + 1 != length:
                raise ValueError("Medusa requires contiguous non-padding positions per row")
            features = hidden_states[row].detach()
            for head in range(self.num_heads):
                # Reference offset: head i at position t is supervised by token t+2+i.
                offset = head + 2
                if length <= offset:
                    continue
                label_positions = valid_positions[offset:]
                feature_positions = label_positions - offset
                keep = loss_mask[row].index_select(0, label_positions).bool()
                selected = keep.nonzero().flatten()
                head_module = self.draft_model.medusa_head[head]
                if selected.numel() and use_fused_medusa_ce(
                    features,
                    head_module[-1].weight,
                    num_rows=selected.numel(),
                ):
                    selected_features = features.index_select(
                        0, feature_positions.index_select(0, selected))
                    selected_labels = label_positions.index_select(0, selected)
                    targets = input_ids[row].index_select(0, selected_labels)
                    projected = head_module[:-1](selected_features)
                    nll, hits = medusa_head_ce(
                        projected,
                        head_module[-1].weight,
                        targets,
                        save_logits=self._save_fused_logits(
                            selected.numel(), head_module[-1].weight.shape[0]
                        ),
                    )
                    sums[head] = sums[head] + nll.sum()
                    correct[head] = correct[head] + hits.sum()
                elif selected.numel():
                    loss_sum, hits_sum = self._reference_head_loss(
                        head_module,
                        features,
                        input_ids[row],
                        feature_positions,
                        label_positions,
                        selected,
                    )
                    sums[head] = sums[head] + loss_sum
                    correct[head] = correct[head] + hits_sum
                counts[head] = counts[head] + selected.numel()
        losses = [value / count.clamp_min(1) for value, count in zip(sums, counts, strict=True)]
        return (losses, [value.detach() for value in losses],
                [value / count.clamp_min(1) for value, count in zip(correct, counts, strict=True)],
                counts, [zero.detach() for _ in range(self.num_heads)])
