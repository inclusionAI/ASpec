# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Adapted from FasterDecoding/Medusa e2a5d20c048a9b0a4092e6933c34313687422518
# (medusa/train/train_legacy.py compute_loss, Apache-2.0). The per-head t+2+i
# supervision offset and assistant-only CE follow the reference; loss masking,
# chunked computation and the metric tuple use ASpec's native MTP contract.
"""Parallel heads-only Medusa objective. No recurrent state or TTT suffix."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class MedusaModel(nn.Module):
    def __init__(self, draft_model, gradient_checkpointing=False):
        super().__init__()
        self.draft_model = draft_model
        self.num_heads = draft_model.medusa_num_heads
        self.gradient_checkpointing = gradient_checkpointing
        self.vocab_pruning = False

    def forward(self, input_ids, hidden_states, loss_mask, attention_mask, thd_layout=None):
        if thd_layout is not None:
            raise NotImplementedError("Medusa currently supports padded CP1 execution only")
        lengths = attention_mask.sum(-1).long().tolist()
        zero = sum(p.reshape(-1)[0] for p in self.parameters() if p.requires_grad) * 0.0
        sums = [zero for _ in range(self.num_heads)]
        counts = [zero.detach() for _ in range(self.num_heads)]
        correct = [zero.detach() for _ in range(self.num_heads)]
        for row, length in enumerate(lengths):
            features = hidden_states[row].detach()
            for head in range(self.num_heads):
                # Reference offset: head i at position t is supervised by token t+2+i.
                offset = head + 2
                valid_length = max(0, length - offset)
                if not valid_length:
                    continue
                labels = input_ids[row, offset:offset + valid_length]
                keep = loss_mask[row, offset:offset + valid_length].bool()
                selected = keep.nonzero().flatten()
                for indices in selected.split(128):
                    targets = labels.index_select(0, indices)
                    chunk_features = features.index_select(0, indices)

                    def loss_chunk(chunk_features, targets, head=head):
                        logits = self.draft_model.medusa_head[head](chunk_features)
                        return (F.cross_entropy(logits.float(), targets, reduction="sum"),
                                (logits.argmax(-1) == targets).float().sum())

                    if self.training and torch.is_grad_enabled():
                        loss, hits = checkpoint(loss_chunk, chunk_features, targets,
                                                use_reentrant=False)
                    else:
                        loss, hits = loss_chunk(chunk_features, targets)
                    sums[head] = sums[head] + loss
                    correct[head] = correct[head] + hits.detach()
                counts[head] = counts[head] + selected.numel()
        losses = [value / count.clamp_min(1) for value, count in zip(sums, counts, strict=True)]
        return (losses, [value.detach() for value in losses],
                [value / count.clamp_min(1) for value, count in zip(correct, counts, strict=True)],
                counts, [zero.detach() for _ in range(self.num_heads)])
