# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Heads-only Medusa uses the existing optimizer/checkpoint/ownership training path."""

from types import SimpleNamespace

from aspec.models.medusa import MedusaModel
from aspec.training.eagle3_trainer import Eagle3Trainer
from aspec.training.trainer import Trainer


class MedusaTrainer(Eagle3Trainer):
    def _create_training_model(self, draft_model):
        return MedusaModel(
            draft_model,
            gradient_checkpointing=getattr(self.args, "gradient_checkpointing", False),
        )

    def _prepare_algorithm_training(self, draft_model):
        if getattr(self.args, "freeze_lm_head", False):
            raise ValueError("Medusa heads own trainable vocabulary projections")

    def _init_target_lm_head(self, target_model_path):
        # CE needs no teacher distribution. The trainer's unconditional
        # weight/norm reads at init only need an object; reuse an existing head.
        self.target_lm_head = SimpleNamespace(
            lm_head=self.draft_model.medusa_head[0][-1], norm=None)

    def get_feature_contract_status(self):
        status = dict(Trainer.get_feature_contract_status(self))
        status["medusa_num_heads"] = getattr(self.args, "ttt_length", None)
        status["medusa_num_layers"] = getattr(
            getattr(self, "draft_model", None), "medusa_num_layers", None)
        return status

    def _forward(self, batch):
        return self.model(
            input_ids=batch["input_ids"].cuda(),
            hidden_states=batch["hidden_states"].cuda(),
            loss_mask=batch["loss_mask"].squeeze(-1).cuda()
            if batch["loss_mask"].ndim == 3 else batch["loss_mask"].cuda(),
            attention_mask=batch["attention_mask"].cuda(),
        )
