# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Text-only conversations using the GLM-5.3-Flash checkpoint template."""


class Glm5FlashRenderer:
    CACHE_VERSION = "v1"
    roles = ("system", "user", "assistant")

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.role_ids = {}
        for role in self.roles:
            marker = f"<|{role}|>"
            ids = tokenizer.encode(marker, add_special_tokens=False)
            if len(ids) != 1 or tokenizer.convert_ids_to_tokens(ids[0]) != marker:
                raise ValueError(f"GLM tokenizer lacks its {marker} role token")
            self.role_ids[ids[0]] = role

    def get_assistant_token_ids(self):
        ids = {role: token for token, role in self.role_ids.items()}
        return [ids["assistant"]], [ids["user"]], 0

    def render(
        self, messages, tools=None, *, max_seq_length, last_turn_only=False, generation_config=None
    ):
        if tools:
            raise ValueError("GLM text training requires conversations without tools")
        for msg in messages:
            if (
                msg.get("role") not in self.roles
                or not isinstance(msg.get("content"), str)
                or msg.get("tool_calls")
            ):
                raise ValueError("Expected GLM system/user/assistant text messages")
            for field in ("content", "reasoning_content"):
                value = msg.get(field)
                if value is None:
                    continue
                if not isinstance(value, str):
                    raise ValueError(f"GLM {field} must be text")
                if any(f"<|{role}|>" in value for role in self.roles):
                    raise ValueError("A GLM message contains a reserved role token")

        kwargs = {
            k: v
            for k, v in (generation_config or {}).items()
            if k in ("reasoning_effort", "clear_thinking")
        }
        ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, return_dict=False, **kwargs
        )
        mask = [0] * len(ids)
        last_assistant = 0
        supervise = False
        for i, token in enumerate(ids):
            role = self.role_ids.get(token)
            if role is not None:
                supervise = role == "assistant"
                if supervise:
                    last_assistant = i
            else:
                mask[i] = int(supervise)
        if last_turn_only:
            mask[:last_assistant] = [0] * last_assistant
        return ids[:max_seq_length], mask[:max_seq_length]
