from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..core.response import Usage
from ..inference.backend import BinaryBackendOutput
from ..inference.prompt import ChatPrompt
from ..utils.probability import validate_probabilities


class SGLangBackend:
    """Prefill-only binary scorer using SGLang's native scoring and prefix cache."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        yes_label: str = "yes",
        no_label: str = "no",
        enable_thinking: bool = False,
        engine_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        options = dict(engine_kwargs or {})
        if options.get("enable_mis"):
            raise ValueError("enable_mis changes the independent candidate scoring path")
        if "model_path" in options:
            raise ValueError("pass model_path directly, not in engine_kwargs")

        try:
            from sglang import Engine
            from transformers import AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "SGLangBackend requires the 'sglang' extra; install llm2jev[sglang]"
            ) from error

        self.model_path = str(model_path)
        self.enable_thinking = enable_thinking
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True,
        )
        self.yes_token_id = self._single_token_id(yes_label, "yes_label")
        self.no_token_id = self._single_token_id(no_label, "no_label")
        if self.yes_token_id == self.no_token_id:
            raise ValueError("yes_label and no_label must encode to different tokens")

        self.engine = Engine(model_path=self.model_path, **options)

    def _single_token_id(self, label: str, field: str) -> int:
        token_ids = self.tokenizer.encode(label, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"{field} must encode to exactly one token, got {token_ids}")
        return token_ids[0]

    def _encode_prompts(self, prompts: Sequence[ChatPrompt]) -> list[list[int]]:
        rendered = [
            self.tokenizer.apply_chat_template(
                list(prompt),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            for prompt in prompts
        ]
        return self.tokenizer(rendered, add_special_tokens=False)["input_ids"]

    def score(
        self,
        *,
        model: str,
        prompts: Sequence[ChatPrompt],
    ) -> BinaryBackendOutput:
        del model  # The engine's model is fixed at construction.
        if self.engine is None:
            raise RuntimeError("SGLangBackend is closed")
        prompt_values = tuple(prompts)
        if not prompt_values:
            raise ValueError("prompts must not be empty")

        input_ids = self._encode_prompts(prompt_values)
        # SGLang 0.5.14 score() uses max_new_tokens=0 and returns next-token
        # label logprobs normalized over just the requested label IDs.
        result = self.engine.score(
            query=[],
            items=input_ids,
            label_token_ids=[self.no_token_id, self.yes_token_id],
            apply_softmax=True,
        )
        if len(result.scores) != len(input_ids):
            raise ValueError("SGLang returned the wrong number of candidate scores")

        probabilities: list[float] = []
        for row in result.scores:
            values = validate_probabilities(row)
            if len(values) != 2 or not math.isclose(sum(values), 1.0, abs_tol=1e-6):
                raise ValueError("SGLang must return a normalized no/yes pair")
            probabilities.append(values[1])

        return BinaryBackendOutput(
            yes_probabilities=probabilities,
            usage=Usage(input_tokens=sum(map(len, input_ids)), output_tokens=0),
        )

    def close(self) -> None:
        """Shut down this backend's SGLang engine and release GPU resources."""
        if self.engine is not None:
            self.engine.shutdown()
            self.engine = None

    def __enter__(self) -> SGLangBackend:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
