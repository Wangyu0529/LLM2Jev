from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..core.response import Usage
from ..inference.prompt import ChatPrompt
from ..utils.probability import validate_probabilities
from .base import BinaryBackendOutput
from .tokenization import _apply_chat_template, _single_token_id


def _encode_chat_prompts(
    tokenizer: Any,
    prompts: Sequence[ChatPrompt],
    *,
    enable_thinking: bool,
) -> list[list[int]]:
    rendered = [
        _apply_chat_template(
            tokenizer,
            prompt,
            enable_thinking=enable_thinking,
        )
        for prompt in prompts
    ]
    return tokenizer(rendered, add_special_tokens=False)["input_ids"]


def _yes_probabilities(
    scores: Sequence[Sequence[float]],
    *,
    expected_count: int,
) -> tuple[float, ...]:
    if len(scores) != expected_count:
        raise ValueError("SGLang returned the wrong number of candidate scores")

    probabilities: list[float] = []
    for row in scores:
        values = validate_probabilities(row)
        if len(values) != 2 or not math.isclose(sum(values), 1.0, abs_tol=1e-6):
            raise ValueError("SGLang must return a normalized no/yes pair")
        probabilities.append(values[1])
    return tuple(probabilities)


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
        self.yes_token_id = _single_token_id(
            self.tokenizer, yes_label, "yes_label",
        )
        self.no_token_id = _single_token_id(
            self.tokenizer, no_label, "no_label",
        )
        if self.yes_token_id == self.no_token_id:
            raise ValueError("yes_label and no_label must encode to different tokens")

        self.engine = Engine(model_path=self.model_path, **options)

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

        input_ids = _encode_chat_prompts(
            self.tokenizer,
            prompt_values,
            enable_thinking=self.enable_thinking,
        )
        # SGLang 0.5.14 score() uses max_new_tokens=0 and returns next-token
        # label logprobs normalized over just the requested label IDs.
        result = self.engine.score(
            query=[],
            items=input_ids,
            label_token_ids=[self.no_token_id, self.yes_token_id],
            apply_softmax=True,
        )
        probabilities = _yes_probabilities(
            result.scores,
            expected_count=len(input_ids),
        )

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
