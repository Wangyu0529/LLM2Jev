import unittest

from llm2jev import TransformersBinaryBackend


class FakeTokenizer:
    def __init__(self, token_ids: list[int]) -> None:
        self.token_ids = token_ids
        self.template_call = None

    def encode(self, label: str, *, add_special_tokens: bool) -> list[int]:
        self.encoded = (label, add_special_tokens)
        return self.token_ids

    def apply_chat_template(self, prompt, **kwargs) -> str:
        self.template_call = (prompt, kwargs)
        return "rendered prompt"


class TransformersBinaryBackendTests(unittest.TestCase):
    def test_rejects_invalid_batch_size_before_loading_model(self) -> None:
        with self.assertRaises(ValueError):
            TransformersBinaryBackend("unused", batch_size=0)

    def test_requires_a_single_token_label(self) -> None:
        backend = object.__new__(TransformersBinaryBackend)
        backend.tokenizer = FakeTokenizer([1, 2])

        with self.assertRaisesRegex(ValueError, "exactly one token"):
            backend._single_token_id("yes", "yes_label")

    def test_applies_generation_template_without_thinking(self) -> None:
        backend = object.__new__(TransformersBinaryBackend)
        backend.tokenizer = FakeTokenizer([1])
        backend.enable_thinking = False
        prompt = (
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        )

        rendered = backend._apply_chat_template(prompt)

        self.assertEqual(rendered, "rendered prompt")
        self.assertEqual(
            backend.tokenizer.template_call,
            (
                list(prompt),
                {
                    "tokenize": False,
                    "add_generation_prompt": True,
                    "enable_thinking": False,
                },
            ),
        )


if __name__ == "__main__":
    unittest.main()
