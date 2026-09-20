import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from llm2jev import Choice, JevRequest, Noul, Score
from llm2jev.sglang_server import (
    _evaluate_request,
    _parse_request,
    _validate_server_args,
)


class SystemOneRequestParsingTests(unittest.TestCase):
    def test_parses_all_question_types(self) -> None:
        request = _parse_request({
            "state": {"message": "Package delayed"},
            "model": "local-model",
            "questions": {
                "department": {
                    "type": "choice",
                    "instructions": "Which department?",
                    "criteria": {"shipping": "Delivery", "billing": None},
                },
                "urgency": {
                    "type": "score",
                    "instructions": "How urgent?",
                    "criteria": ["low", "high"],
                },
                "delivery": {
                    "type": "noul",
                    "instructions": "Is this about delivery?",
                },
            },
        })

        self.assertIsInstance(request.questions["department"], Choice)
        self.assertIsInstance(request.questions["urgency"], Score)
        self.assertIsInstance(request.questions["delivery"], Noul)
        self.assertEqual(request.model, "local-model")

    def test_rejects_missing_and_malformed_fields(self) -> None:
        invalid = (
            [],
            {"model": "model", "questions": {"check": {"type": "noul"}}},
            {"state": "text", "model": "model", "questions": []},
            {
                "state": "text",
                "model": "model",
                "questions": {"check": {"type": "unknown"}},
            },
            {
                "state": "text",
                "model": "model",
                "questions": {"check": {"type": "choice", "criteria": []}},
            },
        )
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises((TypeError, ValueError)):
                _parse_request(payload)


class SystemOneEvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_scores_and_assembles_response(self) -> None:
        tokenizer = Mock()
        tokenizer.encode.side_effect = lambda label, **kwargs: {
            "no": [9],
            "yes": [7],
        }[label]
        tokenizer.apply_chat_template.side_effect = lambda messages, **kwargs: (
            "template:" + messages[-1]["content"]
        )
        tokenizer.return_value = {"input_ids": [[11, 12, 13], [11, 12, 14]]}
        manager = SimpleNamespace(
            tokenizer=tokenizer,
            score_prompts=AsyncMock(
                return_value=SimpleNamespace(
                    scores=[[0.2, 0.8], [0.7, 0.3]],
                    prompt_tokens=6,
                )
            ),
        )
        request = JevRequest(
            state="Package delayed",
            model="local-model",
            questions={
                "department": Choice(criteria={"shipping": None, "billing": None}),
            },
        )

        response = await _evaluate_request(request, manager)

        manager.score_prompts.assert_awaited_once_with(
            [[11, 12, 13], [11, 12, 14]],
            label_token_ids=[9, 7],
            apply_softmax=True,
        )
        self.assertEqual(response.answers["department"].choice, "shipping")
        self.assertEqual(response.usage.input_tokens, 6)
        self.assertEqual(response.usage.output_tokens, 0)

    async def test_requires_tokenizer(self) -> None:
        request = JevRequest(
            state="text",
            model="model",
            questions={"check": Noul()},
        )
        manager = SimpleNamespace(tokenizer=None)

        with self.assertRaisesRegex(RuntimeError, "tokenization"):
            await _evaluate_request(request, manager)


class ServerArgumentTests(unittest.TestCase):
    def test_accepts_standard_single_tokenizer_http_server(self) -> None:
        _validate_server_args(
            SimpleNamespace(
                tokenizer_worker_num=1,
                skip_tokenizer_init=False,
                grpc_mode=False,
                encoder_only=False,
                use_ray=False,
            )
        )

    def test_rejects_unsupported_server_modes(self) -> None:
        defaults = {
            "tokenizer_worker_num": 1,
            "skip_tokenizer_init": False,
            "grpc_mode": False,
            "encoder_only": False,
            "use_ray": False,
        }
        overrides = (
            {"tokenizer_worker_num": 2},
            {"skip_tokenizer_init": True},
            {"grpc_mode": True},
            {"encoder_only": True},
            {"use_ray": True},
        )
        for override in overrides:
            with self.subTest(override=override), self.assertRaises(ValueError):
                _validate_server_args(SimpleNamespace(**(defaults | override)))


if __name__ == "__main__":
    unittest.main()
