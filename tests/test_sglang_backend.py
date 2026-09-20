import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from llm2jev import JevRequest, LLM2Jev, Noul, SGLangBackend, Usage


class SGLangBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = Mock()
        self.tokenizer.encode.side_effect = lambda label, **kwargs: {
            "yes": [7], "no": [9], "multi": [3, 4], "alias": [7],
        }[label]
        self.tokenizer.apply_chat_template.side_effect = lambda messages, **kwargs: (
            "template:" + messages[-1]["content"]
        )
        self.tokenizer.return_value = {"input_ids": [[11, 12, 13], [11, 12, 14, 15]]}
        self.engine = Mock()
        self.engine.score.return_value = SimpleNamespace(scores=[[0.2, 0.8], [0.7, 0.3]])
        self.engine.score.side_effect = lambda **kwargs: SimpleNamespace(
            scores=[{13: [0.2, 0.8], 15: [0.7, 0.3]}[tokens[-1]] for tokens in kwargs["items"]],
        )
        self.engine_class = Mock(return_value=self.engine)
        self.auto_tokenizer = Mock()
        self.auto_tokenizer.from_pretrained.return_value = self.tokenizer
        modules = patch.dict(sys.modules, {
            "sglang": SimpleNamespace(Engine=self.engine_class),
            "transformers": SimpleNamespace(AutoTokenizer=self.auto_tokenizer),
        })
        modules.start()
        self.addCleanup(modules.stop)
        self.prompts = (
            ({"role": "user", "content": "first"},),
            ({"role": "user", "content": "second"},),
        )

    def test_scores_full_token_sequences_with_no_yes_normalization(self) -> None:
        options = {"schedule_policy": "lpm", "mem_fraction_static": 0.4}
        with SGLangBackend("local-model", submission="all", engine_kwargs=options) as backend:
            output = backend.score(model="request-model", prompts=self.prompts)
        self.engine_class.assert_called_once_with(model_path="local-model", **options)
        self.auto_tokenizer.from_pretrained.assert_called_once_with(
            "local-model", local_files_only=True,
        )
        self.tokenizer.assert_called_once_with(
            ["template:first", "template:second"], add_special_tokens=False,
        )
        self.engine.score.assert_called_once_with(
            query=[], items=[[11, 12, 13], [11, 12, 14, 15]],
            label_token_ids=[9, 7], apply_softmax=True,
        )
        self.assertEqual(output.yes_probabilities, (0.8, 0.3))
        self.assertEqual(output.usage, Usage(input_tokens=7, output_tokens=0))
        self.assertEqual(options, {"schedule_policy": "lpm", "mem_fraction_static": 0.4})
        self.engine.shutdown.assert_called_once()

    def test_preserves_chat_template_and_thinking_setting(self) -> None:
        with SGLangBackend("local-model", enable_thinking=True) as backend:
            backend.score(model="model", prompts=self.prompts)
        self.tokenizer.apply_chat_template.assert_any_call(
            list(self.prompts[0]), tokenize=False,
            add_generation_prompt=True, enable_thinking=True,
        )

    def test_default_staged_submission_preserves_full_inputs_output_order_and_usage(self) -> None:
        ids = [[1, 2, 3, 10], [1, 2, 3, 11], [1, 2, 4, 12], [1, 2, 4, 13]]
        original = [tokens[:] for tokens in ids]
        self.tokenizer.return_value = {"input_ids": ids}
        self.engine.score.side_effect = lambda **kwargs: SimpleNamespace(
            scores=[[1 - (tokens[-1] - 9) / 10, (tokens[-1] - 9) / 10]
                    for tokens in kwargs["items"]],
        )
        with SGLangBackend("local-model") as backend:
            output = backend.score(model="model", prompts=self.prompts * 2)
        self.assertEqual([call.kwargs["items"] for call in self.engine.score.call_args_list],
                         [[ids[0]], [ids[1], ids[2]], [ids[3]]])
        self.assertEqual(output.yes_probabilities, (.1, .2, .3, .4))
        self.assertEqual(output.usage, Usage(input_tokens=16, output_tokens=0))
        self.assertEqual(ids, original)
        self.engine.flush_cache.assert_not_called()

    def test_rejects_invalid_submission_and_disabled_cache_before_engine_start(self) -> None:
        for kwargs in ({"submission": "auto"},
                       {"engine_kwargs": {"disable_radix_cache": True}},
                       {"submission": "staged", "engine_kwargs": {"disable_radix_cache": True}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SGLangBackend("local-model", **kwargs)
        self.engine_class.assert_not_called()

    def test_all_submission_allows_disabled_cache(self) -> None:
        with SGLangBackend("local-model", submission="all", engine_kwargs={"disable_radix_cache": True}) as backend:
            backend.score(model="model", prompts=self.prompts)
        self.engine.score.assert_called_once()

    def test_staged_failure_does_not_return_partial_results_or_call_later_stages(self) -> None:
        self.tokenizer.return_value = {
            "input_ids": [[1, 2, 3, 10], [1, 2, 3, 11], [1, 2, 4, 12], [1, 2, 4, 13]],
        }
        self.engine.score.side_effect = [SimpleNamespace(scores=[[.2, .8]]),
                                         SimpleNamespace(scores=[])]
        with SGLangBackend("local-model", submission="staged") as backend:
            with self.assertRaisesRegex(ValueError, "wrong number"):
                backend.score(model="model", prompts=self.prompts * 2)
        self.assertEqual(self.engine.score.call_count, 2)

    def test_rejects_invalid_labels_before_starting_engine(self) -> None:
        for kwargs in ({"yes_label": "multi"}, {"no_label": "alias"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SGLangBackend("local-model", **kwargs)
        self.engine_class.assert_not_called()

    def test_rejects_options_that_change_input_semantics(self) -> None:
        for options in ({"enable_mis": True}, {"model_path": "other-model"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                SGLangBackend("local-model", engine_kwargs=options)
        self.engine_class.assert_not_called()

    def test_rejects_empty_prompts_without_calling_engine(self) -> None:
        with SGLangBackend("local-model") as backend:
            with self.assertRaisesRegex(ValueError, "must not be empty"):
                backend.score(model="model", prompts=[])
        self.engine.score.assert_not_called()

    def test_rejects_malformed_scores(self) -> None:
        invalid = (
            [[0.2, 0.8]],
            [[0.2, 0.8], [0.1, 0.2, 0.7]],
            [[0.2, 0.8], [0.2, 0.2]],
            [[0.2, 0.8], [float("nan"), 0.5]],
            [[0.2, 0.8], [-0.2, 1.2]],
        )
        self.engine.score.side_effect = None
        with SGLangBackend("local-model", submission="all") as backend:
            for scores in invalid:
                with self.subTest(scores=scores), self.assertRaises(ValueError):
                    self.engine.score.return_value = SimpleNamespace(scores=scores)
                    backend.score(model="model", prompts=self.prompts)

    def test_releases_engine_on_error_and_close_is_idempotent(self) -> None:
        backend = SGLangBackend("local-model")
        self.engine.score.side_effect = RuntimeError("engine failed")
        with self.assertRaisesRegex(RuntimeError, "engine failed"), backend:
            backend.score(model="model", prompts=self.prompts)
        backend.close()
        self.engine.shutdown.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            backend.score(model="model", prompts=self.prompts)

    def test_evaluates_jev_request(self) -> None:
        request = JevRequest(
            state="package delayed", model="model",
            questions={"delivery": Noul(), "refund": Noul()},
        )
        with SGLangBackend("local-model") as backend:
            response = LLM2Jev(backend=backend).evaluate(request)
        self.assertEqual(response.answers["delivery"].noul, 0.8)
        self.assertEqual(response.answers["refund"].noul, 0.3)
        self.assertEqual(response.usage.output_tokens, 0)

    def test_missing_optional_dependency_has_install_hint(self) -> None:
        with patch.dict(sys.modules, {"sglang": None}):
            with self.assertRaisesRegex(ImportError, r"llm2jev\[sglang\]"):
                SGLangBackend("local-model")


if __name__ == "__main__":
    unittest.main()
