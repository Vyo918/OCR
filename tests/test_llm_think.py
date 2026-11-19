import os
import subprocess
import sys
import unittest
from unittest import mock

from docsearch import config
from docsearch.search import Engine


class LlmThinkTests(unittest.TestCase):
    def test_environment_values(self):
        for value, expected in ((None, "default"), ("", "default"), ("default", "default"),
                                ("on", "on"), ("off", "off"), ("invalid", None)):
            with self.subTest(value=value):
                env = os.environ.copy()
                env.pop("DOCSEARCH_LLM_THINK", None)
                if value is not None:
                    env["DOCSEARCH_LLM_THINK"] = value
                result = subprocess.run([sys.executable, "-c", "from docsearch.config import LLM_THINK; print(LLM_THINK)"],
                                        env=env, capture_output=True, text=True)
                if expected is None:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("DOCSEARCH_LLM_THINK must be default, on or off", result.stderr)
                else:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.strip(), expected)

    def test_first_request_and_empty_reasoning_answer(self):
        engine = Engine.__new__(Engine)
        engine.seed = 7
        for mode, extra in (("default", {}), ("on", {"chat_template_kwargs": {"enable_thinking": True}}),
                            ("off", {"chat_template_kwargs": {"enable_thinking": False}})):
            for content, finish in (("answer", "stop"), ("<think></think>answer", "stop"),
                                    ("answer", "length")):
                if finish == "length" and mode != "off":
                    continue
                with self.subTest(mode=mode, content=content, finish=finish), \
                        mock.patch.object(config, "LLM_THINK", mode), \
                        mock.patch("docsearch.search._complete", side_effect=AssertionError("unexpected fallback")), \
                        mock.patch("docsearch.search._stream", return_value=("", content, finish)) as stream:
                    self.assertEqual(engine._chat_http("system", "user", 10, 5), ("answer", ""))
                    stream.assert_called_once_with(
                        [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}],
                        {"seed": 7, "max_tokens": 15, **extra})


if __name__ == "__main__":
    unittest.main()
