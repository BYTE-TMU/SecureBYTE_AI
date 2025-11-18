import unittest
import json
import os
import time
from unittest.mock import patch, MagicMock
from src.llm_manager import LLMManager  


class MockProvider:
    def __init__(self):
        self.called = False

    def generate_response(self, system_prompt, user_prompt, config):
        self.called = True
        return f"Response from mock ({config.get('model', 'unknown')})"

    def stream_response(self, system_prompt, user_prompt, config):
        yield "Streamed response chunk 1"
        yield "Streamed response chunk 2"



class TestLLMManager(unittest.TestCase):

    def setUp(self):
        self.mock_providers = {"mock": MockProvider}
        with patch("src.llm_manager.LLMManager._initialize_provider"):
            self.manager = LLMManager(provider="mock")
        self.manager.providers = self.mock_providers
        self.manager.provider_instance = MockProvider()


    def test_initialization_calls_init_provider(self):
        with patch("src.llm_manager.LLMManager._initialize_provider") as mock_init:
            m = LLMManager(provider="mock")
            mock_init.assert_called_once()
            self.assertEqual(m.current_provider, "mock")

    def test_invalid_provider_raises_value_error(self):
        with self.assertRaises(ValueError):
            LLMManager(provider="invalid_provider")


    def test_switch_provider(self):
        self.manager.providers["another"] = self.manager.providers["mock"]
        with patch.object(self.manager, "_initialize_provider") as mock_init:
            self.manager.switch_provider("another")
            mock_init.assert_called_once()
            self.assertEqual(self.manager.current_provider, "another")


    @patch("src.llm_manager.MODELS", {"mock": {"model": "mock-1"}})
    def test_get_model_config(self):
        config = self.manager.get_model_config()
        self.assertIn("model", config)
        self.assertEqual(config["model"], "mock-1")


    @patch("src.llm_manager.MODELS", {"mock": {"model": "mock-1"}})
    def test_generate_response_returns_expected(self):
        result = self.manager.generate_response("Hi")
        self.assertIn("Response from mock", result)


    @patch("src.llm_manager.MODELS", {"mock": {"model": "mock-1"}})
    def test_stream_response_yields_chunks(self):
        chunks = list(self.manager.stream_response("Hi"))
        self.assertEqual(len(chunks), 2)
        self.assertTrue(all("chunk" in c for c in chunks))


    def test_benchmark_provider_collects_statistics(self):
        prompts = ["test 1", "test 2"]
        result = self.manager.benchmark_provider(prompts)
        self.assertIn("provider", result)
        self.assertEqual(len(result["tests"]), 2)
        self.assertIn("average_time", result)
        self.assertIn("average_characters", result)


    def test_compare_providers_combines_results(self):
        self.manager.providers["mock2"] = self.manager.providers["mock"]

        with patch.object(self.manager, "benchmark_provider", return_value={
            "average_time": 1.0,
            "average_characters": 50
        }):
            result = self.manager.compare_providers(["mock", "mock2"], ["prompt"])
            self.assertIn("providers", result)
            self.assertIn("summary", result)
            self.assertIn("fastest_provider", result["summary"])


    def test_save_benchmark_results_creates_json(self):
        data = {"key": "value"}
        filename = "test_results.json"

        self.manager.save_benchmark_results(data, filename)
        self.assertTrue(os.path.exists(filename))

        with open(filename) as f:
            saved = json.load(f)
        self.assertEqual(saved["key"], "value")

        os.remove(filename)


if __name__ == "__main__":
    unittest.main(verbosity=2)