import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vllm_metrics import (  # noqa: E402
    compute_vllm_metrics_delta,
    parse_vllm_metrics_text,
    read_actual_usage_from_vllm_metrics,
)


class VllmMetricsTests(unittest.TestCase):
    def test_parse_prometheus_counters_with_source_labels(self) -> None:
        metrics = """
# HELP vllm:prompt_tokens_cached_total Number of cached prompt tokens.
vllm:prompt_tokens_cached_total{engine="0",model_name="qwen"} 784.0
vllm:prefix_cache_hits_total{engine="0",model_name="qwen"} 784.0
vllm:prefix_cache_queries_total{engine="0",model_name="qwen"} 1568.0
vllm:prompt_tokens_by_source_total{engine="0",model_name="qwen",source="local_cache_hit"} 784.0
vllm:prompt_tokens_by_source_total{engine="0",model_name="qwen",source="local_compute"} 800.0
"""
        parsed = parse_vllm_metrics_text(metrics)
        self.assertEqual(parsed["vllm_prompt_tokens_cached"], 784.0)
        self.assertEqual(parsed["vllm_prefix_cache_hits"], 784.0)
        self.assertEqual(parsed["vllm_prefix_cache_queries"], 1568.0)
        self.assertEqual(parsed["vllm_prompt_tokens_local_cache_hit"], 784.0)
        self.assertEqual(parsed["vllm_prompt_tokens_local_compute"], 800.0)

    def test_compute_delta_and_usage_metrics(self) -> None:
        before = {
            "vllm_prompt_tokens_cached": 784.0,
            "vllm_prefix_cache_hits": 784.0,
            "vllm_prefix_cache_queries": 1568.0,
            "vllm_prompt_tokens_local_cache_hit": 784.0,
            "vllm_prompt_tokens_local_compute": 800.0,
        }
        after = {
            "vllm_prompt_tokens_cached": 1568.0,
            "vllm_prefix_cache_hits": 1568.0,
            "vllm_prefix_cache_queries": 3136.0,
            "vllm_prompt_tokens_local_cache_hit": 1568.0,
            "vllm_prompt_tokens_local_compute": 1600.0,
        }
        delta = compute_vllm_metrics_delta(before, after)
        self.assertEqual(delta["vllm_prompt_tokens_cached_delta"], 784)
        self.assertEqual(delta["vllm_prefix_cache_hits_delta"], 784)
        self.assertEqual(delta["vllm_prefix_cache_queries_delta"], 1568)
        self.assertIsNone(delta["vllm_metrics_error"])

        usage = read_actual_usage_from_vllm_metrics(
            response_json={"usage": {"prompt_tokens": 1000}},
            estimated_cached_tokens=784,
            local_input_tokens=900,
            metrics_delta=delta,
        )
        self.assertEqual(usage["actual_input_tokens"], 1000)
        self.assertEqual(usage["actual_cached_tokens"], 784)
        self.assertEqual(usage["usage_source"], "vllm_metrics_delta")
        self.assertEqual(usage["difference_tokens"], 0)

    def test_missing_metric_reports_error_without_crashing(self) -> None:
        before = {"vllm_prefix_cache_hits": 0.0}
        after = {"vllm_prefix_cache_hits": 0.0}
        delta = compute_vllm_metrics_delta(before, after)
        self.assertIn("missing_metrics", delta["vllm_metrics_error"])

        usage = read_actual_usage_from_vllm_metrics(
            response_json={},
            estimated_cached_tokens=784,
            local_input_tokens=900,
            metrics_delta=delta,
        )
        self.assertEqual(usage["status"], "actual_cache_unknown")
        self.assertIsNone(usage["actual_cached_tokens"])


if __name__ == "__main__":
    unittest.main()
