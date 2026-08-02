from decimal import Decimal
from pathlib import Path
import importlib.util
import sys

root = Path("/home/qwen-token-exp/token_economy")
module_path = root / "src" / "run_experiment.py"

spec = importlib.util.spec_from_file_location(
    "run_experiment",
    module_path,
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

assert module.extract_boxed(r"x \boxed{\frac{1}{2}}") == r"\frac{1}{2}"

answer, method = module.extract_final_answer(
    "work\nFINAL_ANSWER: \\boxed{42}",
    "gsm8k",
)
assert answer == "42"
assert method == "final_answer"

assert module.normalize_gsm_number(r"\frac{1}{2}") == Decimal("0.5")
assert module.normalize_gsm_number("1,200") == Decimal("1200")
assert module.canonical_gsm_key("1000") == module.canonical_gsm_key("1,000")

winner, info = module.select_self_consistency_winner(
    [
        "FINAL_ANSWER: \\boxed{2}",
        "no parseable answer",
        "FINAL_ANSWER: \\boxed{2}",
    ],
    "gsm8k",
)
assert winner == 0
assert info["vote_counts"] == {"2": 2}
assert info["all_votes_invalid"] is False

tokens = module.sum_calls([
    module.CallRecord(
        stage="a",
        input_tokens=10,
        output_tokens=4,
        total_tokens=14,
        generation_latency_sec=1.0,
        end_to_end_latency_sec=1.2,
        peak_gpu_memory_mb=100.0,
        max_new_tokens=8,
        temperature=0.0,
        top_p=1.0,
        finish_reason="eos_or_stop",
        text="",
    ),
    module.CallRecord(
        stage="b",
        input_tokens=20,
        output_tokens=5,
        total_tokens=25,
        generation_latency_sec=2.0,
        end_to_end_latency_sec=2.3,
        peak_gpu_memory_mb=120.0,
        max_new_tokens=8,
        temperature=0.0,
        top_p=1.0,
        finish_reason="length",
        text="",
    ),
])
assert tokens["input_tokens"] == 30
assert tokens["output_tokens"] == 9
assert tokens["total_tokens"] == 39
assert tokens["num_calls"] == 2
assert tokens["truncated"] is True
assert tokens["peak_gpu_memory_mb"] == 120.0

print("run_experiment logic self-test passed.")