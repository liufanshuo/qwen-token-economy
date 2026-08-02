from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from math_verify import parse as math_parse
    from math_verify import verify as math_verify

    HAS_MATH_VERIFY = True
except Exception:
    HAS_MATH_VERIFY = False


SYSTEM_PROMPT = (
    "You are a careful mathematical reasoning assistant. "
    "Follow the requested final-answer format exactly."
)

SUPPORTED_PROTOCOLS = {
    "direct",
    "cot",
    "self_consistency_3",
    "self_consistency_5",
    "self_refine_1",
    "self_refine_2",
}


@dataclass
class CallRecord:
    stage: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    generation_latency_sec: float
    end_to_end_latency_sec: float
    peak_gpu_memory_mb: float
    max_new_tokens: int
    temperature: float
    top_p: float
    finish_reason: str
    text: str


class LocalLLM:
    def __init__(self, model_path: Path, load_mode: str):
        if not model_path.is_dir():
            raise FileNotFoundError(f"Model directory does not exist: {model_path}")

        self.model_path = str(model_path)
        self.load_mode = load_mode

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=False,
            local_files_only=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if load_mode != "bf16":
            raise ValueError(
                f"This experiment is frozen to BF16 on A100; got {load_mode!r}."
            )
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "BF16 was selected but the current GPU/PyTorch does not support it."
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=False,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()
        self.input_device = self.model.get_input_embeddings().weight.device

    @torch.inference_mode()
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        stage: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> CallRecord:
        torch.manual_seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        end_to_end_start = time.perf_counter()

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.input_device)
        attention_mask = torch.ones_like(input_ids)
        input_tokens = int(input_ids.shape[-1])

        context_limit = getattr(self.model.config, "max_position_embeddings", None)
        if (
            isinstance(context_limit, int)
            and context_limit > 0
            and input_tokens + max_new_tokens > context_limit
        ):
            raise ValueError(
                "Requested context exceeds model limit: "
                f"input={input_tokens}, max_new_tokens={max_new_tokens}, "
                f"limit={context_limit}"
            )

        generation_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if temperature > 0:
            generation_kwargs.update(
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            generation_kwargs.update(do_sample=False)

        generation_start = time.perf_counter()
        output = self.model.generate(**generation_kwargs)
        torch.cuda.synchronize()
        generation_latency = time.perf_counter() - generation_start

        generated_ids = output[0, input_tokens:]
        output_tokens = int(generated_ids.numel())
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        end_to_end_latency = time.perf_counter() - end_to_end_start
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        finish_reason = (
            "length" if output_tokens >= max_new_tokens else "eos_or_stop"
        )

        return CallRecord(
            stage=stage,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            generation_latency_sec=generation_latency,
            end_to_end_latency_sec=end_to_end_latency,
            peak_gpu_memory_mb=peak_mb,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            finish_reason=finish_reason,
            text=text,
        )


def direct_prompt(question: str) -> str:
    return f"""Solve the problem and provide only the final answer, without an explanation.
The last line must be exactly in this format:
FINAL_ANSWER: \\boxed{{answer}}

Problem:
{question}"""


def cot_prompt(question: str) -> str:
    return f"""Solve the problem carefully, showing a concise step-by-step derivation.
The last line must be exactly in this format:
FINAL_ANSWER: \\boxed{{answer}}

Problem:
{question}"""


def critique_prompt(question: str, candidate: str) -> str:
    return f"""Check the candidate solution for mathematical, logical, and formatting errors.
Do not assume it is wrong. Identify specific issues, or state that no issue is found.
Do not produce a new final answer yet.

Problem:
{question}

Candidate solution:
{candidate}"""


def revise_prompt(question: str, candidate: str, critique: str) -> str:
    return f"""Produce a corrected final solution using the problem, candidate, and critique.
Recompute independently when needed.
The last line must be exactly in this format:
FINAL_ANSWER: \\boxed{{answer}}

Problem:
{question}

Previous candidate:
{candidate}

Critique:
{critique}"""


def make_messages(user_content: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def extract_boxed(text: str) -> str | None:
    marker = r"\boxed{"
    positions = [match.start() for match in re.finditer(re.escape(marker), text)]
    for start in reversed(positions):
        index = start + len(marker)
        depth = 1
        chars: list[str] = []
        while index < len(text) and depth > 0:
            char = text[index]
            if char == "{":
                depth += 1
                chars.append(char)
            elif char == "}":
                depth -= 1
                if depth > 0:
                    chars.append(char)
            else:
                chars.append(char)
            index += 1
        if depth == 0:
            return "".join(chars).strip()
    return None


def extract_final_answer(
    text: str,
    dataset_name: str,
) -> tuple[str | None, str]:
    final_lines = re.findall(
        r"FINAL_ANSWER\s*:\s*([^\n]+)",
        text,
        flags=re.IGNORECASE,
    )
    if final_lines:
        value = final_lines[-1].strip()
        boxed = extract_boxed(value)
        return (boxed if boxed is not None else value), "final_answer"

    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed, "boxed"

    if dataset_name == "gsm8k":
        numbers = re.findall(
            r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?",
            text,
        )
        if numbers:
            return numbers[-1].strip(), "numeric_fallback"

    return None, "none"


def normalize_gsm_number(value: str | None) -> Decimal | None:
    if value is None:
        return None

    value = value.strip().replace(",", "").replace("$", "")
    value = re.sub(
        r"\\frac\{([^{}]+)\}\{([^{}]+)\}",
        r"\1/\2",
        value,
    )
    value = value.strip(" .")

    try:
        if re.fullmatch(
            r"[-+]?\d+(?:\.\d+)?/[-+]?\d+(?:\.\d+)?",
            value,
        ):
            numerator, denominator = value.split("/", 1)
            return Decimal(numerator) / Decimal(denominator)
        return Decimal(value)
    except (InvalidOperation, ZeroDivisionError):
        return None


def canonical_gsm_key(value: str | None) -> str | None:
    number = normalize_gsm_number(value)
    if number is None:
        return None
    if number == 0:
        return "0"
    return str(number.normalize())


def parse_math_candidate(value: str | None):
    if value is None or not HAS_MATH_VERIFY:
        return None

    candidates = [value, f"${value}$"]
    for candidate in candidates:
        try:
            parsed = math_parse(candidate)
            if parsed:
                return parsed
        except Exception:
            continue
    return None


def math_answers_equivalent(left: str, right: str) -> bool:
    left_parsed = parse_math_candidate(left)
    right_parsed = parse_math_candidate(right)
    if not left_parsed or not right_parsed:
        return False
    try:
        return bool(math_verify(left_parsed, right_parsed))
    except Exception:
        return False


def select_self_consistency_winner(
    outputs: list[str],
    dataset_name: str,
) -> tuple[int, dict[str, Any]]:
    extracted: list[str | None] = []
    extraction_methods: list[str] = []

    for output in outputs:
        value, method = extract_final_answer(output, dataset_name)
        extracted.append(value)
        extraction_methods.append(method)

    if dataset_name == "gsm8k":
        vote_keys = [canonical_gsm_key(value) for value in extracted]
        valid_votes = [
            (index, key)
            for index, key in enumerate(vote_keys)
            if key is not None
        ]
        if not valid_votes:
            return 0, {
                "vote_answers": extracted,
                "vote_keys": vote_keys,
                "vote_counts": {},
                "vote_extraction_methods": extraction_methods,
                "tie": False,
                "all_votes_invalid": True,
                "winner_index": 0,
            }

        counts = Counter(key for _, key in valid_votes)
        max_count = max(counts.values())
        winners = {
            key for key, count in counts.items()
            if count == max_count
        }
        winner_index = next(
            index
            for index, key in valid_votes
            if key in winners
        )
        return winner_index, {
            "vote_answers": extracted,
            "vote_keys": vote_keys,
            "vote_counts": dict(counts),
            "vote_extraction_methods": extraction_methods,
            "tie": len(winners) > 1,
            "all_votes_invalid": False,
            "winner_index": winner_index,
        }

    if dataset_name != "math500":
        raise ValueError(f"Unsupported dataset for voting: {dataset_name}")

    clusters: list[dict[str, Any]] = []
    vote_keys: list[str | None] = [None] * len(extracted)

    for index, value in enumerate(extracted):
        if value is None or parse_math_candidate(value) is None:
            continue

        assigned = False
        for cluster in clusters:
            representative = str(cluster["representative"])
            if math_answers_equivalent(value, representative):
                cluster["indices"].append(index)
                vote_keys[index] = representative
                assigned = True
                break

        if not assigned:
            clusters.append({
                "representative": value,
                "indices": [index],
            })
            vote_keys[index] = value

    if not clusters:
        return 0, {
            "vote_answers": extracted,
            "vote_keys": vote_keys,
            "vote_counts": {},
            "vote_extraction_methods": extraction_methods,
            "tie": False,
            "all_votes_invalid": True,
            "winner_index": 0,
        }

    max_count = max(len(cluster["indices"]) for cluster in clusters)
    winning_clusters = [
        cluster
        for cluster in clusters
        if len(cluster["indices"]) == max_count
    ]
    winner_index = min(
        int(index)
        for cluster in winning_clusters
        for index in cluster["indices"]
    )
    counts = {
        str(cluster["representative"]): len(cluster["indices"])
        for cluster in clusters
    }

    return winner_index, {
        "vote_answers": extracted,
        "vote_keys": vote_keys,
        "vote_counts": counts,
        "vote_extraction_methods": extraction_methods,
        "tie": len(winning_clusters) > 1,
        "all_votes_invalid": False,
        "winner_index": winner_index,
    }


def gsm_gold(answer: str) -> str:
    return answer.split("####")[-1].strip()


def score_prediction(
    dataset_name: str,
    gold: str,
    full_output: str,
) -> tuple[bool, str | None, bool, str]:
    extracted, extraction_method = extract_final_answer(
        full_output,
        dataset_name,
    )

    if dataset_name == "gsm8k":
        pred_num = normalize_gsm_number(extracted)
        gold_num = normalize_gsm_number(gold)
        parse_error = pred_num is None
        correct = (
            pred_num is not None
            and gold_num is not None
            and pred_num == gold_num
        )
        return bool(correct), extracted, parse_error, extraction_method

    if dataset_name != "math500":
        raise ValueError(f"Unknown dataset: {dataset_name}")
    if not HAS_MATH_VERIFY:
        raise RuntimeError(
            "MATH-500 requires math-verify[antlr4_13_2]."
        )

    gold_parsed = parse_math_candidate(gold)
    pred_parsed = parse_math_candidate(extracted)
    parse_error = pred_parsed is None

    if gold_parsed is None:
        raise ValueError(f"Could not parse MATH-500 gold answer: {gold!r}")

    correct = False
    if pred_parsed is not None:
        try:
            correct = bool(math_verify(gold_parsed, pred_parsed))
        except Exception:
            correct = False
            parse_error = True

    return correct, extracted, parse_error, extraction_method


def sum_calls(calls: list[CallRecord]) -> dict[str, Any]:
    generation_latency = sum(
        call.generation_latency_sec for call in calls
    )
    end_to_end_latency = sum(
        call.end_to_end_latency_sec for call in calls
    )
    return {
        "input_tokens": sum(call.input_tokens for call in calls),
        "output_tokens": sum(call.output_tokens for call in calls),
        "total_tokens": sum(call.total_tokens for call in calls),
        "num_calls": len(calls),
        "generation_latency_sec": generation_latency,
        "end_to_end_latency_sec": end_to_end_latency,
        # Backward-compatible alias. In this version it means end-to-end latency.
        "latency_sec": end_to_end_latency,
        "peak_gpu_memory_mb": max(
            (call.peak_gpu_memory_mb for call in calls),
            default=0.0,
        ),
        "truncated": any(
            call.finish_reason == "length" for call in calls
        ),
    }


def run_protocol(
    llm: LocalLLM,
    protocol: str,
    question: str,
    dataset_name: str,
    max_new_tokens: int,
    critique_max_new_tokens: int,
    base_seed: int,
) -> tuple[str, list[CallRecord], dict[str, Any]]:
    calls: list[CallRecord] = []
    extra: dict[str, Any] = {}

    if protocol == "direct":
        call = llm.generate(
            make_messages(direct_prompt(question)),
            stage="direct",
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            seed=base_seed,
        )
        calls.append(call)
        return call.text, calls, extra

    if protocol == "cot":
        call = llm.generate(
            make_messages(cot_prompt(question)),
            stage="cot",
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            seed=base_seed,
        )
        calls.append(call)
        return call.text, calls, extra

    if protocol.startswith("self_consistency_"):
        sample_count = int(protocol.rsplit("_", 1)[1])
        for sample_index in range(sample_count):
            call = llm.generate(
                make_messages(cot_prompt(question)),
                stage=f"sample_{sample_index}",
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.95,
                seed=base_seed + sample_index,
            )
            calls.append(call)

        winner_index, vote_info = select_self_consistency_winner(
            [call.text for call in calls],
            dataset_name,
        )
        extra.update(vote_info)
        return calls[winner_index].text, calls, extra

    if protocol.startswith("self_refine_"):
        rounds = int(protocol.rsplit("_", 1)[1])
        initial = llm.generate(
            make_messages(cot_prompt(question)),
            stage="initial",
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            seed=base_seed,
        )
        calls.append(initial)
        candidate = initial.text

        for round_index in range(rounds):
            critique = llm.generate(
                make_messages(critique_prompt(question, candidate)),
                stage=f"critique_{round_index + 1}",
                max_new_tokens=critique_max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                seed=base_seed + 10 + round_index * 2,
            )
            calls.append(critique)

            revision = llm.generate(
                make_messages(
                    revise_prompt(
                        question,
                        candidate,
                        critique.text,
                    )
                ),
                stage=f"revision_{round_index + 1}",
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                seed=base_seed + 11 + round_index * 2,
            )
            calls.append(revision)
            candidate = revision.text

        return candidate, calls, extra

    raise ValueError(f"Unknown protocol: {protocol}")


def load_local_dataset(
    root: Path,
    dataset_name: str,
    split: str,
) -> Dataset:
    if split != "test":
        raise ValueError(
            f"Only local split='test' is supported; got {split!r}."
        )

    if dataset_name == "gsm8k":
        snapshot_path = root / "data" / "snapshots" / "gsm8k_main"
    elif dataset_name == "math500":
        snapshot_path = root / "data" / "snapshots" / "math500"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if not snapshot_path.is_dir():
        raise FileNotFoundError(
            f"Dataset snapshot does not exist: {snapshot_path}"
        )

    dataset = load_from_disk(snapshot_path)
    if not isinstance(dataset, DatasetDict):
        raise TypeError(
            f"Expected DatasetDict at {snapshot_path}, got {type(dataset)!r}"
        )
    if split not in dataset:
        raise KeyError(
            f"Split {split!r} not found in {snapshot_path}: {list(dataset)}"
        )
    return dataset[split]


def get_fields(
    dataset_name: str,
    row: dict[str, Any],
) -> tuple[str, str]:
    if dataset_name == "gsm8k":
        return str(row["question"]), gsm_gold(str(row["answer"]))
    if dataset_name == "math500":
        return str(row["problem"]), str(row["answer"])
    raise ValueError(f"Unknown dataset: {dataset_name}")


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "experiment_name",
        "model_path",
        "load_mode",
        "dataset",
        "split",
        "n_samples",
        "sample_seed",
        "run_seed",
        "max_new_tokens",
        "critique_max_new_tokens",
        "protocols",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Config is missing keys: {missing}")

    if config["dataset"] not in {"gsm8k", "math500"}:
        raise ValueError(f"Unsupported dataset: {config['dataset']!r}")
    if config["split"] != "test":
        raise ValueError("Only split: test is supported.")
    if config["load_mode"] != "bf16":
        raise ValueError("This experiment is frozen to load_mode: bf16.")

    protocols = list(config["protocols"])
    if not protocols:
        raise ValueError("Config protocols cannot be empty.")
    if len(protocols) != len(set(protocols)):
        raise ValueError(f"Duplicate protocols in config: {protocols}")

    unknown = sorted(set(protocols) - SUPPORTED_PROTOCOLS)
    if unknown:
        raise ValueError(f"Unsupported protocols: {unknown}")

    for key in (
        "n_samples",
        "max_new_tokens",
        "critique_max_new_tokens",
    ):
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive.")


def validate_manifest(
    ids: list[int],
    *,
    expected_size: int,
    dataset_size: int,
) -> None:
    if len(ids) != expected_size:
        raise ValueError(
            f"Manifest length is {len(ids)}, expected {expected_size}."
        )
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest contains duplicate sample IDs.")
    if any(
        not isinstance(sample_id, int)
        or sample_id < 0
        or sample_id >= dataset_size
        for sample_id in ids
    ):
        raise ValueError("Manifest contains an invalid sample ID.")


def make_or_load_manifest(
    root: Path,
    dataset_name: str,
    dataset_size: int,
    n_samples: int,
    seed: int,
) -> list[int]:
    expected_size = min(n_samples, dataset_size)
    path = (
        root
        / "data"
        / "manifests"
        / f"{dataset_name}_n{expected_size}_seed{seed}.json"
    )

    if path.exists():
        ids = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(ids, list):
            raise TypeError(f"Manifest is not a list: {path}")
        validate_manifest(
            ids,
            expected_size=expected_size,
            dataset_size=dataset_size,
        )
        return ids

    rng = random.Random(seed)
    if expected_size == dataset_size:
        ids = list(range(dataset_size))
    else:
        ids = sorted(rng.sample(range(dataset_size), expected_size))

    validate_manifest(
        ids,
        expected_size=expected_size,
        dataset_size=dataset_size,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(ids, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
    return ids


def completed_ids(path: Path) -> set[int]:
    done: set[int] = set()
    if not path.exists():
        return done

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSONL at {path}:{line_number}"
                ) from exc

            if record.get("error") is not None:
                raise ValueError(
                    f"Raw result file contains an error row at "
                    f"{path}:{line_number}"
                )

            sample_id = int(record["sample_id"])
            if sample_id in done:
                raise ValueError(
                    f"Duplicate sample_id={sample_id} in {path}"
                )
            done.add(sample_id)

    return done


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    root = config_path.parents[1]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("Config must contain one YAML mapping.")
    validate_config(config)

    protocol = args.protocol
    if protocol not in config["protocols"]:
        raise ValueError(
            f"{protocol!r} is not listed in config protocols."
        )

    dataset = load_local_dataset(
        root,
        str(config["dataset"]),
        str(config["split"]),
    )
    if int(config["n_samples"]) > len(dataset):
        raise ValueError(
            f"n_samples={config['n_samples']} exceeds "
            f"dataset size={len(dataset)}."
        )

    model_path = root / str(config["model_path"])
    llm = LocalLLM(model_path, str(config["load_mode"]))

    # Unscored deployment warm-up. Its tokens and latency are not recorded.
    _ = llm.generate(
        make_messages("Reply with exactly: OK"),
        stage="warmup_unscored",
        max_new_tokens=8,
        temperature=0.0,
        top_p=1.0,
        seed=int(config["run_seed"]),
    )

    ids = make_or_load_manifest(
        root,
        str(config["dataset"]),
        len(dataset),
        int(config["n_samples"]),
        int(config["sample_seed"]),
    )

    stem = (
        f"{config['experiment_name']}__{protocol}"
        f"__seed{config['run_seed']}"
    )
    output_path = root / "results" / "raw" / f"{stem}.jsonl"
    error_path = root / "results" / "errors" / f"{stem}.jsonl"
    done = completed_ids(output_path)
    failure_count = 0

    for position, sample_id in enumerate(
        tqdm(ids, desc=protocol),
    ):
        if sample_id in done:
            continue

        row = dict(dataset[sample_id])
        question, gold = get_fields(str(config["dataset"]), row)
        base_seed = int(config["run_seed"]) + sample_id * 1000

        try:
            final_output, calls, extra = run_protocol(
                llm=llm,
                protocol=protocol,
                question=question,
                dataset_name=str(config["dataset"]),
                max_new_tokens=int(config["max_new_tokens"]),
                critique_max_new_tokens=int(
                    config["critique_max_new_tokens"]
                ),
                base_seed=base_seed,
            )
            (
                correct,
                prediction,
                parse_error,
                extraction_method,
            ) = score_prediction(
                str(config["dataset"]),
                gold,
                final_output,
            )
            totals = sum_calls(calls)

            record = {
                "experiment_name": config["experiment_name"],
                "dataset": config["dataset"],
                "split": config["split"],
                "sample_id": sample_id,
                "position": position,
                "protocol": protocol,
                "run_seed": int(config["run_seed"]),
                "sample_seed": int(config["sample_seed"]),
                "model_path": str(model_path),
                "load_mode": config["load_mode"],
                "question": question,
                "gold": gold,
                "prediction": prediction,
                "correct": correct,
                "parse_error": parse_error,
                "extraction_method": extraction_method,
                "final_output": final_output,
                "calls": [asdict(call) for call in calls],
                **totals,
                **extra,
                "error": None,
            }
            append_jsonl(output_path, record)
            done.add(sample_id)

        except Exception as exc:
            failure_count += 1
            error_record = {
                "experiment_name": config["experiment_name"],
                "dataset": config["dataset"],
                "sample_id": sample_id,
                "position": position,
                "protocol": protocol,
                "run_seed": int(config["run_seed"]),
                "question": question,
                "gold": gold,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            append_jsonl(error_path, error_record)
            tqdm.write(
                f"ERROR sample_id={sample_id}: {exc!r}"
            )

    print(
        f"Completed {len(done)}/{len(ids)} successful samples. "
        f"Failures in this run: {failure_count}."
    )
    print("Raw results:", output_path)
    if failure_count:
        print("Error log:", error_path)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
