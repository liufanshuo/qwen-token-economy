from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import datasets
import modelscope
from datasets import Dataset, DatasetDict, load_from_disk
from modelscope.msdatasets import MsDataset


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "snapshots"
META_PATH = ROOT / "data" / "dataset_metadata.json"
OUT.mkdir(parents=True, exist_ok=True)

def ensure_hf_dataset(obj):
    """兼容 ModelScope 返回 MsDataset 或直接返回 HF Dataset 的情况。"""
    if isinstance(obj, Dataset):
        return obj

    if hasattr(obj, "to_hf_dataset"):
        converted = obj.to_hf_dataset()
        if isinstance(converted, Dataset):
            return converted

    raise TypeError(
        f"Unsupported dataset object: {type(obj)!r}"
    )

def to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("content", "text", "value", "answer", "problem", "question"):
            if key in value and value[key] is not None:
                text = to_text(value[key])
                if text:
                    return text
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        texts = [to_text(item) for item in value]
        return "\n".join(text for text in texts if text).strip()
    return str(value).strip()


def first_nonempty(row: dict[str, Any], names: tuple[str, ...]) -> str:
    for name in names:
        if name in row:
            value = to_text(row[name])
            if value:
                return value
    return ""


def extract_boxed(text: str) -> str:
    marker = r"\boxed{"
    positions = [match.start() for match in re.finditer(re.escape(marker), text)]
    for start in reversed(positions):
        i = start + len(marker)
        depth = 1
        chars: list[str] = []
        while i < len(text) and depth > 0:
            char = text[i]
            if char == "{":
                depth += 1
                chars.append(char)
            elif char == "}":
                depth -= 1
                if depth > 0:
                    chars.append(char)
            else:
                chars.append(char)
            i += 1
        if depth == 0:
            return "".join(chars).strip()
    return ""


def directory_hashes(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        digest = hashlib.sha256()
        with file_path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        result[str(file_path.relative_to(path))] = digest.hexdigest()
    return result


print("Downloading GSM8K from ModelScope...")

gsm_train = ensure_hf_dataset(
    MsDataset.load(
        "AI-ModelScope/gsm8k",
        subset_name="main",
        split="train",
    )
)

gsm_test = ensure_hf_dataset(
    MsDataset.load(
        "AI-ModelScope/gsm8k",
        subset_name="main",
        split="test",
    )
)

print("GSM8K train rows:", len(gsm_train))
print("GSM8K test rows:", len(gsm_test))
print("GSM8K columns:", gsm_test.column_names)

gsm = DatasetDict({
    "train": gsm_train,
    "test": gsm_test,
})

gsm_out = OUT / "gsm8k_main"
gsm.save_to_disk(gsm_out)

print("Downloading MATH-500 from ModelScope...")

math_raw = ensure_hf_dataset(
    MsDataset.load(
        "AI-ModelScope/MATH-500",
        split="test",
    )
)

print("MATH-500 raw rows:", len(math_raw))
print("MATH-500 raw columns:", math_raw.column_names)
print("MATH-500 first raw row:", math_raw[0])


def normalize_math(row: dict[str, Any]) -> dict[str, Any]:
    problem = first_nonempty(
        row,
        ("problem", "question", "query", "prompt", "input"),
    )
    solution = first_nonempty(
        row,
        ("solution", "response", "reference_solution", "rationale"),
    )
    answer = first_nonempty(
        row,
        ("answer", "target", "gold", "reference_answer", "final_answer"),
    )
    if not answer and solution:
        answer = extract_boxed(solution)

    level = first_nonempty(row, ("level", "difficulty", "subset"))
    subject = first_nonempty(row, ("subject", "type", "category"))

    if not problem:
        raise ValueError(f"MATH-500 row missing problem field: {row}")
    if not answer:
        raise ValueError(f"MATH-500 row missing answer field: {row}")

    return {
        "problem": problem,
        "answer": answer,
        "solution": solution,
        "level": level,
        "subject": subject,
    }


math_rows = [normalize_math(dict(row)) for row in math_raw]
math_test = Dataset.from_list(math_rows)
math = DatasetDict({"test": math_test})
math_out = OUT / "math500"
math.save_to_disk(math_out)

if len(gsm["test"]) != 1319:
    raise RuntimeError(f"GSM8K test size mismatch: {len(gsm['test'])}")
if len(math["test"]) != 500:
    raise RuntimeError(f"MATH-500 size mismatch: {len(math['test'])}")

# Reload once to verify the on-disk snapshots, not only in-memory objects.
gsm_check = load_from_disk(gsm_out)
math_check = load_from_disk(math_out)
assert len(gsm_check["test"]) == 1319
assert len(math_check["test"]) == 500
assert {"question", "answer"}.issubset(gsm_check["test"].column_names)
assert {"problem", "answer"}.issubset(math_check["test"].column_names)

metadata = {
    "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    "source": "ModelScope",
    "modelscope_version": modelscope.__version__,
    "datasets_version": datasets.__version__,
    "datasets": {
        "gsm8k": {
            "repo_id": "AI-ModelScope/gsm8k",
            "source_split": "validation",
            "local_split": "test",
            "output_dir": str(gsm_out),
            "num_rows": len(gsm_check["test"]),
            "columns": gsm_check["test"].column_names,
            "fingerprint": gsm_check["test"]._fingerprint,
            "file_sha256": directory_hashes(gsm_out),
        },
        "math500": {
            "repo_id": "AI-ModelScope/MATH-500",
            "source_split": "test",
            "local_split": "test",
            "output_dir": str(math_out),
            "num_rows": len(math_check["test"]),
            "columns": math_check["test"].column_names,
            "fingerprint": math_check["test"]._fingerprint,
            "file_sha256": directory_hashes(math_out),
        },
    },
}

META_PATH.write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print("GSM8K:", gsm_check)
print("MATH-500:", math_check)
print("Saved metadata to", META_PATH)