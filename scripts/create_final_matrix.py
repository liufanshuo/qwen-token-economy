from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

ALL_PROTOCOLS = [
    "direct",
    "cot",
    "self_consistency_3",
    "self_consistency_5",
    "self_refine_1",
    "self_refine_2",
]
SC_PROTOCOLS_WITH_BASELINE = [
    "direct",
    "self_consistency_3",
    "self_consistency_5",
]


def write_yaml(path: Path, payload: dict) -> None:
    path.write_text(
        yaml.safe_dump(
            payload,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print("Wrote:", path.relative_to(ROOT))


def base_config(
    *,
    experiment_name: str,
    dataset: str,
    n_samples: int,
    run_seed: int,
    max_new_tokens: int,
    critique_max_new_tokens: int,
    protocols: list[str],
) -> dict:
    return {
        "experiment_name": experiment_name,
        "model_path": "models/Qwen2.5-7B-Instruct",
        "load_mode": "bf16",
        "dataset": dataset,
        "split": "test",
        "n_samples": n_samples,
        "sample_seed": 20260730,
        "run_seed": run_seed,
        "max_new_tokens": max_new_tokens,
        "critique_max_new_tokens": critique_max_new_tokens,
        "protocols": protocols,
    }


def main() -> None:
    configs = {
        "gsm8k_main.yaml": base_config(
            experiment_name="gsm8k_main_qwen25_7b",
            dataset="gsm8k",
            n_samples=1319,
            run_seed=20260730,
            max_new_tokens=512,
            critique_max_new_tokens=512,
            protocols=ALL_PROTOCOLS,
        ),
        "math500_smoke.yaml": base_config(
            experiment_name="math500_smoke_qwen25_7b",
            dataset="math500",
            n_samples=20,
            run_seed=20260730,
            max_new_tokens=1024,
            critique_max_new_tokens=512,
            protocols=ALL_PROTOCOLS,
        ),
        "math500_main.yaml": base_config(
            experiment_name="math500_main_qwen25_7b",
            dataset="math500",
            n_samples=500,
            run_seed=20260730,
            max_new_tokens=1024,
            critique_max_new_tokens=512,
            protocols=ALL_PROTOCOLS,
        ),
        "gsm8k_sc_seed20260731.yaml": base_config(
            experiment_name="gsm8k_sc_seed20260731_qwen25_7b",
            dataset="gsm8k",
            n_samples=1319,
            run_seed=20260731,
            max_new_tokens=512,
            critique_max_new_tokens=512,
            protocols=SC_PROTOCOLS_WITH_BASELINE,
        ),
        "gsm8k_sc_seed20260801.yaml": base_config(
            experiment_name="gsm8k_sc_seed20260801_qwen25_7b",
            dataset="gsm8k",
            n_samples=1319,
            run_seed=20260801,
            max_new_tokens=512,
            critique_max_new_tokens=512,
            protocols=SC_PROTOCOLS_WITH_BASELINE,
        ),
        "math500_sc_seed20260731.yaml": base_config(
            experiment_name="math500_sc_seed20260731_qwen25_7b",
            dataset="math500",
            n_samples=500,
            run_seed=20260731,
            max_new_tokens=1024,
            critique_max_new_tokens=512,
            protocols=SC_PROTOCOLS_WITH_BASELINE,
        ),
        "math500_sc_seed20260801.yaml": base_config(
            experiment_name="math500_sc_seed20260801_qwen25_7b",
            dataset="math500",
            n_samples=500,
            run_seed=20260801,
            max_new_tokens=1024,
            critique_max_new_tokens=512,
            protocols=SC_PROTOCOLS_WITH_BASELINE,
        ),
    }

    for filename, config in configs.items():
        write_yaml(CONFIG_DIR / filename, config)

    plan = {
        "bundles": {
            "gsm8k_main_remaining": {
                "runs": [
                    {
                        "config": "configs/gsm8k_main.yaml",
                        "protocols": [
                            "cot",
                            "self_consistency_3",
                            "self_consistency_5",
                            "self_refine_1",
                            "self_refine_2",
                        ],
                        "analyze": True,
                    }
                ]
            },
            "math500_smoke_then_main": {
                "runs": [
                    {
                        "config": "configs/math500_smoke.yaml",
                        "protocols": ALL_PROTOCOLS,
                        "analyze": True,
                        "quality_gate": {
                            "max_parse_error_rate": 0.0,
                            "max_truncation_rate": 0.0,
                        },
                    },
                    {
                        "config": "configs/math500_main.yaml",
                        "protocols": ALL_PROTOCOLS,
                        "analyze": True,
                    },
                ]
            },
            "gsm8k_extra_seeds": {
                "runs": [
                    {
                        "config": "configs/gsm8k_sc_seed20260731.yaml",
                        "protocols": SC_PROTOCOLS_WITH_BASELINE,
                        "analyze": True,
                    },
                    {
                        "config": "configs/gsm8k_sc_seed20260801.yaml",
                        "protocols": SC_PROTOCOLS_WITH_BASELINE,
                        "analyze": True,
                    },
                ]
            },
            "math500_extra_seeds": {
                "runs": [
                    {
                        "config": "configs/math500_sc_seed20260731.yaml",
                        "protocols": SC_PROTOCOLS_WITH_BASELINE,
                        "analyze": True,
                    },
                    {
                        "config": "configs/math500_sc_seed20260801.yaml",
                        "protocols": SC_PROTOCOLS_WITH_BASELINE,
                        "analyze": True,
                    },
                ]
            },
        }
    }
    write_yaml(CONFIG_DIR / "experiment_plan.yaml", plan)

    multiseed = {
        "protocols": [
            "self_consistency_3",
            "self_consistency_5",
        ],
        "datasets": {
            "gsm8k": {
                "n_samples": 1319,
                "runs": [
                    {
                        "seed": 20260730,
                        "experiment_name": "gsm8k_main_qwen25_7b",
                    },
                    {
                        "seed": 20260731,
                        "experiment_name": (
                            "gsm8k_sc_seed20260731_qwen25_7b"
                        ),
                    },
                    {
                        "seed": 20260801,
                        "experiment_name": (
                            "gsm8k_sc_seed20260801_qwen25_7b"
                        ),
                    },
                ],
            },
            "math500": {
                "n_samples": 500,
                "runs": [
                    {
                        "seed": 20260730,
                        "experiment_name": "math500_main_qwen25_7b",
                    },
                    {
                        "seed": 20260731,
                        "experiment_name": (
                            "math500_sc_seed20260731_qwen25_7b"
                        ),
                    },
                    {
                        "seed": 20260801,
                        "experiment_name": (
                            "math500_sc_seed20260801_qwen25_7b"
                        ),
                    },
                ],
            },
        },
    }
    write_yaml(CONFIG_DIR / "multiseed_matrix.yaml", multiseed)

    print("Final experiment matrix created successfully.")


if __name__ == "__main__":
    main()