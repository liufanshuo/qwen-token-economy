from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs" / "experiment_plan.yaml"
STATUS_DIR = ROOT / "results" / "status"
LOG_DIR = ROOT / "logs"
STATUS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return value


def tee_command(command: list[str], log_path: Path) -> None:
    print("COMMAND:", " ".join(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n===== {utc_now()} =====\n")
        log_file.write("COMMAND: " + " ".join(command) + "\n")
        log_file.flush()

        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)


def run_quality_gate(config: dict, gate: dict) -> None:
    experiment_name = str(config["experiment_name"])
    summary_path = (
        ROOT
        / "results"
        / "summary"
        / f"{experiment_name}__summary.csv"
    )
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Quality gate cannot find summary: {summary_path}"
        )

    summary = pd.read_csv(summary_path)
    failures: list[str] = []

    max_parse = gate.get("max_parse_error_rate")
    if max_parse is not None:
        observed = float(summary["parse_error_rate"].max())
        if observed > float(max_parse):
            failures.append(
                f"parse_error_rate max={observed:.6f} "
                f"> allowed={float(max_parse):.6f}"
            )

    max_trunc = gate.get("max_truncation_rate")
    if max_trunc is not None:
        observed = float(summary["truncation_rate"].max())
        if observed > float(max_trunc):
            failures.append(
                f"truncation_rate max={observed:.6f} "
                f"> allowed={float(max_trunc):.6f}"
            )

    if failures:
        raise RuntimeError(
            "Quality gate failed for "
            f"{experiment_name}: "
            + "; ".join(failures)
        )

    print(f"Quality gate passed: {experiment_name}")


def write_status(bundle: str, payload: dict) -> None:
    path = STATUS_DIR / f"{bundle}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Status:", path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()

    plan = load_yaml(PLAN_PATH)
    bundles = plan.get("bundles")
    if not isinstance(bundles, dict):
        raise ValueError("experiment_plan.yaml has no bundles mapping")
    if args.bundle not in bundles:
        raise ValueError(
            f"Unknown bundle {args.bundle!r}; "
            f"available={sorted(bundles)}"
        )

    bundle = bundles[args.bundle]
    runs = bundle.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"Bundle has no runs: {args.bundle}")

    status = {
        "bundle": args.bundle,
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "state": "running",
        "runs": [],
        "error": None,
    }
    write_status(args.bundle, status)

    try:
        for run in runs:
            config_rel = Path(str(run["config"]))
            config_path = (ROOT / config_rel).resolve()
            config = load_yaml(config_path)
            experiment_name = str(config["experiment_name"])
            configured_protocols = list(config["protocols"])
            protocols = list(run["protocols"])

            unknown = [
                protocol
                for protocol in protocols
                if protocol not in configured_protocols
            ]
            if unknown:
                raise ValueError(
                    f"Protocols not listed in {config_rel}: {unknown}"
                )

            run_status = {
                "config": str(config_rel),
                "experiment_name": experiment_name,
                "protocols": protocols,
                "started_at_utc": utc_now(),
                "finished_at_utc": None,
                "state": "running",
            }
            status["runs"].append(run_status)
            write_status(args.bundle, status)

            for protocol in protocols:
                log_path = LOG_DIR / f"{experiment_name}__{protocol}.log"
                tee_command(
                    [
                        sys.executable,
                        "src/run_experiment.py",
                        "--config",
                        str(config_rel),
                        "--protocol",
                        protocol,
                    ],
                    log_path,
                )

            if bool(run.get("analyze", True)):
                analysis_log = LOG_DIR / f"{experiment_name}__analysis.log"
                tee_command(
                    [
                        sys.executable,
                        "src/analyze_results.py",
                        "--root",
                        ".",
                        "--config",
                        str(config_rel),
                        "--n-boot",
                        str(args.n_boot),
                    ],
                    analysis_log,
                )

            quality_gate = run.get("quality_gate")
            if quality_gate is not None:
                if not isinstance(quality_gate, dict):
                    raise TypeError("quality_gate must be a mapping")
                run_quality_gate(config, quality_gate)

            run_status["state"] = "success"
            run_status["finished_at_utc"] = utc_now()
            write_status(args.bundle, status)

        status["state"] = "success"
        status["finished_at_utc"] = utc_now()
        write_status(args.bundle, status)
        print(f"Bundle completed successfully: {args.bundle}")

    except Exception as exc:
        status["state"] = "failed"
        status["finished_at_utc"] = utc_now()
        status["error"] = repr(exc)
        if status["runs"]:
            status["runs"][-1]["state"] = "failed"
            status["runs"][-1]["finished_at_utc"] = utc_now()
        write_status(args.bundle, status)
        raise


if __name__ == "__main__":
    main()