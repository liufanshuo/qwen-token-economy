# Qwen Token Economy Experiments

Inference-time reasoning protocol experiments using:

- Model: Qwen2.5-7B-Instruct
- Precision: BF16
- Hardware: 1× NVIDIA A100 80GB
- Dataset: GSM8K test set, 1,319 samples
- Run seed: 20260730

## Reasoning protocols

- Direct
- Chain-of-Thought
- Self-Consistency@3
- Self-Consistency@5
- Self-Refine@1
- Self-Refine@2

## Metrics

- Accuracy
- Input, output, and total tokens
- Model calls
- Generation and end-to-end latency
- Peak GPU memory
- Accuracy-token Pareto efficiency
- Paired bootstrap confidence intervals
- McNemar exact test

## Repository structure

- `src/`: experiment and analysis programs
- `scripts/`: execution and bundle scripts
- `configs/`: frozen experiment configurations
- `results/summary/`: aggregate result tables
- `results/figures/`: generated figures
- `environment/`: reproducibility metadata
- `data/dataset_metadata.json`: frozen dataset metadata

Raw per-sample JSONL results are attached to the corresponding GitHub Release.
