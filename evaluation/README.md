# HotpotQA Evaluation

This folder contains a BEIR-based evaluation harness for the current Graphiti retrieval system.

## Why this evaluation uses a subset

The official BEIR HotpotQA benchmark contains 7,405 queries over a 5.23M-document corpus. This project's retrieval stack is built on Graphiti ingestion, which performs LLM-backed graph extraction per ingested document. That makes full-corpus ingestion impractical for local evaluation.

The script in this folder therefore:

1. downloads the official BEIR HotpotQA dataset
2. samples a reproducible query subset
3. keeps all positive documents for those queries
4. adds a configurable number of random negative documents
5. ingests only that closed-world subset into an isolated Graphiti store
6. computes BEIR retrieval metrics on that subset

This is a tractable regression benchmark for the current system. It is not a full leaderboard-comparable HotpotQA run.

## Install

Install the evaluation-only dependencies through uv.

```
uv sync --extra eval
```

## Run

```
uv run python evaluation\run_hotpotqa_beir.py --max-queries 20 --negative-docs 200
```

Useful flags:

- `--max-queries`: number of HotpotQA test queries to evaluate
- `--negative-docs`: number of random non-relevant documents to add to the closed-world corpus
- `--top-k`: retrieval cutoff used for ranking export and BEIR metrics
- `--reuse-index`: reuse the previously built Graphiti evaluation index instead of re-ingesting
- `--rebuild-subset`: rebuild the sampled BEIR subset files even if they already exist

## Outputs

The script writes artifacts under this folder:

- `datasets/`: downloaded official BEIR data
- `generated/`: BEIR-compatible sampled subset
- `runtime/`: isolated Graphiti storage used for evaluation
- `results/`: timestamped JSON summaries with metrics and per-query rankings