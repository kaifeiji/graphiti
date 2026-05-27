from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import json
import random
import shutil
import ssl
import sys
import urllib.request
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import certifi
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF

from app.config import Settings
from app.services.graphiti_service import GraphitiRAGService


DATASET_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip"
DATASET_NAME = "hotpotqa"
DOC_PREFIX = "beir-hotpotqa::"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the current Graphiti retrieval stack on a BEIR HotpotQA subset.",
    )
    parser.add_argument("--split", default="test", help="BEIR split to use. Default: test")
    parser.add_argument("--max-queries", type=int, default=20, help="Number of queries to sample")
    parser.add_argument(
        "--negative-docs",
        type=int,
        default=200,
        help="Random non-relevant documents added to the closed-world corpus",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-k cutoff for exported rankings")
    parser.add_argument(
        "--search-limit",
        type=int,
        default=25,
        help="Internal Graphiti search limit before document-level aggregation",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed for reproducible sampling")
    parser.add_argument(
        "--rebuild-subset",
        action="store_true",
        help="Force rebuilding the sampled BEIR subset files",
    )
    parser.add_argument(
        "--reuse-index",
        action="store_true",
        help="Reuse the previous Graphiti evaluation store if it exists",
    )
    parser.add_argument(
        "--ingest-batch-size",
        type=int,
        default=5,
        help="Number of documents to ingest per Graphiti bulk batch. Default: 5",
    )
    return parser.parse_args()


def ensure_dataset(dataset_root: Path) -> Path:
    dataset_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = dataset_root / DATASET_NAME
    if dataset_dir.exists():
        return dataset_dir

    archive_path = dataset_root / f"{DATASET_NAME}.zip"
    if not archive_path.exists():
        print(f"Downloading {DATASET_URL} to {archive_path}")
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(DATASET_URL, context=ssl_context) as response:
            with archive_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
    else:
        print(f"Reusing existing archive: {archive_path}")

    print(f"Extracting {archive_path} to {dataset_root}")
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(dataset_root)
    return dataset_dir


def load_qrels(qrels_path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with qrels_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            query_id = row["query-id"]
            corpus_id = row["corpus-id"]
            score = int(row["score"])
            qrels[query_id][corpus_id] = score
    return dict(qrels)


def load_selected_queries(
    queries_path: Path,
    selected_query_ids: set[str],
) -> dict[str, str]:
    queries: dict[str, str] = {}
    with queries_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            query_id = payload["_id"]
            if query_id in selected_query_ids:
                queries[query_id] = payload["text"]
    return queries


def pick_query_subset(
    all_qrels: dict[str, dict[str, int]],
    max_queries: int,
    seed: int,
) -> list[str]:
    eligible_ids = sorted(query_id for query_id, labels in all_qrels.items() if labels)
    if max_queries <= 0 or max_queries >= len(eligible_ids):
        return eligible_ids
    rng = random.Random(seed)
    sampled_ids = rng.sample(eligible_ids, k=max_queries)
    sampled_ids.sort()
    return sampled_ids


def reservoir_sample_corpus(
    corpus_path: Path,
    positive_doc_ids: set[str],
    negative_doc_count: int,
    seed: int,
) -> dict[str, dict[str, str]]:
    rng = random.Random(seed)
    positives: dict[str, dict[str, str]] = {}
    negatives: list[dict[str, Any]] = []
    seen_negative = 0

    with corpus_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            doc_id = payload["_id"]
            if doc_id in positive_doc_ids:
                positives[doc_id] = payload
                continue

            if negative_doc_count <= 0:
                continue

            seen_negative += 1
            if len(negatives) < negative_doc_count:
                negatives.append(payload)
                continue

            replace_at = rng.randint(0, seen_negative - 1)
            if replace_at < negative_doc_count:
                negatives[replace_at] = payload

    missing_positive_ids = sorted(positive_doc_ids.difference(positives))
    if missing_positive_ids:
        preview = ", ".join(missing_positive_ids[:5])
        raise RuntimeError(f"Failed to find {len(missing_positive_ids)} positive HotpotQA documents: {preview}")

    subset_corpus = {doc_id: _normalize_corpus_doc(doc) for doc_id, doc in positives.items()}
    for payload in negatives:
        subset_corpus[payload["_id"]] = _normalize_corpus_doc(payload)
    return subset_corpus


def _normalize_corpus_doc(payload: dict[str, Any]) -> dict[str, str]:
    return {
        "title": str(payload.get("title", "")).strip(),
        "text": str(payload.get("text", "")).strip(),
    }


def build_subset_dataset(
    source_dataset_dir: Path,
    subset_dir: Path,
    split: str,
    max_queries: int,
    negative_docs: int,
    seed: int,
    rebuild_subset: bool,
) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]], dict[str, Any]]:
    corpus_out = subset_dir / "corpus.jsonl"
    queries_out = subset_dir / "queries.jsonl"
    qrels_out = subset_dir / "qrels" / f"{split}.tsv"
    metadata_out = subset_dir / "metadata.json"

    if not rebuild_subset and corpus_out.exists() and queries_out.exists() and qrels_out.exists() and metadata_out.exists():
        corpus, queries, qrels = GenericDataLoader(data_folder=str(subset_dir)).load(split=split)
        metadata = json.loads(metadata_out.read_text(encoding="utf-8"))
        return corpus, queries, qrels, metadata

    if subset_dir.exists():
        shutil.rmtree(subset_dir)
    (subset_dir / "qrels").mkdir(parents=True, exist_ok=True)

    all_qrels = load_qrels(source_dataset_dir / "qrels" / f"{split}.tsv")
    query_ids = pick_query_subset(all_qrels, max_queries=max_queries, seed=seed)
    qrels = {query_id: all_qrels[query_id] for query_id in query_ids}
    queries = load_selected_queries(source_dataset_dir / "queries.jsonl", set(query_ids))

    missing_queries = sorted(set(query_ids).difference(queries))
    if missing_queries:
        preview = ", ".join(missing_queries[:5])
        raise RuntimeError(f"Failed to load {len(missing_queries)} queries from HotpotQA: {preview}")

    positive_doc_ids = {
        corpus_id
        for labels in qrels.values()
        for corpus_id, score in labels.items()
        if score > 0
    }
    corpus = reservoir_sample_corpus(
        source_dataset_dir / "corpus.jsonl",
        positive_doc_ids=positive_doc_ids,
        negative_doc_count=negative_docs,
        seed=seed,
    )

    with corpus_out.open("w", encoding="utf-8") as handle:
        for doc_id in sorted(corpus):
            handle.write(
                json.dumps(
                    {
                        "_id": doc_id,
                        "title": corpus[doc_id]["title"],
                        "text": corpus[doc_id]["text"],
                    },
                    ensure_ascii=False,
                )
            )
            handle.write("\n")

    with queries_out.open("w", encoding="utf-8") as handle:
        for query_id in query_ids:
            handle.write(
                json.dumps({"_id": query_id, "text": queries[query_id]}, ensure_ascii=False)
            )
            handle.write("\n")

    with qrels_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["query-id", "corpus-id", "score"])
        for query_id in query_ids:
            for corpus_id, score in sorted(qrels[query_id].items()):
                writer.writerow([query_id, corpus_id, score])

    metadata = {
        "dataset": DATASET_NAME,
        "split": split,
        "seed": seed,
        "query_count": len(query_ids),
        "positive_doc_count": len(positive_doc_ids),
        "negative_doc_count": max(len(corpus) - len(positive_doc_ids), 0),
        "subset_doc_count": len(corpus),
    }
    metadata_out.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    corpus, queries, qrels = GenericDataLoader(data_folder=str(subset_dir)).load(split=split)
    return corpus, queries, qrels, metadata


def make_eval_settings(runtime_dir: Path, search_limit: int, reuse_index: bool) -> Settings:
    runtime_storage = runtime_dir / "storage"
    settings = Settings.from_env(project_root=REPO_ROOT)
    group_id = f"{DATASET_NAME}-eval" if reuse_index else f"{DATASET_NAME}-eval-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    return settings.model_copy(
        update={
            "storage_dir": runtime_storage,
            "session_store_path": runtime_storage / "sessions.json",
            "graph_group_id": group_id,
            "chunk_size": 100000,
            "search_limit": search_limit,
        }
    )


def combine_document_text(title: str, text: str) -> str:
    clean_title = title.strip()
    clean_text = text.strip()
    if clean_title and clean_text:
        return f"{clean_title}\n\n{clean_text}"
    return clean_title or clean_text


async def fetch_existing_doc_ids(service: GraphitiRAGService) -> set[str]:
    graphiti = service._require_graphiti()
    result = await graphiti.driver.execute_query(
        """
        MATCH (episode:Episodic {group_id: $group_id})
        WHERE episode.source_description STARTS WITH $doc_prefix
        RETURN DISTINCT episode.source_description AS source_description
        """,
        params={
            "group_id": service._settings.graph_group_id,
            "doc_prefix": DOC_PREFIX,
        },
    )
    return {
        source_description[len(DOC_PREFIX) :]
        for record in result.records
        if isinstance((source_description := record.get("source_description")), str)
        and source_description.startswith(DOC_PREFIX)
    }


async def ingest_subset(
    service: GraphitiRAGService,
    corpus: dict[str, dict[str, str]],
    ingest_batch_size: int,
    existing_doc_ids: set[str] | None = None,
) -> dict[str, str]:
    content_to_doc_id: dict[str, str] = {}
    known_existing_doc_ids = existing_doc_ids or set()
    total_docs = len(corpus)
    skipped_docs = 0
    ingested_docs = 0
    pending_documents: list[tuple[str, str, str]] = []
    pending_count = 0

    async def flush_pending(processed_index: int) -> None:
        nonlocal ingested_docs, pending_count
        if not pending_documents:
            return
        chunk_count = await service.ingest_text_bulk(pending_documents)
        ingested_docs += pending_count
        print(
            f"Processed {processed_index}/{total_docs} documents into Graphiti "
            f"({ingested_docs} ingested, {skipped_docs} skipped as duplicates, {chunk_count} chunks in batch)"
        )
        pending_documents.clear()
        pending_count = 0

    effective_batch_size = max(1, ingest_batch_size)

    for index, doc_id in enumerate(sorted(corpus), start=1):
        document = corpus[doc_id]
        combined_text = combine_document_text(document.get("title", ""), document.get("text", ""))
        content_to_doc_id[combined_text] = doc_id
        if doc_id in known_existing_doc_ids:
            await flush_pending(index - 1)
            skipped_docs += 1
            print(
                f"Processed {index}/{total_docs} documents into Graphiti "
                f"({ingested_docs} ingested, {skipped_docs} skipped as duplicates)"
            )
            continue

        pending_documents.append((doc_id, combined_text, f"{DOC_PREFIX}{doc_id}"))
        pending_count += 1
        if pending_count >= effective_batch_size:
            await flush_pending(index)

    await flush_pending(total_docs)
    return content_to_doc_id


def extract_doc_id(
    item: Any,
    content_to_doc_id: dict[str, str],
    known_doc_ids: set[str],
) -> str | None:
    source_description = getattr(item, "source_description", None)
    if isinstance(source_description, str) and source_description.startswith(DOC_PREFIX):
        return source_description[len(DOC_PREFIX) :]

    name = getattr(item, "name", None)
    if isinstance(name, str) and name.startswith(DOC_PREFIX):
        return name[len(DOC_PREFIX) :]
    if isinstance(name, str) and name in known_doc_ids:
        return name

    content = getattr(item, "content", None)
    if isinstance(content, str):
        return content_to_doc_id.get(content)

    return None


async def rank_documents_for_query(
    service: GraphitiRAGService,
    query_text: str,
    content_to_doc_id: dict[str, str],
    known_doc_ids: set[str],
    top_k: int,
    search_limit: int,
) -> dict[str, float]:
    graphiti = service._require_graphiti()
    search_config = copy.deepcopy(COMBINED_HYBRID_SEARCH_RRF)
    search_config.limit = search_limit
    raw_results = await graphiti.search_(
        query=query_text,
        config=search_config,
        group_ids=[service._settings.graph_group_id],
    )

    document_scores: dict[str, float] = defaultdict(float)
    for result_group in (
        getattr(raw_results, "episodes", []),
        getattr(raw_results, "nodes", []),
        getattr(raw_results, "edges", []),
        getattr(raw_results, "communities", []),
    ):
        for rank_index, item in enumerate(result_group, start=1):
            doc_id = extract_doc_id(item, content_to_doc_id, known_doc_ids)
            if not doc_id:
                continue
            document_scores[doc_id] += 1.0 / rank_index

    ranked_pairs = sorted(document_scores.items(), key=lambda item: item[1], reverse=True)
    return {doc_id: score for doc_id, score in ranked_pairs[:top_k]}


async def evaluate_graphiti_subset(
    corpus: dict[str, dict[str, str]],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    runtime_dir: Path,
    top_k: int,
    search_limit: int,
    ingest_batch_size: int,
    reuse_index: bool,
) -> dict[str, Any]:
    if runtime_dir.exists() and not reuse_index:
        shutil.rmtree(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    settings = make_eval_settings(
        runtime_dir=runtime_dir,
        search_limit=search_limit,
        reuse_index=reuse_index,
    )
    ingest_marker = runtime_dir / "graphiti_eval_ingested.marker"
    had_existing_index = reuse_index and ingest_marker.exists()
    if not settings.arcgis_access_token:
        raise RuntimeError(
            "ARCGIS_ACCESS_TOKEN is missing. Set it in .env before running evaluation."
        )

    service = GraphitiRAGService(settings=settings)
    await service.startup()
    if not service.is_ready():
        raise RuntimeError(service.startup_error() or "Graphiti evaluation service failed to start.")

    try:
        content_to_doc_id: dict[str, str] = {}
        known_doc_ids = set(corpus)
        existing_doc_ids = await fetch_existing_doc_ids(service)
        if had_existing_index and known_doc_ids.issubset(existing_doc_ids):
            for doc_id, document in corpus.items():
                combined_text = combine_document_text(document.get("title", ""), document.get("text", ""))
                content_to_doc_id[combined_text] = doc_id
            print(
                "Reusing existing Graphiti evaluation dataset in "
                f"Neo4j database '{settings.neo4j_database}' with group '{settings.graph_group_id}'"
            )
        else:
            if existing_doc_ids:
                print(
                    f"Found {len(existing_doc_ids)} existing HotpotQA documents in "
                    f"Neo4j group '{settings.graph_group_id}', skipping duplicates during ingest"
                )
            content_to_doc_id = await ingest_subset(
                service=service,
                corpus=corpus,
                ingest_batch_size=ingest_batch_size,
                existing_doc_ids=existing_doc_ids,
            )
            ingest_marker.write_text(settings.graph_group_id, encoding="utf-8")

        retrieval_results: dict[str, dict[str, float]] = {}
        total_queries = len(queries)
        for index, query_id in enumerate(sorted(queries), start=1):
            retrieval_results[query_id] = await rank_documents_for_query(
                service=service,
                query_text=queries[query_id],
                content_to_doc_id=content_to_doc_id,
                known_doc_ids=known_doc_ids,
                top_k=top_k,
                search_limit=search_limit,
            )
            print(f"Evaluated {index}/{total_queries} queries ({query_id})")

        k_values = sorted({1, 3, 5, top_k})
        ndcg, map_scores, recall, precision = EvaluateRetrieval.evaluate(
            qrels,
            retrieval_results,
            k_values,
        )
        return {
            "metrics": {
                "ndcg": ndcg,
                "map": map_scores,
                "recall": recall,
                "precision": precision,
            },
            "rankings": retrieval_results,
        }
    finally:
        await service.shutdown()


def build_summary(
    args: argparse.Namespace,
    metadata: dict[str, Any],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    evaluation_result: dict[str, Any],
) -> dict[str, Any]:
    rankings: dict[str, dict[str, float]] = evaluation_result["rankings"]
    per_query = []
    for query_id in sorted(queries):
        ranking = rankings.get(query_id, {})
        per_query.append(
            {
                "query_id": query_id,
                "query": queries[query_id],
                "relevant_doc_ids": sorted(
                    corpus_id for corpus_id, score in qrels[query_id].items() if score > 0
                ),
                "retrieved": [
                    {"doc_id": doc_id, "score": score} for doc_id, score in ranking.items()
                ],
            }
        )

    return {
        "dataset": DATASET_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "split": args.split,
            "max_queries": args.max_queries,
            "negative_docs": args.negative_docs,
            "top_k": args.top_k,
            "search_limit": args.search_limit,
            "seed": args.seed,
        },
        "subset": metadata,
        "metrics": evaluation_result["metrics"],
        "per_query": per_query,
    }


async def async_main(args: argparse.Namespace) -> Path:
    evaluation_root = Path(__file__).resolve().parent
    dataset_dir = ensure_dataset(evaluation_root / "datasets")
    subset_tag = f"q{args.max_queries}_n{args.negative_docs}_s{args.seed}"
    subset_dir = evaluation_root / "generated" / f"{DATASET_NAME}_{subset_tag}"
    runtime_dir = evaluation_root / "runtime" / f"{DATASET_NAME}_{subset_tag}"
    results_dir = evaluation_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    corpus, queries, qrels, metadata = build_subset_dataset(
        source_dataset_dir=dataset_dir,
        subset_dir=subset_dir,
        split=args.split,
        max_queries=args.max_queries,
        negative_docs=args.negative_docs,
        seed=args.seed,
        rebuild_subset=args.rebuild_subset,
    )

    print(
        "Prepared HotpotQA subset: "
        f"{metadata['query_count']} queries, {metadata['subset_doc_count']} documents "
        f"({metadata['positive_doc_count']} positive + {metadata['negative_doc_count']} negative)"
    )

    evaluation_result = await evaluate_graphiti_subset(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        runtime_dir=runtime_dir,
        top_k=args.top_k,
        search_limit=max(args.search_limit, args.top_k),
        ingest_batch_size=args.ingest_batch_size,
        reuse_index=args.reuse_index,
    )
    summary = build_summary(
        args=args,
        metadata=metadata,
        queries=queries,
        qrels=qrels,
        evaluation_result=evaluation_result,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_path = results_dir / f"{DATASET_NAME}_{subset_tag}_{timestamp}.json"
    results_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Evaluation metrics:")
    for metric_name, metric_scores in summary["metrics"].items():
        metric_values = ", ".join(f"{name}={value:.4f}" for name, value in metric_scores.items())
        print(f"  {metric_name}: {metric_values}")
    print(f"Saved summary to {results_path}")
    return results_path


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()