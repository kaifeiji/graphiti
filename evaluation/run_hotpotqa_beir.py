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
import time
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
from graphiti_core.llm_client.errors import RateLimitError
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF

from app.config import Settings
from app.services.graphiti_service import GraphitiRAGService


DATASET_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip"
DATASET_NAME = "hotpotqa"
DOC_PREFIX = "beir-hotpotqa::"
INGEST_MAX_RATE_LIMIT_RETRIES = 8
INGEST_RATE_LIMIT_BACKOFF_SECONDS = 5.0


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
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=4000,
        help="Max completion tokens used by Graphiti LLM calls during evaluation. Default: 4000",
    )
    parser.add_argument(
        "--evaluate-existing-only",
        action="store_true",
        help="Skip ingest and evaluate only the documents already present in the target Graphiti group.",
    )
    parser.add_argument(
        "--existing-group-id",
        default=None,
        help="Existing Graphiti group id to evaluate against. Useful with --evaluate-existing-only.",
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


def filter_dataset_to_existing_docs(
    corpus: dict[str, dict[str, str]],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    existing_doc_ids: set[str],
) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]], dict[str, int]]:
    filtered_corpus = {
        doc_id: document
        for doc_id, document in corpus.items()
        if doc_id in existing_doc_ids
    }
    filtered_qrels: dict[str, dict[str, int]] = {}
    partially_covered_query_count = 0
    for query_id, labels in qrels.items():
        positive_labels = {
            corpus_id: score
            for corpus_id, score in labels.items()
            if score > 0
        }
        matching_labels = {
            corpus_id: score
            for corpus_id, score in positive_labels.items()
            if corpus_id in existing_doc_ids
        }
        if not positive_labels:
            continue
        if len(matching_labels) == len(positive_labels):
            filtered_qrels[query_id] = matching_labels
        elif matching_labels:
            partially_covered_query_count += 1

    filtered_queries = {
        query_id: query_text
        for query_id, query_text in queries.items()
        if query_id in filtered_qrels
    }
    stats = {
        "existing_doc_count": len(filtered_corpus),
        "retained_query_count": len(filtered_queries),
        "dropped_query_count": max(len(queries) - len(filtered_queries), 0),
        "partial_query_count": partially_covered_query_count,
    }
    return filtered_corpus, filtered_queries, filtered_qrels, stats


def make_eval_settings(
    runtime_dir: Path,
    search_limit: int,
    reuse_index: bool,
    max_completion_tokens: int,
    graph_group_id: str | None,
) -> Settings:
    runtime_storage = runtime_dir / "storage"
    settings = Settings.from_env(project_root=REPO_ROOT)
    group_id = graph_group_id or (
        f"{DATASET_NAME}-eval"
        if reuse_index
        else f"{DATASET_NAME}-eval-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    )
    return settings.model_copy(
        update={
            "storage_dir": runtime_storage,
            "session_store_path": runtime_storage / "sessions.json",
            "graph_group_id": group_id,
            "chunk_size": 100000,
            "search_limit": search_limit,
            "max_completion_tokens": max_completion_tokens,
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


def format_elapsed(seconds: float) -> str:
    rounded_seconds = max(0, int(seconds))
    hours, remainder = divmod(rounded_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


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
    pending_documents: list[tuple[str, str, str]] = []
    batches: list[list[tuple[str, str, str]]] = []

    effective_batch_size = max(1, ingest_batch_size)
    ingest_started_at = time.perf_counter()

    for doc_id in sorted(corpus):
        document = corpus[doc_id]
        combined_text = combine_document_text(document.get("title", ""), document.get("text", ""))
        content_to_doc_id[combined_text] = doc_id
        if doc_id in known_existing_doc_ids:
            skipped_docs += 1
            continue

        pending_documents.append((doc_id, combined_text, f"{DOC_PREFIX}{doc_id}"))
        if len(pending_documents) >= effective_batch_size:
            batches.append(pending_documents)
            pending_documents = []

    if pending_documents:
        batches.append(pending_documents)

    docs_to_ingest = total_docs - skipped_docs
    if skipped_docs:
        print(
            f"Skipping {skipped_docs}/{total_docs} documents already present in Graphiti before ingest starts"
        )

    if not batches:
        print(
            f"Processed {total_docs}/{total_docs} documents into Graphiti "
            f"(0 newly ingested, {skipped_docs} skipped as duplicates, 100.0%, avg 0.00 docs/s)"
        )
        return content_to_doc_id

    ingested_docs = 0
    ingested_chunks = 0
    completed_batches = 0
    total_batches = len(batches)

    for batch_index, batch_documents in enumerate(batches, start=1):
        batch_started_at = time.perf_counter()
        retry_count = 0

        while True:
            try:
                chunk_count = await service.ingest_text_bulk(batch_documents)
                break
            except RateLimitError:
                retry_count += 1
                if retry_count > INGEST_MAX_RATE_LIMIT_RETRIES:
                    raise

                backoff_seconds = min(
                    INGEST_RATE_LIMIT_BACKOFF_SECONDS * (2 ** (retry_count - 1)),
                    120.0,
                ) + random.uniform(0.0, 1.0)
                print(
                    f"Rate limit hit on batch {batch_index}/{total_batches}; retrying in {backoff_seconds:.1f}s "
                    f"(attempt {retry_count}/{INGEST_MAX_RATE_LIMIT_RETRIES})"
                )
                await asyncio.sleep(backoff_seconds)

        batch_elapsed = time.perf_counter() - batch_started_at
        batch_doc_count = len(batch_documents)
        ingested_docs += batch_doc_count
        ingested_chunks += chunk_count
        completed_batches += 1
        elapsed = time.perf_counter() - ingest_started_at
        avg_docs_per_second = ingested_docs / elapsed if elapsed > 0 else 0.0
        percent_complete = ((ingested_docs + skipped_docs) / total_docs) * 100 if total_docs else 100.0
        remaining_docs = max(docs_to_ingest - ingested_docs, 0)
        eta_seconds = (remaining_docs / avg_docs_per_second) if avg_docs_per_second > 0 else 0.0
        batch_docs_per_second = batch_doc_count / batch_elapsed if batch_elapsed > 0 else 0.0
        print(
            f"Processed {ingested_docs + skipped_docs}/{total_docs} documents into Graphiti "
            f"({percent_complete:.1f}%, {ingested_docs} ingested, {skipped_docs} skipped as duplicates, "
            f"{ingested_chunks} chunks total, batch {completed_batches}/{total_batches}, "
            f"batch_id {batch_index}, batch {batch_docs_per_second:.2f} docs/s, "
            f"avg {avg_docs_per_second:.2f} docs/s, elapsed {format_elapsed(elapsed)}, "
            f"eta {format_elapsed(eta_seconds)})"
        )

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
    max_completion_tokens: int,
    evaluate_existing_only: bool,
    existing_group_id: str | None,
    reuse_index: bool,
) -> tuple[dict[str, Any], dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]], dict[str, int] | None]:
    if runtime_dir.exists() and not reuse_index:
        shutil.rmtree(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    settings = make_eval_settings(
        runtime_dir=runtime_dir,
        search_limit=search_limit,
        reuse_index=reuse_index,
        max_completion_tokens=max_completion_tokens,
        graph_group_id=existing_group_id,
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
        filtered_corpus = corpus
        filtered_queries = queries
        filtered_qrels = qrels
        existing_filter_stats: dict[str, int] | None = None
        content_to_doc_id: dict[str, str] = {}
        existing_doc_ids = await fetch_existing_doc_ids(service)
        if evaluate_existing_only:
            filtered_corpus, filtered_queries, filtered_qrels, existing_filter_stats = filter_dataset_to_existing_docs(
                corpus=corpus,
                queries=queries,
                qrels=qrels,
                existing_doc_ids=existing_doc_ids,
            )
            if not filtered_corpus:
                raise RuntimeError(
                    f"No HotpotQA documents from the requested subset exist in Graphiti group '{settings.graph_group_id}'."
                )
            print(
                "Evaluating existing Graphiti ingest only with full positive-doc coverage: "
                f"group '{settings.graph_group_id}', "
                f"{existing_filter_stats['existing_doc_count']} documents, "
                f"{existing_filter_stats['retained_query_count']} queries retained, "
                f"{existing_filter_stats['partial_query_count']} queries dropped with only partial positive-doc coverage, "
                f"{existing_filter_stats['dropped_query_count'] - existing_filter_stats['partial_query_count']} queries dropped with no ingested positives"
            )
            for doc_id, document in filtered_corpus.items():
                combined_text = combine_document_text(document.get("title", ""), document.get("text", ""))
                content_to_doc_id[combined_text] = doc_id
        elif had_existing_index and set(filtered_corpus).issubset(existing_doc_ids):
            for doc_id, document in filtered_corpus.items():
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
                corpus=filtered_corpus,
                ingest_batch_size=ingest_batch_size,
                existing_doc_ids=existing_doc_ids,
            )
            ingest_marker.write_text(settings.graph_group_id, encoding="utf-8")

        retrieval_results: dict[str, dict[str, float]] = {}
        known_doc_ids = set(filtered_corpus)
        total_queries = len(filtered_queries)
        for index, query_id in enumerate(sorted(filtered_queries), start=1):
            retrieval_results[query_id] = await rank_documents_for_query(
                service=service,
                query_text=filtered_queries[query_id],
                content_to_doc_id=content_to_doc_id,
                known_doc_ids=known_doc_ids,
                top_k=top_k,
                search_limit=search_limit,
            )
            print(f"Evaluated {index}/{total_queries} queries ({query_id})")

        k_values = sorted({1, 3, 5, top_k})
        ndcg, map_scores, recall, precision = EvaluateRetrieval.evaluate(
            filtered_qrels,
            retrieval_results,
            k_values,
        )
        return (
            {
                "metrics": {
                    "ndcg": ndcg,
                    "map": map_scores,
                    "recall": recall,
                    "precision": precision,
                },
                "rankings": retrieval_results,
            },
            filtered_corpus,
            filtered_queries,
            filtered_qrels,
            existing_filter_stats,
        )
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
            "max_completion_tokens": args.max_completion_tokens,
            "evaluate_existing_only": args.evaluate_existing_only,
            "existing_group_id": args.existing_group_id,
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

    evaluation_result, corpus, queries, qrels, existing_filter_stats = await evaluate_graphiti_subset(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        runtime_dir=runtime_dir,
        top_k=args.top_k,
        search_limit=max(args.search_limit, args.top_k),
        ingest_batch_size=args.ingest_batch_size,
        max_completion_tokens=args.max_completion_tokens,
        evaluate_existing_only=args.evaluate_existing_only,
        existing_group_id=args.existing_group_id,
        reuse_index=args.reuse_index,
    )
    if existing_filter_stats is not None:
        metadata = {
            **metadata,
            "subset_doc_count": existing_filter_stats["existing_doc_count"],
            "query_count": existing_filter_stats["retained_query_count"],
            "negative_doc_count": 0,
            "existing_only": True,
            "full_positive_doc_coverage_required": True,
            "partial_query_count": existing_filter_stats["partial_query_count"],
            "dropped_query_count": existing_filter_stats["dropped_query_count"],
        }
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