#!/usr/bin/env python
# filepath: c:\Users\Zachv\Documents\svc-ai\scripts\data_ingestion\vector_store_init.py

"""
Vector Store Initialization Script for RAG (updated for new data_prep format)

Changes:
- Auto-detects chunk boundaries produced by the new data_prep.py
  format: lines like `--- CHUNK {i} TOKENS={n} ---` (regex-based).
- Still supports legacy custom separators via --chunk-separator.
- Extracts optional per-chunk metadata (index, token count) for logging.
- Verifies chunk counts against document_index.json when --verify-chunks is set.
- FIX: Uses pre-computed chunks directly (no re-splitting inside RAGService).
- FIX: Embedding issue resolved by supplying a proper embedding function (SentenceTransformer wrapper) to Chroma.

Example usage:
    python vector_store_init.py --document-index "vision-vs/processed-docs/document_index.json"
    python vector_store_init.py --document-index ".../document_index.json" --collection vision-docs
    python vector_store_init.py --document-index ".../document_index.json" --verify-chunks
"""

import os
import sys
import json
import argparse
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Setup logging (will be updated after importing settings)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path to allow importing from src
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import model service & settings (RAGService not needed for ingestion now)
from src.cpe_ai.services.model_service import ModelService
from src.cpe_ai.config.settings import (
    VECTOR_STORE_PATH,
    LOG_LEVEL,
)

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import torch

# Update logging level based on settings
logging.getLogger().setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# Regex for new chunk header lines written by data_prep v2
CHUNK_HEADER_RE = re.compile(r"^---\s*CHUNK\s*(?P<idx>\d+)\s*TOKENS=(?P<tokens>\d+)\s*---\s*$", re.MULTILINE)

# ----------------------------
# Argument Parsing
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize a vector store from processed documents (auto chunk detection, no re-chunking)"
    )
    parser.add_argument(
        "--document-index",
        type=str,
        required=True,
        help="Path to the document index JSON file containing all files to process",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="default",
        help="Name of the collection to create in the vector store",
    )
    parser.add_argument(
        "--chunk-separator",
        type=str,
        default=None,
        help=(
            "Legacy/manual separator used between chunks in processed files. "
            "If omitted, the script auto-detects the new header format (`--- CHUNK i TOKENS=n ---`)."
        ),
    )
    parser.add_argument(
        "--verify-chunks",
        action="store_true",
        help="Verify that the number of chunks in each file matches the document index",
    )
    return parser.parse_args()

# ----------------------------
# Helpers
# ----------------------------

def safe_collection_name(name: str) -> str:
    return ''.join(c if c.isalnum() else '_' for c in name)

def load_document_index(index_path: str) -> Dict[str, Any]:
    try:
        with open(Path(index_path), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading document index: {e}")
        return {}


def autodetect_and_split(content: str, chunk_separator: Optional[str] = None) -> List[Tuple[str, Optional[int], Optional[int]]]:
    """Split file content into chunks.

    Returns a list of tuples: (chunk_text, idx, token_count)
    where idx and token_count are parsed from the header if available.

    Priority:
    1) If chunk_separator is provided and found, split by it (legacy support).
    2) Else, use regex to split on `--- CHUNK {i} TOKENS={n} ---` headers.
    3) Else, treat the entire file as a single chunk.
    """
    # 1) Manual separator path
    if chunk_separator:
        if chunk_separator in content:
            parts = [p.strip() for p in content.split(chunk_separator) if p.strip()]
            return [(p, None, None) for p in parts]
        else:
            logger.warning("Provided --chunk-separator not found in file; falling back to auto-detect.")

    # 2) Regex header path (new data_prep format)
    matches = list(CHUNK_HEADER_RE.finditer(content))
    if matches:
        chunks: List[Tuple[str, Optional[int], Optional[int]]] = []
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            body = content[start:end].strip()
            if not body:
                continue
            try:
                idx = int(m.group("idx"))
            except Exception:
                idx = i + 1
            try:
                tokens = int(m.group("tokens"))
            except Exception:
                tokens = None
            chunks.append((body, idx, tokens))
        return chunks

    # 3) Fallback: single chunk
    body = content.strip()
    return [(body, None, None)] if body else []


def count_chunks_in_text(content: str, chunk_separator: Optional[str]) -> int:
    return len(autodetect_and_split(content, chunk_separator))


def count_chunks_in_file(file_path: Path, chunk_separator: Optional[str]) -> int:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return count_chunks_in_text(content, chunk_separator)
    except Exception as e:
        logger.error(f"Error counting chunks in {file_path}: {e}")
        return 0


# ----------------------------
# Ingestion Logic (no re-chunking)
# ----------------------------

def ingest_file_as_chunks(
    vector_store: Chroma,
    file_path: Path,
    chunk_separator: Optional[str],
    expected_chunks: Optional[int] = None,
) -> Tuple[bool, Dict[str, Any]]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        triplets = autodetect_and_split(content, chunk_separator)
        if expected_chunks is not None and len(triplets) != expected_chunks:
            logger.warning(
                f"Expected {expected_chunks} chunks in {file_path.name} but found {len(triplets)}"
            )
        logger.info(f"Found {len(triplets)} pre-computed chunks in {file_path.name}")
        docs: List[Document] = []
        for i, (chunk_text, idx, tokens) in enumerate(triplets, start=1):
            chunk_idx = idx if idx is not None else i
            metadata = {
                "source": file_path.name,
                "chunk_index": chunk_idx,
                "tokens_reported": tokens,
            }
            docs.append(Document(page_content=chunk_text, metadata=metadata))
        if not docs:
            logger.warning(f"No chunks produced for {file_path.name}")
            return False, {"status": "empty"}
        vector_store.add_documents(docs)
        return True, {
            "status": "success",
            "chunks_processed": len(docs),
            "chunks_found": len(triplets),
            "expected_chunks": expected_chunks,
        }
    except Exception as e:
        logger.error(f"Error ingesting {file_path.name}: {e}")
        return False, {"status": "error", "error": str(e)}

# ----------------------------
# Main
# ----------------------------

def main():
    args = parse_args()

    # Load embedding & reranker models ONLY (skip QA to save memory/time)
    logger.info("Initializing ModelService (embedding + reranker only)...")
    model_service = ModelService()
    model_service.load_embedding_model()
    model_service.load_reranker_model()
    model_service.create_retriever()

    if model_service.embedding_model is None:
        logger.error("Embedding model failed to load; cannot proceed.")
        sys.exit(1)

    collection_name_raw = args.collection
    safe_name = safe_collection_name(collection_name_raw)
    collection_path = os.path.join(VECTOR_STORE_PATH, safe_name)
    os.makedirs(collection_path, exist_ok=True)

    logger.info(f"Using collection: {collection_name_raw} -> {safe_name}")

    # Always create/open Chroma with embedding function
    vector_store = Chroma(
        collection_name=safe_name,
        embedding_function=model_service.retriever,
        persist_directory=collection_path
    )

    # Load document index
    index_path = args.document_index
    if not Path(index_path).exists():
        logger.error(f"Document index file not found: {index_path}")
        sys.exit(1)

    logger.info(f"Loading document index from {index_path}")
    document_index = load_document_index(index_path)
    if not document_index:
        logger.error("Document index is empty or could not be loaded")
        sys.exit(1)

    base_dir = str(Path(index_path).parent)
    chunk_separator = args.chunk_separator
    verify_chunks = args.verify_chunks

    logger.info(f"Ingesting documents into collection: {collection_name_raw}")
    logger.info("Chunk boundary mode: %s", ("manual separator" if chunk_separator else "auto-detect (header regex)"))

    successful_ingestions = 0
    failed_ingestions = 0
    skipped_ingestions = 0
    processing_results: Dict[str, Any] = {}

    for rel_file_path, metadata in document_index.items():
        file_path = Path(base_dir) / rel_file_path
        logger.info(f"Processing file: {file_path}")
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            skipped_ingestions += 1
            processing_results[rel_file_path] = {"status": "skipped", "reason": "file not found"}
            continue
        expected_chunks = None
        if verify_chunks and isinstance(metadata, dict) and "chunks" in metadata:
            expected_chunks = metadata.get("chunks")
        logger.info(f"Expected chunks: {expected_chunks if expected_chunks is not None else 'Unknown'}")
        success, result = ingest_file_as_chunks(
            vector_store=vector_store,
            file_path=file_path,
            chunk_separator=chunk_separator,
            expected_chunks=expected_chunks,
        )
        if success:
            successful_ingestions += 1
            processing_results[rel_file_path] = result
        else:
            failed_ingestions += 1
            processing_results[rel_file_path] = result

    # Persist vector store
    try:
        vector_store.persist()
        logger.info("Vector store persisted to disk")
    except Exception as e:
        logger.warning(f"Failed to persist vector store explicitly (it may still be saved): {e}")

    logger.info("Vector store initialization complete")
    logger.info(f"Collection: {collection_name_raw}")
    logger.info(f"Total documents in index: {len(document_index)}")
    logger.info(f"Successfully ingested: {successful_ingestions}")
    logger.info(f"Failed to ingest: {failed_ingestions}")
    logger.info(f"Skipped (not found): {skipped_ingestions}")
    logger.info(f"Vector store path: {VECTOR_STORE_PATH}")

    results_path = os.path.join(os.path.dirname(index_path), f"ingestion_results_{collection_name_raw}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(processing_results, f, indent=2)
    logger.info(f"Saved ingestion results to {results_path}")

if __name__ == "__main__":
    main()