# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import hashlib
import re
import uuid


def generate_chunk_id(document_id: str, chunk_text: str, chunk_window: str | None = None) -> str:
    """
    Generate a unique chunk ID using a hash of the document ID and chunk text.

    Note: MD5 is used only to calculate an identifier, not for security purposes.
    Adding usedforsecurity=False for compatibility with FIPS environments.
    """
    hash_input = f"{document_id}:{chunk_text}".encode()
    if chunk_window:
        hash_input += f":{chunk_window}".encode()
    return str(uuid.UUID(hashlib.md5(hash_input, usedforsecurity=False).hexdigest()))


def proper_case(s: str) -> str:
    """Convert a string to proper case (first letter uppercase, rest lowercase)."""
    return s[0].upper() + s[1:].lower() if s else s


def sanitize_collection_name(name: str, weaviate_format=False) -> str:
    """
    Sanitize collection name to ensure it only contains numbers, letters, and underscores.
    Any other characters are replaced with underscores.
    """
    if not weaviate_format:
        s = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    else:
        s = proper_case(re.sub(r"[^a-zA-Z0-9]", "", name))
    return s


class Reranker:
    @staticmethod
    def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
        """
        Normalize scores to 0-1 range using min-max normalization.
        Args:
            scores: dictionary of scores with document IDs as keys and scores as values
        Returns:
            Normalized scores with document IDs as keys and normalized scores as values
        """
        if not scores:
            return {}
        min_score, max_score = min(scores.values()), max(scores.values())
        score_range = max_score - min_score
        if score_range > 0:
            return {doc_id: (score - min_score) / score_range for doc_id, score in scores.items()}
        return dict.fromkeys(scores, 1.0)

    @staticmethod
    def weighted_rerank(
        vector_scores: dict[str, float],
        keyword_scores: dict[str, float],
        alpha: float = 0.5,
    ) -> dict[str, float]:
        """
        Rerank via weighted average of scores.
        Args:
            vector_scores: scores from vector search
            keyword_scores: scores from keyword search
            alpha: weight factor between 0 and 1 (default: 0.5)
                   0 = keyword only, 1 = vector only, 0.5 = equal weight
        Returns:
            All unique document IDs with weighted combined scores
        """
        all_ids = set(vector_scores.keys()) | set(keyword_scores.keys())
        normalized_vector_scores = Reranker._normalize_scores(vector_scores)
        normalized_keyword_scores = Reranker._normalize_scores(keyword_scores)

        # Weighted formula: score = (1-alpha) * keyword_score + alpha * vector_score
        # alpha=0 means keyword only, alpha=1 means vector only
        return {
            doc_id: ((1 - alpha) * normalized_keyword_scores.get(doc_id, 0.0))
            + (alpha * normalized_vector_scores.get(doc_id, 0.0))
            for doc_id in all_ids
        }

    @staticmethod
    def rrf_rerank(
        vector_scores: dict[str, float],
        keyword_scores: dict[str, float],
        impact_factor: float = 60.0,
    ) -> dict[str, float]:
        """
        Rerank via Reciprocal Rank Fusion.
        Args:
            vector_scores: scores from vector search
            keyword_scores: scores from keyword search
            impact_factor: impact factor for RRF (default: 60.0)
        Returns:
            All unique document IDs with RRF combined scores
        """

        # Convert scores to ranks
        vector_ranks = {
            doc_id: i + 1
            for i, (doc_id, _) in enumerate(sorted(vector_scores.items(), key=lambda x: x[1], reverse=True))
        }
        keyword_ranks = {
            doc_id: i + 1
            for i, (doc_id, _) in enumerate(sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True))
        }

        all_ids = set(vector_scores.keys()) | set(keyword_scores.keys())
        rrf_scores = {}
        for doc_id in all_ids:
            vector_rank = vector_ranks.get(doc_id, float("inf"))
            keyword_rank = keyword_ranks.get(doc_id, float("inf"))

            # RRF formula: score = 1/(k + r) where k is impact_factor (default: 60.0) and r is the rank
            rrf_scores[doc_id] = (1.0 / (impact_factor + vector_rank)) + (1.0 / (impact_factor + keyword_rank))
        return rrf_scores

    @staticmethod
    def combine_search_results(
        vector_scores: dict[str, float],
        keyword_scores: dict[str, float],
        reranker_type: str = "rrf",
        reranker_params: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """
        Combine vector and keyword search results using specified reranking strategy.
        Args:
            vector_scores: scores from vector search
            keyword_scores: scores from keyword search
            reranker_type: type of reranker to use (default: RERANKER_TYPE_RRF)
            reranker_params: parameters for the reranker
        Returns:
            All unique document IDs with combined scores
        """
        if reranker_params is None:
            reranker_params = {}

        if reranker_type == "weighted":
            alpha = reranker_params.get("alpha", 0.5)
            return Reranker.weighted_rerank(vector_scores, keyword_scores, alpha)
        else:
            # Default to RRF for None, RRF, or any unknown types
            impact_factor = reranker_params.get("impact_factor", 60.0)
            return Reranker.rrf_rerank(vector_scores, keyword_scores, impact_factor)
