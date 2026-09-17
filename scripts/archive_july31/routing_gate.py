#!/usr/bin/env python3
"""
GeometricRoutingGate: fast-path vs deep-path decision for 10M query routing.
Phase 3 of the 10M master plan. Pure Python, additive only.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RoutingDecision:
    path: str          # "fast" or "deep"
    confidence: float  # 0.0–1.0
    reason: str        # human-readable rationale


class GeometricRoutingGate:
    """
    Decides whether a query can take the fast path (direct hash / spectral)
    or needs the deep path (shard → HNSW → PQ re-rank).
    """

    def __init__(self,
                 direct_hit_threshold: float = 0.95,
                 spectral_conf_threshold: float = 0.85,
                 cascade_hit_bonus: float = 0.10):
        self.direct_hit_threshold = direct_hit_threshold
        self.spectral_conf_threshold = spectral_conf_threshold
        self.cascade_hit_bonus = cascade_hit_bonus

        # Runtime tuning targets
        self.fast_path_recall = 1.0
        self.fast_path_fraction = 0.5

    def decide(self,
               direct_hash_score: Optional[float],
               spectral_confidence: Optional[float],
               cascade_tier: int = 0) -> RoutingDecision:
        """
        direct_hash_score: 1.0 = exact match, 0.0 = no match
        spectral_confidence: 0.0–1.0
        cascade_tier: 0=miss, 1=L1 hit, 2=L2 hit, 3=L3 hit
        """
        score = 0.0
        reasons = []

        if direct_hash_score is not None:
            score += direct_hash_score * 0.5
            reasons.append(f"direct={direct_hash_score:.2f}")

        if spectral_confidence is not None and spectral_confidence > 0:
            score += spectral_confidence * 0.2
            reasons.append(f"spectral={spectral_confidence:.2f}")

        score += min(cascade_tier, 3) * 0.15
        reasons.append(f"cascade_tier={cascade_tier}")

        # A strong direct hash hit alone is enough for the fast path.
        if direct_hash_score is not None and direct_hash_score >= self.direct_hit_threshold:
            return RoutingDecision(
                path="fast",
                confidence=direct_hash_score,
                reason="; ".join(reasons)
            )

        # Lower threshold so cascade can actually trigger fast path.
        threshold = 0.70
        if score >= threshold:
            return RoutingDecision(
                path="fast",
                confidence=score,
                reason="; ".join(reasons)
            )
        return RoutingDecision(
            path="deep",
            confidence=1.0 - score,
            reason="; ".join(reasons)
        )

    def update_thresholds_from_burn_in(self,
                                       fast_results: list,
                                       deep_results: list):
        """
        Stub for self-tuning loop integration.
        Adjust thresholds so fast path keeps >99% recall while maximizing fraction.
        """
        pass
