#!/usr/bin/env python3
"""
PIDSearchPath: wraps any YP engine and wires PID into the actual search path.
Cache → Cascade → Router → HNSW. Falls back safely if any layer is missing.
"""

import time
import numpy as np
from scripts.pid_extensions import PIDManager


class PIDSearchPath:
    """
    Wrapper that adds PID-integrated search to any engine with:
      - search_hybrid_by_embedding(q, k, scale)
    Optional: _cascade_prefix_search(q, k, l1, l2, l3), search_geometric(q, k, scale)
    """

    def __init__(self, engine, pid_manager=None,
                 enable_cache=True, enable_cascade=True, enable_router=True):
        self.engine = engine
        self.pid = pid_manager or PIDManager()
        self.enable_cache = enable_cache
        self.enable_cascade = enable_cascade
        self.enable_router = enable_router

        # Init cache if requested
        self.cache = None
        if enable_cache:
            try:
                from result_cache import ResultCache
                self.cache = ResultCache()
            except Exception as e:
                print(f"[PIDSearchPath] Cache init failed: {e}")
                self.enable_cache = False

        # Metrics accumulator for PID feedback
        self._reset_metrics()

    def _reset_metrics(self):
        self._metrics = {
            'cache_hits': 0, 'cache_misses': 0,
            'cascade_l1': 0, 'cascade_l2': 0, 'cascade_l3': 0,
            'cascade_miss': 0,
            'geo_routed': 0, 'fast_routed': 0,
            'latencies': [],
        }

    def _cache_key(self, embedding):
        """Fast hash of embedding for cache key."""
        return hash(embedding.tobytes()) & 0xFFFFFFFFFFFFFFFF

    def search(self, query_embedding, k=10, scale="1m"):
        """
        Full PID path:
        1. Check result cache
        2. Try cascade prefix match
        3. Router decides geometric vs fast
        4. Execute search
        5. Store in cache + record latency
        """
        t0 = time.perf_counter()

        # ── 1. CACHE ──────────────────────────────────────────────
        if self.enable_cache and self.cache:
            key = self._cache_key(query_embedding)
            cached = self.cache.get(key)
            if cached is not None:
                self._metrics['cache_hits'] += 1
                return cached
            self._metrics['cache_misses'] += 1

        # ── 2. CASCADE ────────────────────────────────────────────
        cascade_results = None
        if self.enable_cascade and hasattr(self.engine, '_cascade_prefix_search'):
            thresholds = self.pid.cascade.thresholds
            try:
                cascade_results = self.engine._cascade_prefix_search(
                    query_embedding, k=k,
                    l1_threshold=thresholds['L1'],
                    l2_threshold=thresholds['L2'],
                    l3_threshold=thresholds['L3']
                )
            except Exception as e:
                # Cascade failed — fall through to HNSW
                cascade_results = None

            if cascade_results:
                level = cascade_results.get('level', 'L1')
                self._metrics[f'cascade_{level.lower()}'] += 1
                if self.enable_cache and self.cache:
                    self.cache.put(key, cascade_results['results'])
                return cascade_results['results']
            else:
                self._metrics['cascade_miss'] += 1

        # ── 3. ROUTER ─────────────────────────────────────────────
        use_geometric = False
        if self.enable_router:
            route_pct = self.pid.router.route_percent
            # Deterministic but pseudo-random routing based on query hash
            use_geometric = (hash(query_embedding.tobytes()) % 100) < route_pct

        # ── 4. SEARCH ─────────────────────────────────────────────
        if use_geometric and hasattr(self.engine, 'search_geometric'):
            results = self.engine.search_geometric(query_embedding, k=k, scale=scale)
            self._metrics['geo_routed'] += 1
        else:
            results = self.engine.search_hybrid_by_embedding(query_embedding, k=k, scale=scale)
            self._metrics['fast_routed'] += 1

        # ── 5. METRICS + CACHE STORE ──────────────────────────────────────────────
        latency = (time.perf_counter() - t0) * 1e6
        self._metrics['latencies'].append(latency)

        if self.enable_cache and self.cache:
            self.cache.put(key, results)

        return results

    def step_pid(self, n_queries=0):
        """
        Advance all three PID controllers with accumulated metrics.
        Call this every N queries (e.g., every 50 or 100).
        """
        # Cascade PID
        total_cascade = (self._metrics['cascade_l1'] +
                         self._metrics['cascade_l2'] +
                         self._metrics['cascade_l3'] +
                         self._metrics['cascade_miss'])
        if total_cascade > 0 and self.enable_cascade:
            self.pid.cascade.step(
                self._metrics['cascade_l1'],
                self._metrics['cascade_l2'],
                self._metrics['cascade_l3'],
                total_cascade
            )

        # Router PID
        if self._metrics['latencies'] and self.enable_router:
            p50 = float(np.percentile(self._metrics['latencies'], 50))
            # Fast-path precision proxy: cascade L1 hit rate
            total = total_cascade if total_cascade > 0 else 1
            fast_precision = self._metrics['cascade_l1'] / total
            self.pid.router.step(p50, fast_precision, n_queries)

        # Cache PID
        if self.enable_cache and self.cache:
            hits = self._metrics['cache_hits']
            misses = self._metrics['cache_misses']
            if hits + misses > 0:
                self.pid.cache.step(hits, misses)

        # Reset for next window
        self._reset_metrics()

    def state(self):
        """Current PID state for logging."""
        return {
            'cascade_thresholds': self.pid.cascade.thresholds.copy(),
            'route_percent': round(self.pid.router.route_percent, 1),
            'cache_ttl': self.pid.cache.ttl_seconds,
        }
