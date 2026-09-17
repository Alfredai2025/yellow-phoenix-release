#!/usr/bin/env python3
"""SAH Stage 6: Assign cluster_tag to documents based on nearest beacon."""

import sys
sys.path.insert(0, "/Users/mac/yellow_phoenix")

from yp_bridge import RustBridge


def assign_cluster_tags(bridge: RustBridge, doc_hashes: list, doc_ids: list) -> dict:
    """
    For each document hash, find nearest beacon and assign its cluster.
    Returns {doc_id: cluster_tag_int}.
    """
    assignments = {}
    for doc_id, doc_hash in zip(doc_ids, doc_hashes):
        # Probe beacon index with document hash
        beacons = bridge.beacon_index_search(doc_hash, k=1)
        if beacons:
            # beacons[0] = (node_id, distance, tag)
            # Use tag as cluster identifier
            _, dist, tag = beacons[0]
            if dist <= 6:  # Within weak hit range
                assignments[doc_id] = tag
            else:
                assignments[doc_id] = 0  # Unclustered
        else:
            assignments[doc_id] = 0
    return assignments


def batch_retag_documents(bridge: RustBridge, doc_hashes: list, doc_ids: list) -> dict:
    """Re-tag all documents. Returns stats."""
    tags = assign_cluster_tags(bridge, doc_hashes, doc_ids)
    clustered = sum(1 for t in tags.values() if t != 0)
    return {
        "total": len(doc_ids),
        "clustered": clustered,
        "unclustered": len(doc_ids) - clustered,
        "tag_distribution": {hex(k): sum(1 for v in tags.values() if v == k)
                            for k in set(tags.values())}
    }
