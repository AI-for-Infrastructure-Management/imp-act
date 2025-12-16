"""Lightweight validation for reduced graphs and OD traffic tables.

Checks
- Graph structure: no orphan nodes, unique edge IDs, numeric coordinates.
- Trips: all OD nodes exist in the graph and are connected (path exists).

Usage
- Import and call from create_large_graph.py before saving outputs.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import networkx as nx
import pandas as pd


def _is_finite_number(x) -> bool:
    try:
        return isinstance(x, (int, float)) and math.isfinite(float(x))
    except Exception:
        return False


def validate_graph_structure(G: nx.DiGraph) -> None:
    """Validate basic structural properties of the reduced directed graph.

    Raises ValueError with a summary if any check fails.
    """
    issues: List[str] = []

    # 1) No orphan nodes (in_degree + out_degree > 0)
    orphans = [n for n in G.nodes if G.in_degree(n) + G.out_degree(n) == 0]
    if orphans:
        issues.append(
            f"Orphan nodes (degree 0): {len(orphans)} examples: {orphans[:20]}"
        )

    # 2) Edge attribute 'id' consistency across directed edges
    #
    # After converting to directed, each undirected edge becomes two edges with
    # the same 'id' in opposite directions. That is allowed. Detect only cases
    # where an 'id' is reused across multiple distinct undirected pairs or
    # appears more than twice.
    from collections import defaultdict

    id_to_pairs = defaultdict(list)  # id -> list[(u,v)]
    missing_id_edges = []
    for u, v, data in G.edges(data=True):
        if "id" not in data:
            missing_id_edges.append((u, v))
            continue
        try:
            eid = int(data["id"])
        except Exception:
            issues.append(f"Non-integer edge id on edge ({u},{v}): {data.get('id')}")
            continue
        id_to_pairs[eid].append((u, v))

    if missing_id_edges:
        issues.append(
            f"Edges missing 'id' attribute: {len(missing_id_edges)} examples: {missing_id_edges[:20]}"
        )

    bad_id_usage = []
    for eid, pairs in id_to_pairs.items():
        undirected_sets = {frozenset(p) for p in pairs}
        # If an id maps to more than one undirected pair, it's a true duplicate
        if len(undirected_sets) > 1:
            bad_id_usage.append((eid, list(undirected_sets)[:3]))
            continue
        # If it appears more than twice, or exactly two but not reciprocal, flag
        if len(pairs) > 2:
            bad_id_usage.append((eid, pairs[:4]))
            continue
        if len(pairs) == 2:
            (u1, v1), (u2, v2) = pairs
            if not (u1 == v2 and v1 == u2):
                bad_id_usage.append((eid, pairs))

    if bad_id_usage:
        issues.append(
            "Edge 'id' reused across different edges or with invalid multiplicity: "
            f"{bad_id_usage[:10]}"
        )

    # 3) Coordinates present and numeric for all nodes
    bad_coords = [
        n
        for n, d in G.nodes(data=True)
        if not (
            _is_finite_number(d.get("position_x"))
            and _is_finite_number(d.get("position_y"))
        )
    ]
    if bad_coords:
        issues.append(
            f"Nodes with missing/non-numeric coordinates: {len(bad_coords)} examples: {bad_coords[:20]}"
        )

    if issues:
        raise ValueError("Graph validation failed:\n- " + "\n- ".join(issues))


def validate_trips_connectivity(
    G: nx.DiGraph,
    trips: pd.DataFrame,
    origin_col: str = "origin",
    destination_col: str = "destination",
) -> None:
    """Validate that OD nodes exist in the graph and are connected.

    Raises ValueError if any OD pair uses unknown nodes or lacks a path.
    """
    issues: List[str] = []

    # 1) All OD nodes exist
    nodes = set(G.nodes)
    missing_nodes: List[Tuple[int, int, str]] = []  # (row_idx, value, which)
    for idx, row in trips.iterrows():
        o = int(row[origin_col])
        d = int(row[destination_col])
        if o not in nodes:
            missing_nodes.append((idx, o, "origin"))
        if d not in nodes:
            missing_nodes.append((idx, d, "destination"))
    if missing_nodes:
        examples = missing_nodes[:20]
        issues.append(
            f"Trips reference nodes not present in graph: {len(missing_nodes)} examples: {examples}"
        )

    # 2) Connectivity for all OD pairs (directed)
    no_path_rows: List[int] = []
    for idx, row in trips.iterrows():
        o = int(row[origin_col])
        d = int(row[destination_col])
        if o in nodes and d in nodes:
            if not nx.has_path(G, o, d):
                no_path_rows.append(idx)
    if no_path_rows:
        issues.append(
            f"No directed path for some OD pairs: {len(no_path_rows)} examples: {no_path_rows[:20]}"
        )

    if issues:
        raise ValueError("Trips validation failed:\n- " + "\n- ".join(issues))
