"""
Script to check and fix trip edge path directions in truck traffic flow data.

This script analyzes the Edge_path_E_road for each trip and determines if the path
is oriented correctly (first edge near origin region, last edge near destination region).
If the path is reversed, it will be fixed by reversing the edge list.

Usage:
    python fix_traffic_paths.py                    # Analyze only, print statistics
    python fix_traffic_paths.py --fix              # Analyze and save fixed data
    python fix_traffic_paths.py --fix --output output.csv  # Save to custom file
"""

import pandas as pd
import ast
import argparse
from pathlib import Path
from tqdm import tqdm


def load_data(data_dir: Path, custom_traffic_file=None) -> tuple:
    """Load all required data files."""
    edges_file = data_dir / "04_network-edges.csv"
    regions_file = data_dir / "02_NUTS-3-Regions.csv"

    # Use provided input file or default
    if custom_traffic_file:
        traffic_file = Path(custom_traffic_file)
    else:
        traffic_file = data_dir / "01_Trucktrafficflow.csv"

    edges_df = pd.read_csv(edges_file)
    regions_df = pd.read_csv(regions_file)
    traffic_df = pd.read_csv(traffic_file)

    return edges_df, regions_df, traffic_df


def build_lookups(edges_df: pd.DataFrame, regions_df: pd.DataFrame) -> tuple:
    """Build lookup dictionaries for efficient querying."""
    # Edge ID to nodes lookup
    edge_id_to_nodes = {}
    for _, row in edges_df.iterrows():
        eid = row["Network_Edge_ID"]
        u, v = row["Network_Node_A_ID"], row["Network_Node_B_ID"]
        edge_id_to_nodes[eid] = (u, v)

    # Region ID to network node lookup
    region_to_node = {}
    for _, row in regions_df.iterrows():
        zone_id = row["ETISPlus_Zone_ID"]
        node_id = row["Network_Node_ID"]
        if zone_id in region_to_node:
            raise ValueError(f"Duplicate region ID found: {zone_id}")
        region_to_node[zone_id] = node_id

    return edge_id_to_nodes, region_to_node


def check_path_orientation(
    origin_region_id: int,
    dest_region_id: int,
    edge_path: list,
    edge_id_to_nodes: dict,
    region_to_node: dict,
) -> tuple:
    """
    Check if the edge path is correctly oriented using direct node matching only.

    Returns:
        (keep, should_reverse)
        - keep: True if trip should be kept (direct match found)
        - should_reverse: True if path should be reversed to fix orientation
    """
    if not edge_path:
        return False, False

    first_edge_id = edge_path[0]
    last_edge_id = edge_path[-1]

    first_nodes = edge_id_to_nodes.get(first_edge_id)
    last_nodes = edge_id_to_nodes.get(last_edge_id)

    if first_nodes is None or last_nodes is None:
        return False, False

    first_nodes, last_nodes = set(first_nodes), set(last_nodes)

    # Get the origin and destination network nodes
    origin_node = region_to_node.get(origin_region_id)
    dest_node = region_to_node.get(dest_region_id)

    if origin_node is None or dest_node is None:
        return False, False

    # Check if origin/destination nodes are among the endpoints
    if origin_node in first_nodes and dest_node in last_nodes:
        return True, False  # Correct orientation
    elif origin_node in last_nodes and dest_node in first_nodes:
        return True, True  # Reversed orientation

    # No direct match found
    return False, False


def analyze_and_fix_traffic(
    traffic_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    regions_df: pd.DataFrame,
    fix: bool = False,
) -> pd.DataFrame:
    """
    Analyze all trips and optionally fix reversed paths.
    Only keeps trips that can be determined by direct match.
    Discards trips that cannot be determined.

    Returns the (possibly modified) traffic DataFrame.
    """
    # Build lookups
    edge_id_to_nodes, region_to_node = build_lookups(edges_df, regions_df)

    stats = {
        "total": 0,
        "correct": 0,
        "reversed": 0,
        "discarded": 0,
    }

    # Track which rows to keep
    rows_to_keep = []

    # Work on a copy of the dataframe
    traffic_df = traffic_df.copy()

    for idx, row in tqdm(
        traffic_df.iterrows(), total=len(traffic_df), desc="Analyzing trips"
    ):
        stats["total"] += 1

        origin_region_id = row["ID_origin_region"]
        dest_region_id = row["ID_destination_region"]
        edge_path_str = row["Edge_path_E_road"]

        # Parse edge path
        try:
            edge_path = ast.literal_eval(edge_path_str)
        except (ValueError, SyntaxError):
            stats["discarded"] += 1
            continue

        if not edge_path:
            stats["discarded"] += 1
            continue

        # Check orientation
        keep, should_reverse = check_path_orientation(
            origin_region_id,
            dest_region_id,
            edge_path,
            edge_id_to_nodes,
            region_to_node,
        )

        if not keep:
            stats["discarded"] += 1
            continue

        rows_to_keep.append(idx)
        if should_reverse:
            stats["reversed"] += 1
            if fix:
                # Reverse the path and update directly in dataframe
                reversed_path = list(reversed(edge_path))
                traffic_df.at[idx, "Edge_path_E_road"] = str(reversed_path)
        else:
            stats["correct"] += 1

    # Print statistics
    kept_count = stats["correct"] + stats["reversed"]
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Total trips analyzed:    {stats['total']:>10,}")
    print(
        f"Kept (direct match):     {kept_count:>10,} ({100*kept_count/stats['total']:.2f}%)"
    )
    print(
        f"  Correctly oriented:    {stats['correct']:>10,} ({100*stats['correct']/stats['total']:.2f}%)"
    )
    print(
        f"  Reversed (fixed):      {stats['reversed']:>10,} ({100*stats['reversed']/stats['total']:.2f}%)"
    )
    print(
        f"Discarded:               {stats['discarded']:>10,} ({100*stats['discarded']/stats['total']:.2f}%)"
    )
    print("=" * 60)

    if fix:
        # Filter to only keep rows with direct match
        result_df = traffic_df.loc[rows_to_keep]
        print(f"\nKept {len(result_df)} trips with direct match.")
        print(f"Fixed {stats['reversed']} reversed paths.")
        print(
            f"Discarded {stats['discarded']} trips that could not be determined by direct match."
        )
        return result_df
    else:
        # Return original dataframe for analysis-only mode
        return traffic_df


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Check and fix trip edge path directions in truck traffic flow data."
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        type=Path,
        default=script_dir / "data",
        help="Directory containing the data files (default: <script_dir>/data)",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=None,
        help="Input traffic flow file (default: 01_Trucktrafficflow.csv in data dir)",
    )
    parser.add_argument(
        "--fix", action="store_true", help="Fix reversed paths and save to output file"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("01_Trucktrafficflow_fixed.csv"),
        help="Output file path (default: 01_Trucktrafficflow_fixed.csv in data dir)",
    )

    args = parser.parse_args()

    # Load data
    print("Loading data files...")
    edges_df, regions_df, traffic_df = load_data(args.data_dir, args.input)
    print(f"Loaded {len(traffic_df)} trips")

    # Analyze and optionally fix
    print("Analyzing trip orientations...")
    fixed_df = analyze_and_fix_traffic(traffic_df, edges_df, regions_df, fix=args.fix)

    # Save if fixing
    if args.fix:
        output_path: Path = args.output
        if not output_path.is_absolute():
            output_path = args.data_dir / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fixed_df.to_csv(output_path, index=False)
        print(f"\nSaved fixed data to: {output_path}")


if __name__ == "__main__":
    main()
