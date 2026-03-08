"""
Correlation-Aware Equity Selection — prevents secret beta concentration.

Problem: "Top 3 momentum" can select SPY, QQQ, XLK which are all
highly correlated US growth equity. In a stress event they collapse
together — you're not diversified, just holding 3 flavors of the same bet.

Solution: Cluster assets by return correlation, then enforce max picks
per cluster. Forces the portfolio to span different risk exposures.

Clusters (predefined, based on known market structure):
- US Growth: SPY, QQQ, XLK, XLY, IWM
- US Defensive: XLP, XLU, XLV
- US Cyclical: XLF, XLE, XLI, XLB, XLRE
- International: VGK, VWO, EFA, EEM
- Real Assets: GLD, SLV, VNQ
- Fixed Income: AGG, TLT, HYG

Usage:
    from risk.correlation import filter_correlated_picks
    filtered = filter_correlated_picks(ranked_symbols, max_per_cluster=1)
"""

from core.logging import get_logger

logger = get_logger("risk.correlation")

# Predefined clusters based on known market structure.
# This is more robust than computing rolling correlations on a small account
# with limited history — the cluster memberships are stable facts about
# these ETFs, not parameters to be estimated.
EQUITY_CLUSTERS = {
    "us_growth": {"SPY", "QQQ", "XLK", "XLY", "IWM"},
    "us_defensive": {"XLP", "XLU", "XLV"},
    "us_cyclical": {"XLF", "XLE", "XLI", "XLB", "XLRE"},
    "international": {"VGK", "VWO", "EFA", "EEM"},
    "real_assets": {"GLD", "SLV", "VNQ"},
    "fixed_income": {"AGG", "TLT", "HYG", "SHY"},
}

# Default max picks from any single cluster
DEFAULT_MAX_PER_CLUSTER = 1


def get_cluster(symbol):
    """
    Get the cluster name for a symbol.

    Returns:
        Cluster name string, or "unclustered" if not in any predefined cluster.
    """
    for cluster_name, members in EQUITY_CLUSTERS.items():
        if symbol in members:
            return cluster_name
    return "unclustered"


def filter_correlated_picks(ranked_symbols, max_per_cluster=None, existing_holdings=None):
    """
    Filter a ranked list of symbols to enforce cluster diversity.

    Takes the momentum-ranked list and removes picks that would
    over-concentrate in any single cluster.

    Args:
        ranked_symbols: List of (symbol, score) tuples, sorted by score descending.
                        Or list of signal dicts with "symbol" key.
        max_per_cluster: Max picks from any single cluster. Default 1.
        existing_holdings: List of symbols already held (count toward cluster limits).

    Returns:
        Filtered list in the same format as input, with concentrated picks removed.
    """
    if max_per_cluster is None:
        max_per_cluster = DEFAULT_MAX_PER_CLUSTER

    if existing_holdings is None:
        existing_holdings = []

    # Count existing cluster usage
    cluster_counts = {}
    for sym in existing_holdings:
        cluster = get_cluster(sym)
        cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1

    filtered = []
    removed = []

    for item in ranked_symbols:
        # Support both (symbol, score) tuples and signal dicts
        if isinstance(item, dict):
            symbol = item["symbol"]
        elif isinstance(item, (tuple, list)):
            symbol = item[0]
        else:
            symbol = str(item)

        # Crypto symbols bypass cluster filtering
        if "/" in symbol:
            filtered.append(item)
            continue

        # Safe assets bypass cluster filtering (always allow rotation to bonds)
        cluster = get_cluster(symbol)
        if cluster == "fixed_income":
            filtered.append(item)
            continue

        current_count = cluster_counts.get(cluster, 0)
        if current_count >= max_per_cluster:
            removed.append((symbol, cluster))
            logger.info(
                f"Correlation filter: BLOCKED {symbol} (cluster={cluster}, "
                f"already have {current_count}/{max_per_cluster})",
            )
            continue

        # Accept this pick and increment cluster count
        cluster_counts[cluster] = current_count + 1
        filtered.append(item)

    if removed:
        logger.info(
            f"Correlation filter: removed {len(removed)} concentrated pick(s)",
            extra={"extra_data": {
                "removed": [(s, c) for s, c in removed],
                "cluster_counts": cluster_counts,
                "max_per_cluster": max_per_cluster,
            }},
        )

    return filtered


def check_portfolio_concentration(holdings, max_per_cluster=None):
    """
    Check if current holdings are over-concentrated in any cluster.

    Args:
        holdings: List of held symbols.
        max_per_cluster: Max allowed per cluster.

    Returns:
        Dict with concentration analysis:
        {
            "concentrated": bool,
            "clusters": {cluster: [symbols]},
            "violations": [(cluster, count, max)],
        }
    """
    if max_per_cluster is None:
        max_per_cluster = DEFAULT_MAX_PER_CLUSTER

    clusters = {}
    for sym in holdings:
        cluster = get_cluster(sym)
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(sym)

    violations = [
        (cluster, len(syms), max_per_cluster)
        for cluster, syms in clusters.items()
        if len(syms) > max_per_cluster and cluster != "fixed_income"
    ]

    return {
        "concentrated": len(violations) > 0,
        "clusters": clusters,
        "violations": violations,
    }
