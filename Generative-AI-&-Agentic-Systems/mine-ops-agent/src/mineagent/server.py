"""MCP server exposing the mine safety tools.

    # stdio (used by the agent in this repo)
    DATA_DIR=... REGS_DIR=... python -m mineagent.server
    # HTTP, so Dify or another MCP client can connect
    DATA_DIR=... REGS_DIR=... python -m mineagent.server --http --port 8000
"""
import argparse
import json
import os
import tempfile

from mcp.server.fastmcp import FastMCP

from .data import MineData, ToolInputError

mcp = FastMCP("mine-safety")
_data: MineData | None = None
METRICS = {"violations", "ss_violations", "inspections", "proposed_penalty", "injuries", "accidents"}


def data() -> MineData:
    global _data
    if _data is None:
        _data = MineData(os.environ["DATA_DIR"], os.environ.get("REGS_DIR"))
    return _data


def _safe(fn, *args, **kwargs) -> str:
    """Tools return JSON text; bad input comes back as an error the model can fix."""
    try:
        return json.dumps(fn(*args, **kwargs), default=str)
    except ToolInputError as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def find_mines(state: str | None = None, name_contains: str | None = None, limit: int = 10) -> str:
    """Look up mines by two-letter state code and/or part of the mine name. Returns mine IDs."""
    return _safe(data().find_mines, state, name_contains, limit)


@mcp.tool()
def get_mine_profile(mine_id: str) -> str:
    """Basic facts about one mine: name, state, type, status, commodity, operator, employees."""
    return _safe(data().get_mine_profile, mine_id)


@mcp.tool()
def get_violation_history(mine_id: str, quarters: int = 8) -> str:
    """Inspections, violations, significant-and-substantial (S&S) violations and proposed
    penalties per quarter, for the most recent `quarters` quarters in the data, with totals."""
    return _safe(data().get_violation_history, mine_id, quarters)


@mcp.tool()
def get_accident_summary(mine_id: str, years: int = 5) -> str:
    """Accidents, injuries, days lost, serious accidents, and the most common accident types
    and equipment for one mine over the last `years` years in the data."""
    return _safe(data().get_accident_summary, mine_id, years)


@mcp.tool()
def search_regulations(query: str, part: str | None = None, k: int = 3) -> str:
    """Search U.S. mine safety regulations (30 CFR). Optionally limit to a part such as
    '56' (surface metal/non-metal), '57' (underground metal/non-metal), '75' (underground coal)."""
    return _safe(data().search_regulations, query, part, k)


@mcp.tool()
def plot_trend(mine_id: str, metric: str = "violations", quarters: int = 12) -> str:
    """Save a line chart of one quarterly metric for a mine and return the image path.
    metric: violations, ss_violations, inspections, proposed_penalty, injuries or accidents."""
    if metric not in METRICS:
        return json.dumps({"error": f"metric must be one of {sorted(METRICS)}"})
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mine_id_ok = data().get_mine_profile(mine_id)["mine_id"]
        q = data().quarterly
        q = q[(q["MINE_ID"] == mine_id_ok) & (q["q_index"] > data().latest_q_index - quarters)].sort_values("q_index")
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot([f"{y}-Q{qq}" for y, qq in zip(q["year"], q["quarter"])], q[metric], marker="o")
        ax.set_title(f"{metric.replace('_', ' ')} per quarter, mine {mine_id_ok}")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        out_dir = os.environ.get("PLOT_DIR", tempfile.gettempdir())
        path = os.path.join(out_dir, f"{mine_id_ok}_{metric}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return json.dumps({"image_path": path})
    except ToolInputError as e:
        return json.dumps({"error": str(e)})


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--http", action="store_true")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)
    if args.http:
        mcp.settings.port = args.port
        mcp.run(transport="streamable-http")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
