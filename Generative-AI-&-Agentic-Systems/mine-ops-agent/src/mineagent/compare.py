"""Run the agent and the workflow on the same mines and compare them.

Report numbers are checked against the true values computed directly from the
data, so "accuracy" measures whether the LLM copied facts correctly.

    export GEMINI_API_KEY=...
    python -m mineagent.compare --data-dir ../mine-safety-lakehouse/lakehouse_data \
        --regs-dir ../regs-to-vectors/vectors --mines 20
"""
import argparse
import asyncio
import os

import pandas as pd

from .agent import run_agent, run_workflow
from .data import MineData

FACTS = ["total_violations", "total_ss_violations", "accidents", "injuries"]


def truth(data: MineData, mine_id: str) -> dict:
    v, a = data.get_violation_history(mine_id, 8), data.get_accident_summary(mine_id, 5)
    return {"total_violations": v["total_violations"], "total_ss_violations": v["total_ss_violations"],
            "accidents": a["accidents"], "injuries": a["injuries"]}


def score(result, expected: dict, valid_sections: set) -> dict:
    r = result.report
    row = {"mode": result.mode, "mine_id": result.mine_id, "valid_report": r is not None, "error": result.error,
           "tool_calls": len(result.tool_calls), "input_tokens": result.input_tokens,
           "output_tokens": result.output_tokens, "seconds": round(result.seconds, 2)}
    if r is not None:
        row["facts_correct"] = sum(getattr(r, f) == expected[f] for f in FACTS) / len(FACTS)
        cited = set(r.relevant_regulations)
        row["invented_citations"] = len(cited - valid_sections)
        row["risk_level"] = r.risk_level
    else:
        row["facts_correct"] = 0.0
    return row


def pick_mines(data: MineData, n: int, seed: int = 3) -> list[str]:
    """A spread of mines: some with many violations, some with few."""
    totals = data.quarterly.groupby("MINE_ID")["violations"].sum().sort_values()
    step = max(len(totals) // n, 1)
    return list(totals.index[::step][:n])


async def compare(data_dir, regs_dir, mine_ids, llm, tools_factory) -> pd.DataFrame:
    data = MineData(data_dir, regs_dir)
    rows = []
    async with tools_factory() as tools:
        for mine_id in mine_ids:
            expected = truth(data, mine_id)
            for runner in (run_agent, run_workflow):
                res = await runner(mine_id, llm, tools)
                rows.append(score(res, expected, res.returned_sections))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("mode").agg(
        mines=("mine_id", "count"), valid_report=("valid_report", "mean"), facts_correct=("facts_correct", "mean"),
        invented_citations=("invented_citations", "sum"), tool_calls=("tool_calls", "mean"),
        input_tokens=("input_tokens", "mean"), output_tokens=("output_tokens", "mean"),
        seconds=("seconds", "mean")).round(3)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--regs-dir")
    p.add_argument("--mines", type=int, default=20)
    p.add_argument("--out", default="results")
    args = p.parse_args(argv)

    from .gemini import GeminiAgentLLM
    from .tools import MCPTools

    mine_ids = pick_mines(MineData(args.data_dir, args.regs_dir), args.mines)
    df = asyncio.run(compare(args.data_dir, args.regs_dir, mine_ids, GeminiAgentLLM(),
                             lambda: MCPTools(args.data_dir, args.regs_dir)))
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "per_mine.csv"), index=False)
    table = summarize(df)
    open(os.path.join(args.out, "summary.md"), "w").write(table.to_markdown())
    print(table.to_string())


if __name__ == "__main__":
    main()
