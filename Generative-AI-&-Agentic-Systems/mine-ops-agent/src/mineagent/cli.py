"""Generate one mine safety report with the agent.

    export GEMINI_API_KEY=...
    python -m mineagent.cli 0100003 --data-dir ../mine-safety-lakehouse/lakehouse_data \
        --regs-dir ../regs-to-vectors/vectors
"""
import argparse
import asyncio
import json

from .agent import run_agent, run_workflow


async def _run(args):
    from .gemini import GeminiAgentLLM
    from .tools import MCPTools

    async with MCPTools(args.data_dir, args.regs_dir) as tools:
        runner = run_workflow if args.workflow else run_agent
        res = await runner(args.mine_id, GeminiAgentLLM(), tools)
    print(f"Tool calls: {[c.name for c in res.tool_calls]}")
    if res.report is None:
        print(f"No valid report: {res.error}")
    else:
        print(json.dumps(res.report.model_dump(), indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mine_id")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--regs-dir")
    p.add_argument("--workflow", action="store_true", help="Use the fixed workflow instead of the agent")
    asyncio.run(_run(p.parse_args(argv)))


if __name__ == "__main__":
    main()
