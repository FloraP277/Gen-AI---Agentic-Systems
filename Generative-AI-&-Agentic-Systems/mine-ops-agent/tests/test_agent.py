import asyncio
import json
import os

import pytest

from mineagent.agent import Step, ToolCall, run_agent, run_workflow
from mineagent.compare import compare, summarize, truth
from mineagent.data import ToolInputError
from mineagent.tools import LocalTools, MCPTools


def test_tools_return_correct_facts(mine_data):
    p = mine_data.get_mine_profile("200001")              # leading zero restored
    assert p["mine_id"] == "0200001" and p["employees"] == 210
    v = mine_data.get_violation_history("0200001", 8)
    assert len(v["by_quarter"]) == 8 and v["by_quarter"][-1]["quarter"] == "2026-Q2"
    assert v["total_violations"] == sum(range(6, 14))
    a = mine_data.get_accident_summary("0200001", 5)
    assert a["accidents"] == 3 and a["serious"] == 1 and a["days_lost"] == 15   # 2019 is outside 5 years
    assert list(a["top_accident_types"])[0] == "Struck by"
    with pytest.raises(ToolInputError):
        mine_data.get_mine_profile("ABC")
    with pytest.raises(ToolInputError):
        mine_data.get_violation_history("0200001", 99)
    assert mine_data.search_regulations("scoop struck by")[0]["section"] == "75.1725"
    assert mine_data.search_regulations("seat belts", part="75") == []


def test_server_tools_turn_bad_input_into_errors(mine_data, tmp_path, monkeypatch):
    tools = LocalTools(mine_data)
    out = json.loads(asyncio.run(tools.call("get_mine_profile", {"mine_id": "9999999"})))
    assert "error" in out
    out = json.loads(asyncio.run(tools.call("get_violation_history", {"wrong_arg": 1})))
    assert "bad arguments" in out["error"]
    monkeypatch.setenv("PLOT_DIR", str(tmp_path))
    out = json.loads(asyncio.run(tools.call("plot_trend", {"mine_id": "0100003", "metric": "violations"})))
    assert os.path.exists(out["image_path"])


class Scripted:
    """A fake model: calls the planned tools, then writes a report from the results it saw."""

    def __init__(self, plan, wrong_numbers=False, bad_json_first=False):
        self.plan, self.wrong, self.bad_first = list(plan), wrong_numbers, bad_json_first
        self.finals = 0

    def step(self, system, history, specs):
        if self.plan:
            return Step(tool_calls=[self.plan.pop(0)], input_tokens=100, output_tokens=10)
        self.finals += 1
        if self.bad_first and self.finals == 1:
            return Step(text="Here is the report: risk is high.")
        facts = {}
        for h in history:
            for r in h.get("results", []):
                facts.setdefault(r["name"], json.loads(r["result"]))
            if h["role"] == "user" and "Facts gathered" in h["text"]:
                facts = json.loads(h["text"].split("\n", 1)[1].rsplit("\n\nWrite", 1)[0])
        v, a = facts["get_violation_history"], facts["get_accident_summary"]
        regs = [s["section"] for s in facts.get("search_regulations", [])]
        report = {"mine_id": v["mine_id"], "mine_name": facts["get_mine_profile"]["name"],
                  "total_violations": v["total_violations"] + (1 if self.wrong else 0),
                  "total_ss_violations": v["total_ss_violations"], "accidents": a["accidents"],
                  "injuries": a["injuries"], "risk_level": "high", "summary": "s", "violation_trend": "rising",
                  "accident_patterns": "struck by scoops", "relevant_regulations": regs + ["99.999"],
                  "recommendations": ["r"]}
        return Step(text=json.dumps(report), input_tokens=500, output_tokens=200)


AGENT_PLAN = [ToolCall("get_mine_profile", {"mine_id": "0200001"}),
              ToolCall("get_violation_history", {"mine_id": "0200001", "quarters": 8}),
              ToolCall("get_accident_summary", {"mine_id": "0200001", "years": 5}),
              ToolCall("search_regulations", {"query": "Struck by scoop"})]


def test_agent_produces_valid_report_and_tracks_citations(mine_data):
    res = asyncio.run(run_agent("0200001", Scripted(AGENT_PLAN), LocalTools(mine_data)))
    assert res.report is not None and res.report.total_violations == truth(mine_data, "0200001")["total_violations"]
    assert [c.name for c in res.tool_calls] == [c.name for c in AGENT_PLAN]
    assert "75.1725" in res.returned_sections and "99.999" not in res.returned_sections


def test_agent_guardrails(mine_data):
    repeat = [ToolCall("get_mine_profile", {"mine_id": "0200001"})] * 2 + AGENT_PLAN[1:]
    res = asyncio.run(run_agent("0200001", Scripted(repeat), LocalTools(mine_data)))
    assert len(res.tool_calls) == 4                       # the repeated call was blocked
    blocked = res.transcript[4]["results"][0]["result"]
    assert "already called" in blocked

    looping = [ToolCall("find_mines", {"limit": i + 1}) for i in range(20)]
    res = asyncio.run(run_agent("0200001", Scripted(looping), LocalTools(mine_data), max_tool_calls=5))
    assert res.report is None and res.error == "tool budget exceeded"

    res = asyncio.run(run_agent("0200001", Scripted(AGENT_PLAN, bad_json_first=True), LocalTools(mine_data)))
    assert res.report is not None                         # recovered after one retry


def test_workflow_and_comparison(dirs, mine_data):
    res = asyncio.run(run_workflow("0200001", Scripted([]), LocalTools(mine_data)))
    assert res.report is not None and len(res.tool_calls) == 4
    assert res.returned_sections == {"75.1725"}

    class Factory:
        def __init__(self, s): self.s = s
        async def __aenter__(self): return self.s
        async def __aexit__(self, *e): return None

    class Fresh:
        """New scripted model per run, so each run gets its own tool plan."""
        def step(self, system, history, specs):
            if not hasattr(self, "_cur") or history[-1].get("text", "").startswith(("Write the safety", "Facts")):
                mine = history[0]["text"].split()[-1].strip(".") if "Write the safety" in history[0]["text"] else None
                plan = [ToolCall(c.name, {**c.args, **({"mine_id": mine} if "mine_id" in c.args else {})})
                        for c in AGENT_PLAN] if mine else []
                self._cur = Scripted(plan, wrong_numbers=(mine == "0100003"))
            return self._cur.step(system, history, specs)

    df = asyncio.run(compare(*dirs, ["0200001", "0100003"], Fresh(), lambda: Factory(LocalTools(mine_data))))
    table = summarize(df)
    assert set(table.index) == {"agent", "workflow"}
    agent = df[df["mode"] == "agent"].set_index("mine_id")
    assert agent.loc["0200001", "facts_correct"] == 1.0
    assert agent.loc["0100003", "facts_correct"] == 0.75   # one number copied wrong
    assert (df["invented_citations"] == 1).all()            # the fake always adds 99.999


def test_mcp_server_over_stdio(dirs):
    async def go():
        async with MCPTools(*dirs) as tools:
            names = {t["name"] for t in await tools.list_tools()}
            profile = json.loads(await tools.call("get_mine_profile", {"mine_id": "0100003"}))
            return names, profile

    names, profile = asyncio.run(go())
    assert names == {"find_mines", "get_mine_profile", "get_violation_history", "get_accident_summary",
                     "search_regulations", "plot_trend"}
    assert profile["name"] == "Red Hill Quarry"
