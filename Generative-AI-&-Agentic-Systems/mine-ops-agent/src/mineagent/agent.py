"""Two ways to produce the same report.

Agent (Lab 3): the LLM decides which tools to call, in what order, and when to stop.
Workflow (Lab 4): a fixed sequence of tool calls, then one LLM call to write the report.

Guardrails on the agent: a tool-call budget, a block on repeating the exact same
call, tool errors returned to the model so it can correct itself, and schema
validation of the final report with one retry.
"""
import json
import time
from dataclasses import dataclass, field

from .report import REPORT_INSTRUCTIONS, parse_report

AGENT_SYSTEM = """You are a mine safety analyst. Use the tools to gather facts about the mine you are
asked about: its profile, its violation history, its accident summary, and the regulations most
relevant to its main problems (search with the accident types or violation topics you found).
Call tools only when you need new information. When you have enough, reply with the final report.
""" + REPORT_INSTRUCTIONS


@dataclass
class ToolCall:
    name: str
    args: dict


@dataclass
class Step:
    """One model turn: either tool calls or final text."""
    tool_calls: list = field(default_factory=list)
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class RunResult:
    mode: str
    mine_id: str
    report: object
    error: str | None
    tool_calls: list
    input_tokens: int
    output_tokens: int
    seconds: float
    transcript: list
    returned_sections: set = field(default_factory=set)   # regulation sections the tools returned


def _sections_in(result_json: str) -> set:
    try:
        data = json.loads(result_json)
    except json.JSONDecodeError:
        return set()
    return {d["section"] for d in data if isinstance(d, dict) and "section" in d} if isinstance(data, list) else set()


async def run_agent(mine_id: str, llm, tools, max_tool_calls: int = 12) -> RunResult:
    start = time.perf_counter()
    specs = await tools.list_tools()
    history = [{"role": "user", "text": f"Write the safety report for mine {mine_id}."}]
    calls, seen, tin, tout = [], set(), 0, 0
    returned, attempts = set(), 0
    retried = False
    while True:
        step: Step = llm.step(AGENT_SYSTEM, history, specs)
        tin, tout = tin + step.input_tokens, tout + step.output_tokens
        if step.tool_calls:
            history.append({"role": "model", "tool_calls": step.tool_calls})
            results = []
            for c in step.tool_calls:
                attempts += 1
                key = (c.name, json.dumps(c.args, sort_keys=True))
                if len(calls) >= max_tool_calls:
                    out = json.dumps({"error": "tool budget used up; write the report now"})
                elif key in seen:
                    out = json.dumps({"error": "you already called this tool with these arguments; use the earlier result"})
                else:
                    seen.add(key)
                    calls.append(c)
                    out = await tools.call(c.name, c.args)
                    if c.name == "search_regulations":
                        returned |= _sections_in(out)
                results.append({"name": c.name, "result": out})
            history.append({"role": "tool", "results": results})
            if attempts >= max_tool_calls + 3:   # the model keeps calling tools despite the budget message
                return RunResult("agent", mine_id, None, "tool budget exceeded", calls, tin, tout,
                                 time.perf_counter() - start, history, returned)
            continue
        report, err = parse_report(step.text)
        history.append({"role": "model", "text": step.text})
        if report is None and not retried:
            retried = True
            history.append({"role": "user", "text": f"That report was invalid ({err}). Reply with only the corrected JSON."})
            continue
        return RunResult("agent", mine_id, report, err, calls, tin, tout, time.perf_counter() - start, history, returned)


WORKFLOW_PLAN = [
    ("get_mine_profile", lambda m: {"mine_id": m}),
    ("get_violation_history", lambda m: {"mine_id": m, "quarters": 8}),
    ("get_accident_summary", lambda m: {"mine_id": m, "years": 5}),
]


async def run_workflow(mine_id: str, llm, tools) -> RunResult:
    start = time.perf_counter()
    gathered, calls = {}, []
    for name, make_args in WORKFLOW_PLAN:
        args = make_args(mine_id)
        calls.append(ToolCall(name, args))
        gathered[name] = json.loads(await tools.call(name, args))
    # Regulations: search with the mine's most common accident type (fixed rule, no LLM choice).
    types = list(gathered["get_accident_summary"].get("top_accident_types", {}) or {})
    query = types[0] if types else "examination of working places"
    calls.append(ToolCall("search_regulations", {"query": query}))
    regs = await tools.call("search_regulations", {"query": query})
    gathered["search_regulations"] = json.loads(regs)

    prompt = f"Facts gathered for mine {mine_id}:\n{json.dumps(gathered, indent=1)}\n\nWrite the report."
    step = llm.step(REPORT_INSTRUCTIONS, [{"role": "user", "text": prompt}], [])
    report, err = parse_report(step.text)
    return RunResult("workflow", mine_id, report, err, calls, step.input_tokens, step.output_tokens,
                     time.perf_counter() - start, [{"role": "user", "text": prompt}, {"role": "model", "text": step.text}],
                     _sections_in(regs))
