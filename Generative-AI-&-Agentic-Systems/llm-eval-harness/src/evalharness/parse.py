"""Turn a model reply into a routing decision, and record what went wrong if it can't."""
import json
import re
from dataclasses import dataclass

JSON_OBJ = re.compile(r"\{.*\}", re.S)


@dataclass
class Parsed:
    intent: str | None
    queue: str | None
    needs_human: bool | None
    valid_json: bool
    valid_intent: bool


def _norm(s) -> str:
    return re.sub(r"[\s\-]+", "_", str(s).strip().lower())


def parse_reply(text: str, intents: list, queues: list) -> Parsed:
    m = JSON_OBJ.search(text or "")
    if not m:
        return Parsed(None, None, None, False, False)
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return Parsed(None, None, None, False, False)
    if not isinstance(obj, dict):
        return Parsed(None, None, None, False, False)
    lookup = {_norm(i): i for i in intents}
    intent = lookup.get(_norm(obj.get("intent", "")))
    queue = obj.get("queue")
    queue = queue if queue in queues else None
    nh = obj.get("needs_human")
    nh = nh if isinstance(nh, bool) else None
    return Parsed(intent, queue, nh, True, intent is not None)
