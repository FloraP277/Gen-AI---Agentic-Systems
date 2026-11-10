"""Map a model's free-text reply to one of the 77 intents."""
import difflib
import re


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")


def parse_prediction(text: str, intents: list, cutoff: float = 0.85) -> str | None:
    """Exact match after normalizing case and punctuation; otherwise the closest
    intent name if it is very similar (catches small typos); otherwise None,
    which is scored as wrong."""
    if not text:
        return None
    first = text.strip().splitlines()[0]
    lookup = {_norm(i): i for i in intents}
    key = _norm(first)
    if key in lookup:
        return lookup[key]
    close = difflib.get_close_matches(key, list(lookup), n=1, cutoff=cutoff)
    return lookup[close[0]] if close else None
