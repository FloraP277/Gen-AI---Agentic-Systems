"""Versioned prompt templates loaded from prompts/*.yaml.

Each file pins one technique (zero-shot, few-shot, chain-of-thought, role).
Model settings are pinned in the experiment config, not here, so every result
can be traced to an exact prompt version plus exact settings.
"""
import glob
import hashlib
import os
from dataclasses import dataclass

import yaml


@dataclass
class PromptTemplate:
    name: str
    version: int
    technique: str
    system: str
    user: str
    few_shot_per_intent: int = 0

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(f"{self.system}\n{self.user}".encode()).hexdigest()[:10]

    def render(self, message: str, intents: list, queues: list, examples=None) -> tuple[str, str]:
        ex = ""
        if examples is not None and len(examples):
            ex = "\n".join(f'Message: "{r.text}"\nIntent: {r.intent}' for r in examples.itertuples())
        fields = {"intents": "\n".join(intents), "queues": ", ".join(queues), "examples": ex, "message": message}
        return self.system.format(**fields), self.user.format(**fields)


def load_prompts(folder: str) -> dict:
    out = {}
    for path in sorted(glob.glob(os.path.join(folder, "*.yaml"))):
        d = yaml.safe_load(open(path))
        out[d["name"]] = PromptTemplate(**d)
    return out
