"""LLM clients with every sampling setting pinned explicitly."""
import os
import time
from dataclasses import dataclass


@dataclass
class Settings:
    model: str
    temperature: float = 0.0
    top_p: float = 1.0
    max_output_tokens: int = 512
    stop_sequences: tuple = ()

    def key(self) -> str:
        return f"{self.model}|t={self.temperature}|p={self.top_p}|max={self.max_output_tokens}|stop={list(self.stop_sequences)}"


@dataclass
class Reply:
    text: str
    input_tokens: int
    output_tokens: int
    seconds: float


class GeminiLLM:
    """Set GEMINI_API_KEY. Retries on rate limits with exponential backoff."""

    def __init__(self, retries: int = 5):
        from google import genai
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.retries = retries

    def generate(self, system: str, user: str, s: Settings) -> Reply:
        from google.genai import types
        cfg = types.GenerateContentConfig(
            system_instruction=system, temperature=s.temperature, top_p=s.top_p,
            max_output_tokens=s.max_output_tokens, stop_sequences=list(s.stop_sequences) or None)
        for attempt in range(self.retries):
            try:
                start = time.perf_counter()
                r = self.client.models.generate_content(model=s.model, contents=user, config=cfg)
                u = r.usage_metadata
                return Reply(r.text or "", u.prompt_token_count or 0, u.candidates_token_count or 0,
                             time.perf_counter() - start)
            except Exception as e:  # rate limits and transient errors
                if attempt == self.retries - 1:
                    raise
                print(f"retrying after error: {e}")
                time.sleep(2 ** attempt)


class FakeLLM:
    """Scripted replies for tests: respond(system, user, settings) -> str."""

    def __init__(self, respond):
        self.respond = respond
        self.calls = 0

    def generate(self, system: str, user: str, s: Settings) -> Reply:
        self.calls += 1
        text = self.respond(system, user, s)
        return Reply(text, len(system.split()) + len(user.split()), len(text.split()), 0.01)
