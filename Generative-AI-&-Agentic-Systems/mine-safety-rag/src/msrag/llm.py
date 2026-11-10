"""LLM clients. Gemini for real runs; a scripted fake for tests."""
import os
import time
from dataclasses import dataclass


@dataclass
class LLMResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0


class GeminiLLM:
    """Google Gemini via the google-genai SDK. Set GEMINI_API_KEY."""

    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.0):
        from google import genai
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model, self.temperature = model, temperature
        self.name = model

    def generate(self, prompt: str, system: str | None = None) -> LLMResult:
        from google.genai import types
        start = time.perf_counter()
        resp = self.client.models.generate_content(
            model=self.model, contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system, temperature=self.temperature))
        usage = getattr(resp, "usage_metadata", None)
        return LLMResult(resp.text or "",
                         getattr(usage, "prompt_token_count", 0) or 0,
                         getattr(usage, "candidates_token_count", 0) or 0,
                         time.perf_counter() - start)


class FakeLLM:
    """Returns answers from a function of the prompt. For tests only."""
    name = "fake"

    def __init__(self, respond):
        self.respond = respond
        self.calls = []

    def generate(self, prompt: str, system: str | None = None) -> LLMResult:
        self.calls.append((system, prompt))
        return LLMResult(self.respond(prompt, system))
