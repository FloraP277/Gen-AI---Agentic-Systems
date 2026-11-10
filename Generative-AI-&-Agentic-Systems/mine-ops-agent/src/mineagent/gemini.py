"""Gemini tool calling, adapted to the agent's Step interface. Set GEMINI_API_KEY."""
import json
import os

from .agent import Step, ToolCall


class GeminiAgentLLM:
    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.0):
        from google import genai
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model, self.temperature = model, temperature

    def _contents(self, history):
        from google.genai import types
        out = []
        for h in history:
            if h["role"] == "user":
                out.append(types.Content(role="user", parts=[types.Part.from_text(text=h["text"])]))
            elif h["role"] == "model" and "tool_calls" in h:
                out.append(types.Content(role="model", parts=[
                    types.Part.from_function_call(name=c.name, args=c.args) for c in h["tool_calls"]]))
            elif h["role"] == "model":
                out.append(types.Content(role="model", parts=[types.Part.from_text(text=h["text"])]))
            elif h["role"] == "tool":
                out.append(types.Content(role="user", parts=[
                    types.Part.from_function_response(name=r["name"], response={"result": r["result"]})
                    for r in h["results"]]))
        return out

    def step(self, system: str, history: list, tool_specs: list) -> Step:
        from google.genai import types
        tools = None
        if tool_specs:
            tools = [types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t["name"], description=t["description"] or "",
                                          parameters_json_schema=t["input_schema"])
                for t in tool_specs])]
        cfg = types.GenerateContentConfig(
            system_instruction=system, temperature=self.temperature, tools=tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        r = self.client.models.generate_content(model=self.model, contents=self._contents(history), config=cfg)
        u = r.usage_metadata
        calls = [ToolCall(fc.name, dict(fc.args or {})) for fc in (r.function_calls or [])]
        return Step(calls, "" if calls else (r.text or ""),
                    u.prompt_token_count or 0, u.candidates_token_count or 0)
