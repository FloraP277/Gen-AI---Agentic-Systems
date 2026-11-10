"""Ways to call the tools: through the MCP server (real), or directly (tests)."""
import json
import os
import sys
from contextlib import AsyncExitStack


class LocalTools:
    """Calls the same functions the MCP server wraps, without a server process."""

    def __init__(self, data):
        from . import server
        server._data = data
        self._fns = {n: getattr(server, n) for n in
                     ["find_mines", "get_mine_profile", "get_violation_history", "get_accident_summary",
                      "search_regulations", "plot_trend"]}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def list_tools(self) -> list[dict]:
        from . import server
        tools = await server.mcp.list_tools()
        return [{"name": t.name, "description": t.description, "input_schema": t.inputSchema} for t in tools]

    async def call(self, name: str, args: dict) -> str:
        if name not in self._fns:
            return json.dumps({"error": f"unknown tool {name}"})
        try:
            return self._fns[name](**args)
        except TypeError as e:  # wrong or missing arguments from the model
            return json.dumps({"error": f"bad arguments: {e}"})


class MCPTools:
    """Starts the MCP server as a subprocess and talks to it over stdio."""

    def __init__(self, data_dir: str, regs_dir: str | None = None):
        self.env = {**os.environ, "DATA_DIR": data_dir, **({"REGS_DIR": regs_dir} if regs_dir else {})}
        self.stack = AsyncExitStack()
        self.session = None

    async def __aenter__(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        src = os.path.join(os.path.dirname(__file__), "..")
        params = StdioServerParameters(command=sys.executable, args=["-m", "mineagent.server"],
                                       env={**self.env, "PYTHONPATH": os.path.abspath(src)})
        read, write = await self.stack.enter_async_context(stdio_client(params))
        self.session = await self.stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        return self

    async def __aexit__(self, *exc):
        await self.stack.aclose()

    async def list_tools(self) -> list[dict]:
        res = await self.session.list_tools()
        return [{"name": t.name, "description": t.description, "input_schema": t.inputSchema} for t in res.tools]

    async def call(self, name: str, args: dict) -> str:
        res = await self.session.call_tool(name, args)
        if res.isError:
            return json.dumps({"error": " ".join(c.text for c in res.content if hasattr(c, "text"))})
        return "".join(c.text for c in res.content if hasattr(c, "text"))
