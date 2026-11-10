# Building the workflow version in Dify (Lab 4 style)

The Python workflow in `src/mineagent/agent.py` (`run_workflow`) is the reference. To rebuild it in Dify:

1. **Expose the tools over HTTP.** Start the MCP server in HTTP mode:
   ```bash
   DATA_DIR=... REGS_DIR=... python -m mineagent.server --http --port 8000
   ```
   If Dify runs in Docker on the same machine, it reaches your computer at `http://host.docker.internal:8000/mcp`.
2. **Add the MCP server to Dify.** In Dify, add an MCP server under Tools and enter that URL. The six tools should appear. (Menu names vary between Dify versions; MCP support needs a recent version.)
3. **Create a Workflow app** with one short text input: `mine_id`.
4. **Add tool nodes in a fixed order**, each taking `mine_id` from the input:
   `get_mine_profile` → `get_violation_history` (quarters = 8) → `get_accident_summary` (years = 5).
5. **Add `search_regulations`**, with a query built from the accident summary (a Template node can pull out the first accident type, as you did with JSON in Lab 4).
6. **Add an LLM node** (Gemini 2.5 Flash). Put the text of `REPORT_INSTRUCTIONS` from `src/mineagent/report.py` in the system prompt, and the four tool outputs in the user prompt.
7. **Add an End node** returning the LLM text.
8. **Export the app** as YAML (DSL) and commit it to `dify/MineSafetyWorkflow.yml`.

Run it on a few of the same mines the Python comparison used, and note any differences in the README.
