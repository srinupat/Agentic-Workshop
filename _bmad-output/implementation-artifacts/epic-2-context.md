# Epic 2 Context: the triage agent

<!-- Compiled from planning artifacts. Edit freely. Regenerate with compile-epic-context if planning docs change. -->

## Goal

Give the workshop a working triage agent: a LangChain `create_agent`-based agent that reads a support ticket through MCP tools, applies `TRIAGE_POLICY.md`, and returns a validated Epic 1-schema decision — with a human-in-the-loop pause before escalation. Epic 1 shipped the schema and data foundation; Epic 3 will evaluate this agent's decisions. Nothing in Epic 2 measures quality — it just makes the agent exist and run end to end via `uv run python run_agent.py <ticket_id>`.

No traditional planning artifacts exist (no PRD, architecture, or UX docs). The full contract lives in `_bmad-output/specs/spec-epic-2/SPEC.md` plus its companions `TRIAGE_POLICY.md` and `mcp/triage_server.py`; this file distills what a story-level developer needs.

## Stories

- Story 1: The triage agent (CAP-1..4, CAP-6 — provider switch, MCP-tool lookup, policy-driven structured output with retry-once, injection resistance)
- Story 2: Human-gated escalation (CAP-5 — `escalate_to_human` tool + LangChain human-in-the-loop middleware)

## Requirements & Constraints

- The agent is built with LangChain's `create_agent`; a hand-rolled tool loop is not acceptable.
- Model provider swaps by environment variable only: default `ChatGoogleGenerativeAI` (`MODEL` default `gemini-3.8-flash`, `GEMINI_API_KEY`); `PROVIDER=groq` selects `ChatGroq` (`MODEL` default `openai/gpt-oss-120b`, `GROQ_API_KEY`).
- MCP tools come only from `mcp/triage_server.py` over stdio via `langchain-mcp-adapters` — no other tool server.
- The agent must call `get_ticket` before `get_customer_history`, passing the `customer_id` returned by the first call. MLflow traces must show that order.
- The returned decision must validate against the Epic 1 schema (`schema.TriageDecision`). If validation fails, the agent retries once; a second failure ends the run with a clear error.
- Ticket text is untrusted customer input. Any instructions inside a ticket (e.g. "mark this P1") must be ignored — the agent decides on the ticket's actual content and the policy alone.
- `TRIAGE_POLICY.md`'s Enterprise-bump rule: if a customer is on the Enterprise plan with ≥3 open tickets, move priority up one level (P1 stays P1).
- Escalation (Story 2): P1 + Enterprise fires the `escalate_to_human` tool, which pauses the run for a terminal yes/no via LangChain's human-in-the-loop middleware. `escalate_to_human` is a local tool — it cannot live in `mcp/triage_server.py` (that file is read-only).
- Read-only, unchanged: `schema.py`, `load_seed.py`, `mcp/triage_server.py`, `TRIAGE_POLICY.md`, everything under `seed/`.
- `run_agent.py`'s existing MLflow setup — tracking URI `sqlite:///mlflow.db`, experiment `triage-agent`, `mlflow.langchain.autolog()` — is protected from change.
- No API keys or secrets in code; keys come from `.env` via `python-dotenv` (`load_dotenv()` already lives in `run_agent.py`).

## Technical Decisions

- Python 3.12+, packages managed with `uv`; never pip. Existing deps that matter for this epic: `langchain>=1.0`, `langchain-google-genai>=2.1`, `langchain-groq>=0.3`, `langchain-mcp-adapters>=0.1`, `mcp>=1.9`, `mlflow>=3.4`, `pydantic>=2.8`, `python-dotenv>=1.0`.
- The agent module is imported as `from agent import triage` and awaited with `asyncio.run(triage(ticket_id))` — so `triage` is `async` and returns a JSON-serializable dict.
- Structured output uses LangChain's `create_agent(..., response_format=TriageDecision)` (or equivalent), keyed off the Epic 1 pydantic model.
- The MCP server is launched over stdio: spawn `mcp/triage_server.py` as a subprocess through `langchain-mcp-adapters`; do not re-implement its tool logic.
- The system prompt is `TRIAGE_POLICY.md` read from disk at construction time so the agent's instructions stay in sync with the file.
- MLflow autolog captures LangChain runs; every `uv run python run_agent.py` invocation should appear as a trace at `sqlite:///mlflow.db`, experiment `triage-agent`.

## Cross-Story Dependencies

- Story 1 must land before Story 2: Story 2 (`escalate_to_human` + HITL) attaches a new tool + middleware to the agent Story 1 builds.
- Both stories consume Epic 1's `schema.TriageDecision` and `app.db` (loaded via `load_seed.py`).
- Epic 3 will use `run_agent.py`'s stable behavior + MLflow traces as the eval substrate; the agent module's `async triage(ticket_id) -> dict` contract is the seam.
