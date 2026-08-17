"""FastMCP server exposing hplot sub-commands as MCP tools.

Tools are registered from the hand-written command table in
:mod:`hplot.mcp.schema` (hplot has no ``describe`` command, so the table
is the single source of truth). Long-running commands (``test``,
``screen``, ``loci``) return a ``job_id`` immediately and the agent
polls ``job_status`` / ``job_logs`` / ``cancel_job``. Short commands
(``plot``, ``gam``) run synchronously and return the subprocess exit
code plus a tail of its stdout/stderr.

Each sub-command is exposed with the exact argparse parameter names and
types from :mod:`hplot.mcp.schema`, so the MCP tool's input schema is a
faithful per-parameter mirror of the CLI's ``--help`` rather than a
generic ``args: dict`` blob.

Usage::

    from hplot.mcp.server import build_server

    mcp = build_server(max_concurrent=2)
    mcp.run()                      # stdio
    mcp.run(transport="http",      # streamable HTTP
            host="127.0.0.1", port=8767)
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
import time
from typing import Annotated, Any

try:
    from fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "hplot.mcp requires the 'fastmcp' package. "
        "Install it with: pip install 'hplot[mcp]'"
    ) from exc

from pydantic import Field

from hplot.mcp.adapters import args_to_argv
from hplot.mcp.jobs import JobManager
from hplot.mcp.schema import (
    COMMANDS,  # noqa: F401 - re-exported for tests
    discover_commands,
    is_long_running,
)


# -- schema kind -> Python type mapping ------------------------------------

_KIND_TO_PY: dict = {
    "string": str,
    "int": int,
    "integer": int,
    "float": float,
    "number": float,
    "bool": bool,
    "boolean": bool,
    "path": str,
    "choice": str,
}


def _build_signature(command: dict):
    """Return ``(signature, annotations_dict)`` for one CLI command."""
    parameters = []
    annotations: dict = {}
    for p in command.get("params", []):
        pname = p["name"]
        kind = str(p.get("kind", "string")).lower()
        py_type = _KIND_TO_PY.get(kind, str)
        if p.get("multiple"):
            py_type = list
        help_text = " ".join(str(p.get("help", "")).split())

        if p.get("required"):
            annotation = (
                Annotated[py_type, Field(description=help_text)]
                if help_text
                else py_type
            )
            default = inspect.Parameter.empty
        else:
            # Allow None as the absence sentinel so the adapter can drop it.
            annotation = (
                Annotated[Any, Field(description=help_text)]
                if help_text
                else Any
            )
            default = p.get("default", None)
        parameters.append(
            inspect.Parameter(
                name=pname,
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=annotation,
            )
        )
        annotations[pname] = annotation
    return inspect.Signature(parameters), annotations


# -- subprocess runners ----------------------------------------------------


def _run_sync(argv_tail: list, timeout_s: float = 600.0) -> dict:
    """Run ``python -m hplot <argv_tail>`` synchronously and capture output."""
    started = time.time()
    try:
        proc = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "hplot"] + argv_tail,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "argv": argv_tail,
            "duration_s": round(time.time() - started, 3),
            "error": f"command exceeded {timeout_s}s synchronous timeout",
        }
    out = (proc.stdout or "") + (proc.stderr or "")
    tail = out.splitlines()[-50:]
    return {
        "status": "done" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "argv": argv_tail,
        "duration_s": round(time.time() - started, 3),
        "log_tail": tail,
    }


def _make_long_tool(jobs: JobManager, name: str, command: dict):
    """Build a per-command tool function whose signature mirrors the CLI."""

    sig, ann = _build_signature(command)

    def _impl(**kwargs) -> dict:
        cleaned = {k: v for k, v in kwargs.items() if v is not None}
        argv_tail = args_to_argv(command, cleaned)
        job_id = jobs.submit(name, argv_tail)
        return {
            "job_id": job_id,
            "status": "started",
            "argv": argv_tail,
            "hint": (
                f"Poll job_status(job_id={job_id!r}) and "
                f"job_logs(job_id={job_id!r}). "
                f"Cancel with cancel_job(job_id={job_id!r})."
            ),
        }

    _impl.__signature__ = sig  # type: ignore[attr-defined]
    _impl.__annotations__ = dict(ann)
    _impl.__name__ = name.replace("-", "_")
    return _impl


def _make_short_tool(name: str, command: dict):
    sig, ann = _build_signature(command)

    def _impl(**kwargs) -> dict:
        cleaned = {k: v for k, v in kwargs.items() if v is not None}
        argv_tail = args_to_argv(command, cleaned)
        return _run_sync(argv_tail)

    _impl.__signature__ = sig  # type: ignore[attr-defined]
    _impl.__annotations__ = dict(ann)
    _impl.__name__ = name.replace("-", "_")
    return _impl


# -- builder ---------------------------------------------------------------


def build_server(
    *,
    max_concurrent: int | None = None,
    server_name: str = "hplot",
) -> "FastMCP":
    """Build and return a configured (but not-yet-running) :class:`FastMCP` server."""
    mcp = FastMCP(server_name)
    jobs = JobManager(max_concurrent=max_concurrent)

    # 1. Per-subcommand tools.
    for name, cmd in discover_commands().items():
        long_running = is_long_running(name)
        fn = _make_long_tool(jobs, name, cmd) if long_running else _make_short_tool(name, cmd)
        help_text = " ".join(str(cmd.get("help", "")).split())
        if long_running:
            description = (
                help_text
                + "\n\n[long-running] Returns a job_id; poll job_status / job_logs "
                "and stop early with cancel_job."
            )
        else:
            description = help_text
        mcp.tool(name=name.replace("-", "_"), description=description)(fn)

    # 2. Job-management meta-tools.
    @mcp.tool(
        name="job_status",
        description=(
            "Return a snapshot of one job (status, pid, duration, "
            "returncode, total log lines). Use job_logs to read output."
        ),
    )
    def job_status(job_id: str):
        return jobs.status(job_id)

    @mcp.tool(
        name="job_logs",
        description=(
            "Return the next chunk of stdout/stderr lines for a job. Pass "
            "since_line from a previous response's next_line to paginate."
        ),
    )
    def job_logs(job_id: str, since_line: int = 0, max_lines: int = 500):
        return jobs.logs(job_id, since_line=since_line, max_lines=max_lines)

    @mcp.tool(
        name="cancel_job",
        description=(
            "Request graceful cancellation (SIGINT) of a running job. Calling "
            "cancel_job a second time on the same job escalates to SIGTERM."
        ),
    )
    def cancel_job(job_id: str):
        return jobs.cancel(job_id)

    @mcp.tool(
        name="list_jobs",
        description="List all jobs (running and completed) known to this server.",
    )
    def list_jobs() -> list:
        return jobs.list()

    # 3. Resources.
    @mcp.resource(
        "hplot://schema",
        name="cli_schema",
        description=(
            "The full hplot CLI command table (single source of truth for "
            "this server's tools)."
        ),
        mime_type="application/json",
    )
    def schema_resource() -> str:
        return json.dumps(COMMANDS, indent=2)

    # 4. Prompt.
    @mcp.prompt(
        name="hplot_workflow",
        description=(
            "Walk through the canonical hplot analysis: H-Plot curves, "
            "per-layer test, GAM effect size, and the multi-feature screen."
        ),
    )
    def hplot_workflow() -> str:
        return (
            "You are an analyst running the hplot pipeline on a cohort CSV. "
            "Use the tools exposed by this MCP server to:\n"
            "1. Call `plot` (synchronous) to draw H-Plot curves per group "
            "from the input CSV.\n"
            "2. Call `test` (long-running) with --permutations > 0 for the "
            "per-layer test + cluster-mass permutation; poll job_status / "
            "job_logs until done.\n"
            "3. Call `gam` (synchronous) with --at-layer at the boundary "
            "layer of interest for the Stage-2 GAM effect size.\n"
            "4. Call `screen` (long-running) on a long CSV "
            "(sample x layer x unit x value) for the multi-feature "
            "border-gradient ranking; poll until done.\n"
            "5. Call `loci` (long-running) with the ranking CSV to render "
            "the H-Loci Summary panel.\n"
            "Use job_logs to surface progress to the user; cancel with "
            "cancel_job if asked."
        )

    return mcp


__all__ = ["build_server"]
