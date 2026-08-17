"""hplot-mcp: FastMCP server exposing the ``hplot`` CLI as MCP tools.

The hplot package itself is a plain-argparse CLI with no ``describe``
command, so this server's tool surface is defined by the hand-written
command table in :mod:`hplot.mcp.schema` (one entry per sub-command:
``plot``, ``test``, ``gam``, ``screen``, ``loci``). Each sub-command is
exposed as a tool whose input schema is a faithful per-parameter mirror
of ``hplot <sub-command> --help``.

Long-running sub-commands (``test``, ``screen``, ``loci``) return a
``job_id`` immediately; the agent polls ``job_status`` / ``job_logs``
and can stop early with ``cancel_job``. Short sub-commands (``plot``,
``gam``) run synchronously and return the exit code plus a tail of the
output.

Requires the optional ``fastmcp`` dependency::

    pip install 'hplot[mcp]'

Run the server::

    hplot-mcp                     # stdio (default)
    hplot-mcp --http 127.0.0.1:8767   # streamable HTTP
"""

__all__ = ["schema", "adapters", "jobs", "server"]
