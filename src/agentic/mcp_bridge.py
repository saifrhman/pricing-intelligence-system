"""Optional bridge for attaching trusted external MCP servers to OpenAI agents."""
from __future__ import annotations
import os
from typing import List

def external_mcp_servers_from_env() -> List[object]:
    """Build Streamable HTTP MCP server definitions from PRICING_EXTERNAL_MCP_URLS.

    URLs are comma-separated. This function creates definitions only; callers are
    responsible for entering each server's async context before passing it to an Agent.
    External MCP is deliberately opt-in and should only point to trusted servers.
    """
    raw = os.getenv("PRICING_EXTERNAL_MCP_URLS", "").strip()
    if not raw:
        return []
    try:
        from agents.mcp import MCPServerStreamableHttp
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("External MCP requires openai-agents") from exc
    servers = []
    for idx, url in enumerate(x.strip() for x in raw.split(",") if x.strip()):
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Unsupported MCP URL: {url}")
        servers.append(
            MCPServerStreamableHttp(
                name=f"external-mcp-{idx+1}",
                params={"url": url, "timeout": 10},
                cache_tools_list=True,
                max_retry_attempts=2,
                require_approval="always",
            )
        )
    return servers
