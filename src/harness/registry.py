"""Typed tool registry and permission boundary."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable

@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    handler: Callable[..., Any]
    allowed_agents: frozenset[str]
    read_only: bool = True

class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str, agent_name: str) -> ToolSpec:
        try:
            spec = self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc
        if agent_name not in spec.allowed_agents and "*" not in spec.allowed_agents:
            raise PermissionError(f"Agent '{agent_name}' is not allowed to call '{name}'")
        return spec

    def names_for(self, agent_name: str) -> Iterable[str]:
        return sorted(
            name
            for name, spec in self._tools.items()
            if agent_name in spec.allowed_agents or "*" in spec.allowed_agents
        )
