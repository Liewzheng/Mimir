"""Mimir: A plastic embedding system that remembers."""

from mimir.adapters.agents import (
    AgentMemoryInterface,
    InMemoryAgentAdapter,
    Memory,
    Message,
)
from mimir.application.factories import create_mimir
from mimir.core.config import MimirConfig
from mimir.core.mimir import Mimir

__all__ = [
    "Mimir",
    "create_mimir",
    "MimirConfig",
    "AgentMemoryInterface",
    "InMemoryAgentAdapter",
    "Message",
    "Memory",
]
__version__ = "0.2.0.dev2"
