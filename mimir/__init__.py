"""Mimir: A plastic embedding system that remembers."""

from mimir.adapters.agents import (
    AgentMemoryInterface,
    InMemoryAgentAdapter,
)
from mimir.application.factories import create_mimir
from mimir.core.config import MimirConfig
from mimir.core.mimir import Mimir
from mimir.domain.model import Memory, Message

__all__ = [
    "Mimir",
    "create_mimir",
    "MimirConfig",
    "AgentMemoryInterface",
    "InMemoryAgentAdapter",
    "Message",
    "Memory",
]
__version__ = "0.3.0"
