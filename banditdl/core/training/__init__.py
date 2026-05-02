"""Training module for Byzantine-resilient distributed learning."""

from .dynamic.worker import P2PWorker
from .fixed_graph.worker import P2PWorker as FixedGraphP2PWorker
from .byzantine import ByzantineWorker, DecByzantineWorker

__all__ = [
    "P2PWorker",
    "FixedGraphP2PWorker",
    "ByzantineWorker",
    "DecByzantineWorker",
]
