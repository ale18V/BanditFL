"""Topology module for network communication graphs and gossip matrices."""

from .graph import CommunicationNetwork, create_graph
from .fxgraph import generate_connected_graph, graph_byz_robust
from .gossip import LaplacianGossipMatrix

__all__ = [
    "CommunicationNetwork",
    "create_graph",
    "generate_connected_graph",
    "graph_byz_robust",
    "LaplacianGossipMatrix",
]
