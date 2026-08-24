"""Capability reporting for canonical graph IR."""

from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass

from .graph import GraphIR, IRNode


@dataclass(frozen=True)
class CapabilityReport:
    """Operator counts and nodes unsupported by a consumer."""

    operator_counts: dict[str, int]
    unsupported_nodes: tuple[IRNode, ...]

    @property
    def supported(self) -> bool:
        return not self.unsupported_nodes


def analyze_capabilities(
    graph: GraphIR,
    supported_ops: Collection[str],
) -> CapabilityReport:
    """Compare graph operators with a backend or pass capability set."""
    counts = Counter(node.op_type for node in graph.nodes)
    unsupported = tuple(
        node for node in graph.nodes if node.op_type not in supported_ops
    )
    return CapabilityReport(
        operator_counts=dict(counts),
        unsupported_nodes=unsupported,
    )
