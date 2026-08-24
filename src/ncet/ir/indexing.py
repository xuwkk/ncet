"""Conversion of canonical static indices into Python indexing objects."""

from ..errors import UnsupportedOperatorError
from .graph import IRNode


def static_index(node: IRNode) -> tuple[object, ...]:
    """Return the Python index represented by a GetItem or Slice IR node."""
    index: list[object] = []
    for item in node.attrs["index"]:
        kind = item[0]
        if kind == "index":
            index.append(item[1])
        elif kind == "slice":
            index.append(slice(*item[1:]))
        elif kind == "newaxis":
            index.append(None)
        else:
            raise UnsupportedOperatorError(
                f"invalid canonical index kind {kind!r} at node '{node.name}'"
            )
    return tuple(index)
