#    Copyright 2023 Alexander Koziell-Pipe

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
"""Vertex, Hyperedge and Hypergraph classes."""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Vertex:
    """A hypergraph vertex.

    Vertices may be typed. This still permits untyped hypergraphs, which are
    equivalent to typed hypergraphs where all vertices have the same type
    (eg `None`).
    """

    vtype: Any | None = None
    """A type associated with the vertex."""


@dataclass
class Hyperedge:
    """A hypergraph edge.

    This particular flavour of hyperedge has a total ordering on each of the
    source and target vertices, implemented by storing them as lists.
    """

    sources: List[int]
    """A list of integer identifiers for vertices into the hyperedge."""
    targets: List[int]
    """A list of integer identifiers for vertices from the hyperedge."""
    label: str | None = None
    """A label to identify the hyperedge when drawing."""


class Hypergraph:
    """A directed hypergraph with boundaries.

    This particular flavour of hypergraph has two totally ordered sets of
    priviledged vertices: the input and output vertices. Together, these
    are referred to as boundary vertices.

    The :py:class:`Hypergraph` instance begins with no vertices or hyperedges,
    which can then be added using :py:meth:`add_vertex` and
    :py:meth:`add_edge` to build out the hypergraph.
    """

    def __init__(self) -> None:
        """Initialize a `Hypergraph`."""
        self.vertices: Dict[int, Vertex] = {}
        self.edges: Dict[int, Hyperedge] = {}
        self.inputs: List[int] = []
        self.outputs: List[int] = []

    def add_vertex(self, vtype: Any | None = None) -> Vertex:
        """Add a vertex to the hypergraph.

        Args:
            vtype: A type associated with the vertex.
        """
        # Give the new vertex a unique identifier
        index = max((i for i in self.vertices.keys()), default=-1) + 1
        new_vertex = Vertex(vtype)
        self.vertices[index] = new_vertex
        return new_vertex

    def add_edge(self,
                 sources: List[int], targets: List[int],
                 label: str | None = None) -> Hyperedge:
        """Add a new hyperedge to the hypergraph.

        Args:
            sources: A list of integer identifiers for vertices
                    directed to the hyperedge.
            targets: A list of integer identifiers for vertices
                    directed from the hyperedge.
            label: A label to identify the hyperedge when drawing.
        """
        # Check all the vertex identifiers in sources and targets correspond to
        # existing vertices in the hypergraph.
        if not all(vertex in self.vertices
                   for boundary in (sources, targets)
                   for vertex in boundary):
            raise ValueError(
              'Hyperedge attached to vertices that are not in the hypergraph.'
            )

        # Give the new hyperedge a unique identifier
        index = max((i for i in self.edges.keys()), default=-1) + 1
        new_edge = Hyperedge(sources, targets, label)
        self.edges[index] = new_edge
        return new_edge
