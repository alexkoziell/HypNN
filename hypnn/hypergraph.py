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

from __future__ import annotations
from copy import deepcopy
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

    def add_vertex(self, vtype: Any | None = None) -> int:
        """Add a vertex to the hypergraph.

        Args:
            vtype: A type associated with the vertex.

        Returns:
            vertex_id: A unique integer identifier for the hypergraph
                       to reference the newly created vertex.
        """
        # Give the new vertex a unique integer identifier
        vertex_id = max((i for i in self.vertices.keys()), default=-1) + 1
        new_vertex = Vertex(vtype)
        self.vertices[vertex_id] = new_vertex
        return vertex_id

    def add_edge(self,
                 sources: List[int], targets: List[int],
                 label: str | None = None) -> int:
        """Add a new hyperedge to the hypergraph.

        Args:
            sources: A list of integer identifiers for vertices
                    directed to the hyperedge.
            targets: A list of integer identifiers for vertices
                    directed from the hyperedge.
            label: A label to identify the hyperedge when drawing.

        Returns:
            edge_id: A unique integer identifier for the hypergraph
                     to reference the newly created edge.
        """
        # Check all the vertex identifiers in sources and targets correspond to
        # existing vertices in the hypergraph.
        if not all(vertex in self.vertices
                   for boundary in (sources, targets)
                   for vertex in boundary):
            raise ValueError(
              'Hyperedge attached to vertices that are not in the hypergraph.'
            )

        # Give the new hyperedge a unique integer identifier
        edge_id = max((i for i in self.edges.keys()), default=-1) + 1
        new_edge = Hyperedge(sources, targets, label)
        self.edges[edge_id] = new_edge
        return edge_id

    def parallel_comp(self, other: Hypergraph,
                      in_place: bool = False) -> Hypergraph | None:
        """Compose this hypergraph in parallel with `other`.

        Args:
            other: The hypergraph to compose with.
            in_place: Whether to modify `self` rather than creating
                      a new :py:class:`Hypergraph` instance.

        Returns:
            composed: The parallel composition of `self` with `other`.
        """
        # If in_place, modify self directly
        composed = self if in_place else deepcopy(self)

        # Add vertices to self for each vertex in other, and create a
        # one-to-one correspondence between the added vertices and
        # the vertices of other
        vertex_map: Dict[int, int] = {}
        for vertex_id, vertex in other.vertices.items():
            vertex_map[vertex_id] = composed.add_vertex(vertex.vtype)

        # Add edges to self for each edge in other, with connectivity that
        # matches that of other in a compatible way with the vertex map
        # established above
        for edge in other.edges.values():
            composed.add_edge([vertex_map[s] for s in edge.sources],
                              [vertex_map[t] for t in edge.targets],
                              edge.label)

        # For parallel composition, append the boundary vertices of other
        # to the boundary vertices of self.
        composed.inputs += [vertex_map[i] for i in other.inputs]
        composed.outputs += [vertex_map[o] for o in other.outputs]

        # To make it clear when an in place modifications has been made,
        # return None if in_place
        return None if in_place else composed

    def sequential_comp(self, other: Hypergraph,
                        in_place: bool = False) -> Hypergraph | None:
        """Compose this hypergraph in sequence with `other`.

        The outputs of `self` must match the inputs of `other` in order
        for the composition to be well-defined.

        Args:
            other: The hypergraph to compose with.
            in_place: Whether to modify `self` rather than creating
                      a new :py:class:`Hypergraph` instance.

        Returns:
            composed: The sequential composition of `self` with `other`.
        """
        if n_out := len(self.outputs) != (n_in := len(other.inputs)):
            raise ValueError(
                f'Cannot sequentially compose hypergraph with {n_out} outputs'
                + f'with one with {n_in} inputs.'
            )
        if any(self.vertices[o].vtype != other.vertices[i].vtype
               for o, i in zip(self.outputs, other.inputs)):
            raise ValueError(
                'Vertex types do not match at composition boundary.'
            )

        # If in_place, modify self directly
        composed = self if in_place else deepcopy(self)

        # Add vertices to self for each non-boundary vertex in other,
        # and create a one-to-one correspondence between the added vertices
        # and the vertices of other
        vertex_map: Dict[int, int] = {}
        for vertex_id, vertex in other.vertices.items():
            # Inputs of other must be mapped to outputs of self
            if vertex_id in other.inputs:
                pass
            vertex_map[vertex_id] = composed.add_vertex(vertex.vtype)

        # Identify output vertices of self with input vertices of other
        for out_vid, in_vid in zip(composed.outputs, other.inputs):
            vertex_map[in_vid] = out_vid

        # The outputs of the composition correspond to the outputs of other
        composed.outputs = [vertex_map[o] for o in other.outputs]

        # Add edges to self for each edge in other, with connectivity that
        # matches that of other in a compatible way with the vertex map
        # established above
        for edge in other.edges.values():
            composed.add_edge([vertex_map[s] for s in edge.sources],
                              [vertex_map[t] for t in edge.targets],
                              edge.label)

        # To make it clear when an in place modifications has been made,
        # return None if in_place
        return None if in_place else composed
