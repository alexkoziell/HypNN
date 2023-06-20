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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple


@dataclass
class Vertex:
    """A hypergraph vertex.

    Vertices may be typed. This still permits untyped hypergraphs, which are
    equivalent to typed hypergraphs where all vertices have the same type
    (eg `None`).
    """

    vtype: Any | None = None
    """A type associated with the vertex."""
    sources: Set[int] = field(default_factory=set)
    """A set of integer identifiers for hyperedges into the vertex."""
    targets: Set[int] = field(default_factory=set)
    """A set of integer identifiers for hyperedges from the hyperedge."""


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
    identity: bool = False
    """Whether this hyperedge is an identity or not."""


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
                 label: str | None = None,
                 identity: bool = False) -> int:
        """Add a new hyperedge to the hypergraph.

        Args:
            sources: A list of integer identifiers for vertices
                    directed to the hyperedge.
            targets: A list of integer identifiers for vertices
                    directed from the hyperedge.
            label: A label to identify the hyperedge when drawing.
            identity: Whether the new hyperedge is an identity or not.

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

        # The source and target types must be the same for identity hyperedges
        if identity:
            if (len(sources) != len(targets)
                or any(self.vertices[s].vtype != self.vertices[o].vtype
                       for s, o in zip(sources, targets))):
                raise ValueError(
                    'Source and target types of identity hyperedges must match'
                )

        # Give the new hyperedge a unique integer identifier
        edge_id = max((i for i in self.edges.keys()), default=-1) + 1
        new_edge = Hyperedge(sources, targets, label, identity)
        self.edges[edge_id] = new_edge

        # Register the edge as source or target of relevant vertices
        for s in sources:
            self.vertices[s].targets.add(edge_id)
        for t in targets:
            self.vertices[t].sources.add(edge_id)

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
                continue
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

    def insert_identity_after(self, vertex_id: int) -> int:
        """Add an identity hyperedge after a vertex in the hypergraph.

        This method adds a new vertex with source the new identity edge
        and targets those of the original vertex. The targets of the original
        vertex are replaced with just the new identity edge.

        Args:
            vertex_id: The identifier of the vertex after which to place
                       the identity.

        Returns:
            edge_id: The identifier of the newly created identity hyperedge.
        """
        # Create a new vertex with the same vtype as the original vertex
        old_vertex = self.vertices[vertex_id]
        new_vertex_id = self.add_vertex(old_vertex.vtype)
        new_vertex = self.vertices[new_vertex_id]

        # Add the new identity hyperedge
        edge_id = self.add_edge([vertex_id], [new_vertex_id],
                                label='id', identity=True)
        # Sources of the new vertex is just the new identity edge, and
        # targets of are those of the original vertex
        new_vertex.sources = {edge_id}
        new_vertex.targets = set(
            target_id for target_id in old_vertex.targets
            # add_edge method added identity edge as target of old vertex
            if target_id != edge_id
        )

        # New targets of the original vertex is just the new identity edge
        old_vertex.targets = {edge_id}
        # Update the sources lists of any targets of the new vertex to replace
        # the orignal vertex with the new vertex
        for target_id in new_vertex.targets:
            self.edges[target_id].sources = [
                source_id if source_id != vertex_id else new_vertex_id
                for source_id in self.edges[target_id].sources
            ]
        # If original vertex was an output, the new vertex now replaces it
        # in the list of outputs
        self.outputs = [output_id if output_id != vertex_id else new_vertex_id
                        for output_id in self.outputs]

        return edge_id

    def layer_decomp(self, in_place: bool = False
                     ) -> Tuple[Hypergraph, List[List[int]]]:
        """Decompose this hypergraph into layers of its hyperedges.

        Args:
            in_place: Whether to modify `self` rather than creating
                      a new :py:class:`Hypergraph` instance.

        Returns:
            tuple: a tuple containing multiple values
                - decomposed: The original hypergraph with possible additional
                  identity edges inserted during the decomposition.
                - edge_layers: A list of list of edge ids corresponding to
                  layers of the decomposition.
        """
        # If in_place, modify graph when identity edges are inserted
        decomposed = self if in_place else deepcopy(self)

        # This will become the final layer decomposition
        edge_layers: List[List[int]] = []
        # The target vertices of the edge layer
        # most recently added to the decomposition
        prev_vertex_layer: List[int] = []
        # The vertices already sitting between two edge
        # layers already added to the decomposition
        placed_vertices: Set[int] = set()
        # Edges ready to be placed into the current layer
        ready_edges: Set[int] = set()

        # Mark all input vertices as placed into the layer decomposition
        # if the input is also an output, split the vertex and insert an
        # identity edge between the two new vertices
        for input in decomposed.inputs:
            if input in decomposed.outputs:
                new_identity = decomposed.insert_identity_after(input)
                ready_edges.add(new_identity)
            prev_vertex_layer.append(input)
            placed_vertices.add(input)

        # Place the edges into layers
        # Edges not yet placed into any layer
        unplaced_edges: Set[int] = set(decomposed.edges.keys())
        # Track newly created identity edges from this point
        # to ensure new layers contain at least one non-identity edge
        new_identities: Set[int] = set()

        while len(unplaced_edges) > 0:
            # If all the source vertices of an edge have been placed,
            # it is ready to be added to the layer decomposition
            for edge_id in unplaced_edges:
                if all(vertex_id in placed_vertices
                        for vertex_id in decomposed.edges[edge_id].sources):
                    ready_edges.add(edge_id)
            # For the input vertices of the layer under construction,
            # If the input vertex is also an output or any of the
            # target edges are not ready, split the vertex and add
            # an identity to traverse the layer
            for vertex_id in prev_vertex_layer:
                if (
                    vertex_id in decomposed.outputs or
                    any(edge_id not in ready_edges
                        for edge_id in decomposed.vertices[vertex_id].targets)
                ):
                    new_identity = decomposed.insert_identity_after(vertex_id)
                    new_identities.add(new_identity)
                    ready_edges.add(new_identity)

            # Populate the current layer with edges that are ready to
            # be placed, and remove them from the set of unplaced edges
            current_layer: List[int] = []
            for edge_id in ready_edges:
                current_layer.append(edge_id)
                unplaced_edges.discard(edge_id)

            # Raise an error if the new layer is just a wall of identities,
            # in which case no pending edges could be placed
            if all(edge_id in new_identities for edge_id in current_layer):
                raise ValueError(
                    'No existing edges could be placed into the next layer.'
                    + 'This be because the graph contains cycles.'
                )

            # Add the newly constructed layer to the decomposition
            edge_layers.append(current_layer)

            # If there are still unplaced edges, prepare prev_vertex_layer
            # and ready_edges for construction of the next edge layer and
            # register vertices in the new prev_vertex_layer as placed
            if len(unplaced_edges) > 0:
                ready_edges.clear()
                prev_vertex_layer.clear()
                for edge_id in current_layer:
                    for vertex_id in decomposed.edges[edge_id].targets:
                        prev_vertex_layer.append(vertex_id)
                        placed_vertices.add(vertex_id)

        # After all edges have been placed, rearrange layers to try to minimize
        # crossings of connections between vertices and edges
        for layer_num in range(len(edge_layers)):

            # Rearrange layers in input-to-output and output-to-input
            # directions in simultaneous forward and backward passes
            fwd_pass_layer = edge_layers[layer_num]
            bwd_pass_layer = edge_layers[-layer_num - 1]

            # Determine the source vertices of the forward pass layer
            # and the target vertices of the backward pass layer
            if layer_num == 0:
                source_vertices = decomposed.inputs
                target_vertices = decomposed.outputs
            else:
                # Source vertices of current forward pass layer are targets
                # of previous forward pass layer
                source_vertices = [v for e in edge_layers[layer_num - 1]
                                   for v in decomposed.edges[e].targets]
                # Target vertices of current backward pass layer are sources
                # of previous backward pass layer
                target_vertices = [v for e in edge_layers[-layer_num]
                                   for v in decomposed.edges[e].sources]

            # Normalized vertical order of the source vertices
            source_positions = {vertex: i/len(source_vertices)
                                for i, vertex in enumerate(source_vertices)}
            # Normalized vertical order of the target vertices
            target_positions = {vertex: i/len(target_vertices)
                                for i, vertex in enumerate(target_vertices)}

            # Give edges in the current layers a 'vertical position score'
            # based on the center of mass of their source and/or
            # target vertices
            edge_positions: Dict[int, float] = {edge_id: 0.0 for
                                                edge_id in decomposed.edges}
            for edge_id in fwd_pass_layer:
                sources = decomposed.edges[edge_id].sources
                edge_positions[edge_id] += (sum(source_positions[vertex_id] for
                                                vertex_id in sources)
                                            / len(sources)
                                            if len(sources) != 0 else 0)
            for edge_id in bwd_pass_layer:
                targets = decomposed.edges[edge_id].targets
                edge_positions[edge_id] += (sum(target_positions[vertex_id] for
                                                vertex_id in targets)
                                            / len(targets)
                                            if len(targets) != 0 else 0)

            # Sort the edges in the current layers according to their scores
            fwd_pass_layer.sort(key=lambda edge_id: edge_positions[edge_id])
            # If forward and backward pass layers are the same,
            # no need to sort twice
            if not layer_num == len(edge_layers) - layer_num - 1:
                bwd_pass_layer.sort(
                    key=lambda edge_id: edge_positions[edge_id])

        return decomposed, edge_layers

    def __repr__(self) -> str:
        """Print a layered decomposition of this hypergraph."""
        decomposed, edge_layers = self.layer_decomp()

        max_height = max(len(layer) for layer in edge_layers)
        max_width = max(len(edge.label or '')
                        for edge in decomposed.edges.values())
        repr = ''
        for height in range(max_height):
            current_row = ''
            for layer in edge_layers:
                current_row += (
                    f'{decomposed.edges[layer[height]].label:^{max_width}}; '
                    if height < len(layer) else max_width * ' ' + '; ')

            repr += current_row + '\n'
        return repr
