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
from typing import Any, Generic, Self, TypeVar


@dataclass
class Vertex:
    """A hypergraph vertex.

    Vertices may be typed. This still permits untyped hypergraphs, which are
    equivalent to typed hypergraphs where all vertices have the same type
    (eg `None`).
    """

    vtype: Any | None = None
    """A type associated with the vertex."""
    sources: set[int] = field(default_factory=set)
    """A set of integer identifiers for hyperedges into the vertex."""
    targets: set[int] = field(default_factory=set)
    """A set of integer identifiers for hyperedges from the hyperedge."""


@dataclass
class Hyperedge:
    """A hypergraph edge.

    This particular flavour of hyperedge has a total ordering on each of the
    source and target vertices, implemented by storing them as lists.
    """

    sources: list[int]
    """A list of integer identifiers for vertices into the hyperedge."""
    targets: list[int]
    """A list of integer identifiers for vertices from the hyperedge."""
    label: str | None = None
    """A label to identify the hyperedge when drawing."""
    identity: bool = False
    """Whether this hyperedge is an identity or not."""

    @classmethod
    def create_identity(cls, sources: list[int],
                        targets: list[int]) -> Self:
        """Create an identity operation."""
        return cls(sources, targets,
                   label='id', identity=True)


VertexType = TypeVar('VertexType', bound=Vertex)
EdgeType = TypeVar('EdgeType', bound=Hyperedge)


class BaseHypergraph(Generic[VertexType, EdgeType]):
    """Base class for hypergraphs: subclass for concrete implementations.

    This particular flavour of hypergraph has two totally ordered sets of
    priviledged vertices: the input and output vertices. Together, these
    are referred to as boundary vertices.

    This base class is designed to be subclassed, specifying a `VertexType`,
    `EdgeType` and assigning these to :py:attr:`vertex_class` and
    :py:attr:`edge_class` respectively. These types can :py:class:`Vertex`
    and :py:class:`Hyperedge` or subclasses thereof in order to specify
    additional behaviours.

    By default, hypergraph instances begin with no vertices or hyperedges,
    which can then be added using :py:meth:`add_vertex`, :py:meth:`add_edge`
    to build out the hypergraph.
    """

    vertex_class: type[VertexType]
    edge_class: type[EdgeType]

    def __init__(self) -> None:
        """Initialize a :py:class:`BaseHypergraph` subclass."""
        if not (hasattr(self, 'vertex_class')
                and hasattr(self, 'edge_class')):
            raise TypeError(
                'Cannot instantiate hypergraph class without specifying vertex'
                + 'and hyperedge classes to use.'
            )
        self.vertices: dict[int, VertexType] = {}
        self.edges: dict[int, EdgeType] = {}
        self.inputs: list[int] = []
        self.outputs: list[int] = []

    def create_identity(self, sources: list[int],
                        targets: list[int]) -> EdgeType:
        """Return an identity hyperedge.

        The source and target types of identities must be the same, and there
        is a unique identity for each such type.
        Thus only the source and target ids should be required
        to build an identity edge.

        Implement this method in subclasses to specify the operational
        behaviour of identity hyperedges.

        Args:
            sources: A list of integer identifiers for vertices
                directed to the hyperedge.
            targets: A list of integer identifiers for vertices
                    directed from the hyperedge.

        Returns:
            edge: An identity hyperedge.
        """
        if (len(sources) != len(targets) or
            any(self.vertices[s].vtype != self.vertices[t].vtype
                for s, t in zip(sources, targets))):
            raise ValueError(
                'Identity source and target types must match.'
            )
        return self.edge_class.create_identity(sources, targets)

    def add_vertex(self, vertex: VertexType) -> int:
        """Add a vertex to the hypergraph.

        Args:
            vertex: The vertex to be added to the hypergraph.

        Returns:
            vertex_id: A unique integer identifier for the hypergraph
                       to reference the newly added vertex.
        """
        # Give the new vertex a unique integer identifier
        vertex_id = max((i for i in self.vertices.keys()), default=-1) + 1
        self.vertices[vertex_id] = vertex
        return vertex_id

    def add_edge(self, edge: EdgeType) -> int:
        """Add a new hyperedge to the hypergraph.

        Args:
            edge: The hyperedge to be added to the hypergraph.

        Returns:
            edge_id: A unique integer identifier for the hypergraph
                     to reference the newly added hyperedge.
        """
        # Check all the vertex identifiers in sources and targets correspond to
        # existing vertices in the hypergraph.
        if not all(vertex in self.vertices
                   for boundary in (edge.sources, edge.targets)
                   for vertex in boundary):
            raise ValueError(
              'Hyperedge attached to vertices that are not in the hypergraph.'
            )

        # The source and target types must be the same for identity hyperedges
        if edge.identity:
            if (len(edge.sources) != len(edge.targets)
                or any(self.vertices[s].vtype != self.vertices[o].vtype
                       for s, o in zip(edge.sources, edge.targets))):
                raise ValueError(
                    'Source and target types of identity hyperedges must match'
                )

        # Give the new hyperedge a unique integer identifier
        edge_id = max((i for i in self.edges.keys()), default=-1) + 1
        self.edges[edge_id] = edge

        # Register the edge as source or target of relevant vertices
        for s in edge.sources:
            self.vertices[s].targets.add(edge_id)
        for t in edge.targets:
            self.vertices[t].sources.add(edge_id)

        return edge_id

    def parallel_comp(self, other: BaseHypergraph,
                      in_place: bool = False) -> BaseHypergraph | None:
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

        # Add copies of each vertex in other to self and create a
        # one-to-one correspondence between the added vertices and
        # the vertices of other
        vertex_map: dict[int, int] = {}
        for vertex_id, vertex in other.vertices.items():
            # Copying allows additional data to be carried over in subclasses
            copied_vertex = deepcopy(vertex)
            # Reset the sources and targets, these will be updated later
            copied_vertex.sources.clear()
            copied_vertex.targets.clear()
            vertex_map[vertex_id] = composed.add_vertex(copied_vertex)

        # Add copies of edge in other to self, with connectivity that
        # matches that of other in a compatible way with the vertex map
        # established above
        for edge in other.edges.values():
            # Copying allows additional data to be carried over in subclasses
            copied_edge = deepcopy(edge)
            # Update the sources and targets to their images in composed
            copied_edge.sources = [vertex_map[s] for s in edge.sources]
            copied_edge.targets = [vertex_map[t] for t in edge.targets]
            # The add_edge method will sort out the sources and targets
            # of the vertices copied from `other`
            composed.add_edge(copied_edge)

        # For parallel composition, append the boundary vertices of other
        # to the boundary vertices of self.
        composed.inputs += [vertex_map[i] for i in other.inputs]
        composed.outputs += [vertex_map[o] for o in other.outputs]

        # To make it clear when an in place modifications has been made,
        # return None if in_place
        return None if in_place else composed

    def sequential_comp(self, other: BaseHypergraph,
                        in_place: bool = False) -> BaseHypergraph | None:
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

        # Copy each non-boundary vertex in other to self and create a
        # one-to-one correspondence between the copied vertices
        # and the vertices of other
        vertex_map: dict[int, int] = {}
        for vertex_id, vertex in other.vertices.items():
            # Inputs of other must be mapped to outputs of self
            if vertex_id in other.inputs:
                continue
            # Copying allows additional data to be carried over in subclasses
            copied_vertex = deepcopy(vertex)
            # Reset the sources and targets, these will be updated later
            copied_vertex.sources.clear()
            copied_vertex.targets.clear()
            vertex_map[vertex_id] = composed.add_vertex(copied_vertex)

        # Identify output vertices of self with input vertices of other
        for out_vid, in_vid in zip(composed.outputs, other.inputs):
            vertex_map[in_vid] = out_vid

        # The outputs of the composition correspond to the outputs of other
        composed.outputs = [vertex_map[o] for o in other.outputs]

        # Copy edges of other into self, with connectivity that
        # matches that of other in a compatible way with the vertex map
        # established above
        for edge in other.edges.values():
            # Copying allows additional data to be carried over in subclasses
            copied_edge = deepcopy(edge)
            # Update the sources and targets to their images in composed
            copied_edge.sources = [vertex_map[s] for s in edge.sources]
            copied_edge.targets = [vertex_map[t] for t in edge.targets]
            # The add_edge method will sort out the sources and targets
            # of the vertices copied from `other`
            composed.add_edge(copied_edge)

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
            identity_id: The identifier of the new identity hyperedge.
        """
        # Create a new vertex with the same attributes as the original vertex
        old_vertex = self.vertices[vertex_id]
        # Sources and targets will be updated later
        new_vertex = deepcopy(old_vertex)
        new_vertex_id = self.add_vertex(new_vertex)

        # Add the new identity hyperedge
        identity_edge = self.create_identity([vertex_id],
                                             [new_vertex_id])
        identity_id = self.add_edge(identity_edge)
        # Sources of the new vertex is just the new identity edge, and
        # targets of are those of the original vertex
        new_vertex.sources = {identity_id}
        new_vertex.targets = set(
            target_id for target_id in old_vertex.targets
            # add_edge method added identity edge as target of old vertex
            if target_id != identity_id
        )

        # New targets of the original vertex is just the new identity edge
        old_vertex.targets = {identity_id}
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

        return identity_id

    def insert_vertex_to_edge_identities(self, vertex_id: int,
                                         edge_id: int) -> set[int]:
        """Add identity hyperedges between a vertex and one of its targets.

        This method disconnects the original vertex from the hyperedge and
        adds a new vertex connected to the hyperedge in the orignal vertex's
        place. The original vertex is then connected to the new vertex by an
        identity hyperedge.

        Args:
            vertex_id: The identifier of the vertex to be connected to the new
                       the identity.
            edge_id: The identifier of the hyperedge to be connected to the
                       new the identity.

        Returns:
            identity_ids: The identifier of the new identity hyperedges.
        """
        old_vertex = self.vertices[vertex_id]
        identity_ids = set()

        for port_num, source_id in enumerate(self.edges[edge_id].sources):
            if source_id == vertex_id:
                # Create a new vertex with the same attributes as the original
                # vertex. Sources and targets will be updated later
                new_vertex = deepcopy(old_vertex)
                new_vertex_id = self.add_vertex(new_vertex)

                # Add the new identity hyperedge
                identity_edge = self.create_identity([vertex_id],
                                                     [new_vertex_id])
                identity_id = self.add_edge(identity_edge)
                identity_ids.add(identity_id)
                # Sources of the new vertex is just the new identity edge, and
                # targets is the original hyperedge
                new_vertex.sources = {identity_id}
                new_vertex.targets = {edge_id}

                # Disconnect original vertex from hyperedge
                if edge_id in old_vertex.targets:
                    old_vertex.targets.remove(edge_id)
                # Connect to new identity hyperedge
                old_vertex.targets.add(identity_id)
                # Update the sources lists the edge to replace
                # the orignal vertex with the new vertex
                self.edges[edge_id].sources[port_num] = new_vertex_id

        return identity_ids

    def insert_edge_to_vertex_identities(self, edge_id: int,
                                         vertex_id: int) -> set[int]:
        """Add identity hyperedges between an edge and one of its targets.

        This method disconnects the original vertex from the hyperedge and
        adds a new vertex connected to the hyperedge in the orignal vertex's
        place. The original vertex is then connected to the new vertex by an
        identity hyperedge.

        Args:
            edge_id: The identifier of the hyperedge to be connected to the
                       new the identity.
            vertex_id: The identifier of the vertex to be connected to the new
                       the identity.

        Returns:
            identity_ids: The identifier of the new identity hyperedges.
        """
        old_vertex = self.vertices[vertex_id]
        identity_ids = set()

        for source_edge_id in old_vertex.sources.copy():
            if source_edge_id == edge_id:
                for port_num, target_vertex_id in enumerate(
                    self.edges[edge_id].targets
                ):
                    if target_vertex_id == vertex_id:
                        # Create a new vertex with the same attributes as the
                        # original vertex.
                        # Sources and targets will be updated later
                        new_vertex = deepcopy(old_vertex)
                        new_vertex_id = self.add_vertex(new_vertex)

                        # Add the new identity hyperedge
                        identity_edge = self.create_identity([new_vertex_id],
                                                             [vertex_id])
                        identity_id = self.add_edge(identity_edge)
                        identity_ids.add(identity_id)
                        # Sources of the new vertex is just the new identity
                        # edge, and targets is the original hyperedge
                        new_vertex.sources = {edge_id}
                        new_vertex.targets = {identity_id}

                        # Disconnect original vertex from hyperedge
                        if edge_id in old_vertex.sources:
                            old_vertex.sources.remove(edge_id)
                        # Connect to new identity hyperedge
                        old_vertex.sources.add(identity_id)
                        # Update the target list of the edge to replace
                        # the orignal vertex with the new vertex
                        self.edges[edge_id].targets[port_num] = new_vertex_id

        return identity_ids

    def layer_decomp(self, in_place: bool = False
                     ) -> tuple[BaseHypergraph, list[list[int]]]:
        """Decompose this hypergraph into layers of its hyperedges.

        This decomposition requires the hypergraph to be acyclic and
        'source monogamous': that every vertex has at most one source
        edge.

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
        edge_layers: list[list[int]] = []
        # The target vertices of the edge layer
        # most recently added to the decomposition
        prev_vertex_layer: list[int] = []
        # The vertices already sitting between two edge
        # layers already added to the decomposition
        placed_vertices: set[int] = set()
        # Edges ready to be placed into the current layer
        ready_edges: set[int] = set()

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
        unplaced_edges: set[int] = set(decomposed.edges.keys())
        # Track newly created identity edges from this point
        # to ensure new layers contain at least one non-identity edge
        new_identities: set[int] = set()

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
                if vertex_id in decomposed.outputs:
                    new_identity = decomposed.insert_identity_after(vertex_id)
                    new_identities.add(new_identity)
                    ready_edges.add(new_identity)
                vertex_targets = decomposed.vertices[vertex_id].targets.copy()
                for edge_id in vertex_targets:
                    if edge_id not in ready_edges:
                        identity_ids = (
                            decomposed.insert_vertex_to_edge_identities(
                                vertex_id, edge_id
                            ))
                        new_identities |= identity_ids
                        ready_edges |= new_identities

            # Populate the current layer with edges that are ready to
            # be placed, and remove them from the set of unplaced edges
            current_layer: list[int] = []
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
            edge_positions: dict[int, float] = {edge_id: 0.0 for
                                                edge_id in decomposed.edges}
            for edge_id in fwd_pass_layer:
                sources = decomposed.edges[edge_id].sources
                edge_positions[edge_id] += (
                    sum(source_positions[vertex_id]
                        # nodes with multiple targets may
                        # not appear in source_positions
                        if vertex_id in source_positions
                        else 0
                        for vertex_id in sources)
                    / len(sources)
                    if len(sources) != 0 else 0)
            for edge_id in bwd_pass_layer:
                targets = decomposed.edges[edge_id].targets
                edge_positions[edge_id] += (sum(
                    target_positions[vertex_id] for
                    vertex_id in targets
                    if vertex_id in target_positions
                    )
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

    def frobenius_layer_decomp(self, in_place: bool = False
                               ) -> tuple[BaseHypergraph, list[list[int]]]:
        """Decompose this hypergraph into layers of vertices and hyperedges.

        This decomposition requires that the hypergraph is acyclic.

        Args:
            in_place: Whether to modify `self` rather than creating
                      a new :py:class:`Hypergraph` instance.

        Returns:
            tuple: a tuple containing multiple values
                - decomposed: The original hypergraph with possible additional
                  identity edges inserted during the decomposition.
                - layers: A list of alternating lists of vertex and edge ids
                  corresponding to layers of the decomposition.
        """
        # If in_place, modify graph when identity edges are inserted
        decomposed = self if in_place else deepcopy(self)

        # This will become the final layer decomposition
        layers: list[list[int]] = []

        # A vertex is ready if all its source edges have been placed
        ready_vertices: set[int] = set()

        # Edges and vertices not yet placed into any layer
        unplaced_vertices: set[int] = set(decomposed.vertices.keys())
        unplaced_edges: set[int] = set(decomposed.edges.keys())

        # First layer: zero-source vertices
        vertex_layer: list[int] = []
        for vertex_id, vertex in decomposed.vertices.items():
            if len(vertex.sources) == 0:
                ready_vertices.add(vertex_id)
                unplaced_vertices.remove(vertex_id)
                vertex_layer.append(vertex_id)

        layers.append(vertex_layer)

        while len(unplaced_edges) > 0:
            # Edges ready to be placed into the current edge layer
            ready_edges: set[int] = set()
            # Track newly created identity edges to ensure
            # each layer contains at least one non-identity edge
            new_identities: set[int] = set()

            # If all the source vertices of an edge are ready,
            # it is ready to be added to the layer decomposition
            for edge_id in unplaced_edges:
                if all(vertex_id in ready_vertices
                       for vertex_id in decomposed.edges[edge_id].sources):
                    ready_edges.add(edge_id)

            # For all ready vertices
            for vertex_id in ready_vertices:
                # If a vertex is an output, traverse the layer
                # with an identity edge
                if vertex_id in decomposed.outputs:
                    new_identity = decomposed.insert_identity_after(vertex_id)
                    new_identities.add(new_identity)
                    ready_edges.add(new_identity)
                vertex_targets = decomposed.vertices[vertex_id].targets.copy()
                for edge_id in vertex_targets:
                    # If a target edge is unplaced and still not
                    # yet ready to be placed, add an identity hyperedge
                    # to traverse this layer
                    if (edge_id in unplaced_edges
                       and edge_id not in ready_edges):
                        identity_ids = (
                         decomposed.insert_vertex_to_edge_identities(
                            vertex_id, edge_id
                         ))
                        new_identities |= identity_ids
                        ready_edges |= new_identities

            # Populate the current layer with edges that are ready to
            # be placed, and remove them from the set of unplaced edges
            current_edge_layer: list[int] = []
            for edge_id in ready_edges:
                current_edge_layer.append(edge_id)
                unplaced_edges.discard(edge_id)

            # Raise an error if the new layer is just a wall of identities,
            # in which case no pending edges could be placed
            if all(edge_id in new_identities
                   for edge_id in current_edge_layer):
                raise ValueError(
                    'No existing edges could be placed into the next layer.'
                    + 'This may be because the graph contains cycles.'
                )

            # Get next vertex layer and update ready_vertices
            current_vertex_layer: list[int] = []
            for vertex_id, vertex in decomposed.vertices.items():
                if not any(source_id in unplaced_edges
                           for source_id in vertex.sources):
                    if vertex_id not in ready_vertices:
                        current_vertex_layer.append(vertex_id)
                        ready_vertices.add(vertex_id)

            for edge_id in current_edge_layer:
                edge_targets = decomposed.edges[edge_id].targets.copy()
                for vertex_id in edge_targets:
                    if vertex_id not in ready_vertices:
                        identity_ids = (
                         decomposed.insert_edge_to_vertex_identities(
                            edge_id, vertex_id
                         ))
                        for identity_id in identity_ids:
                            # identity edges only have one source vertex
                            identity = decomposed.edges[identity_id]
                            new_vertex_id = identity.sources[0]
                            current_vertex_layer.append(new_vertex_id)
                            ready_vertices.add(new_vertex_id)
                        unplaced_edges |= identity_ids

            # Add the newly constructed layers to the decomposition
            layers.append(current_edge_layer)
            layers.append(current_vertex_layer)

            # TODO: move zero source, non-input vertices forward and move
            # zero target, non-output vertices backward where possible

        # After all edges have been placed, rearrange layers to try to minimize
        # crossings of connections between vertices and edges
        for layer_num in range(1, len(layers)):

            # Rearrange layers in input-to-output and output-to-input
            # directions in simultaneous forward and backward passes
            fwd_pass_layer = layers[layer_num]
            prev_fwd_layer = layers[layer_num]
            bwd_pass_layer = layers[-layer_num - 1]
            prev_bwd_layer = layers[-layer_num]

            # Normalized vertical order of the source vertices
            source_positions = {vertex_id: i/len(prev_fwd_layer)
                                for i, vertex_id in enumerate(prev_fwd_layer)}
            # Normalized vertical order of the target vertices
            target_positions = {vertex_id: i/len(prev_bwd_layer)
                                for i, vertex_id in enumerate(prev_bwd_layer)}

            # Give items in the current layers a 'vertical position score'
            # based on the center of mass of their source and/or
            # target vertices
            fwd_positions: dict[int, float] = {id: 0.0 for
                                               id in fwd_pass_layer}
            bwd_positions: dict[int, float] = {id: 0.0 for
                                               id in bwd_pass_layer}
            for id in fwd_pass_layer:
                sources: set[int] | list[int]
                if layer_num % 2 == 0:  # vertex layer
                    sources = decomposed.vertices[id].sources
                else:  # edge layer
                    sources = decomposed.edges[id].sources
                fwd_positions[id] += (
                    sum(source_positions[id]
                        # nodes with multiple targets may
                        # not appear in source_positions
                        if id in source_positions
                        else 0
                        for id in sources)
                    / len(sources)
                    if len(sources) != 0 else 0)
            for id in bwd_pass_layer:
                targets: set[int] | list[int]
                if layer_num % 2 == 0:  # vertex layer
                    targets = decomposed.vertices[id].targets
                else:  # edge layer
                    targets = decomposed.edges[id].targets
                bwd_positions[id] += (
                    sum(target_positions[id]
                        # nodes with multiple sources may
                        # not appear in target_positions
                        if id in target_positions
                        else 0
                        for id in targets)
                    / len(targets)
                    if len(targets) != 0 else 0)

            # Sort the edges in the current layers according to their scores
            fwd_pass_layer.sort(key=lambda id: fwd_positions[id])
            # If forward and backward pass layers are the same,
            # no need to sort twice
            if not layer_num == len(layers) - layer_num - 1:
                bwd_pass_layer.sort(key=lambda id: bwd_positions[id])

        return decomposed, layers

    def __repr__(self) -> str:
        """Print a layered decomposition of this hypergraph."""
        try:
            decomposed, edge_layers = self.layer_decomp()
        except ValueError:
            decomposed, layers = self.frobenius_layer_decomp()
            edge_layers = layers[1::2]

        layer_reprs = (
            ('(' if len(layer) > 1 else '')
            + ' ⨂ '.join(decomposed.edges[edge_id].label for edge_id in layer)
            + (')' if len(layer) > 1 else '')
            for layer in edge_layers)

        repr = ' ⨟ '.join(layer_reprs)

        return repr


class Hypergraph(BaseHypergraph[Vertex, Hyperedge]):
    """Concrete hypergraph class."""

    vertex_class = Vertex
    edge_class = Hyperedge
