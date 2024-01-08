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
"""Neural networks as hypergraphs."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Self

import numpy as np

from hypnn.hypergraph import Vertex, Hyperedge, BaseHypergraph


@dataclass
class Variable(Vertex):
    """Represents a variable in a neural network computation.

    Subclasses :py:class:`Vertex`.
    Variables can hold numeric values, which take the form of
    finite-dimensional tensors. Hence :py:attr:`vtype` is restricted to
    be tuple of integers indicating the tensor dimensions.
    """

    vtype: tuple[int, ...]
    """The dimensions of the tensor values this vertex can hold."""
    value: np.ndarray | None = None
    """The value carried by this vertex."""
    name: str | None = None
    """A name for this variable."""

    @property
    def label(self) -> str:
        """Return a label to be drawn with this variable.

        The label consists of the variable :py:attr:`name`, if it exists
        and the :py:attr:`value`, if it exists.
        """
        if self.value is not None:
            return f'{self.name} = {self.value}'
        elif self.name is not None:
            return self.name
        else:
            return str()

    def __post_init__(self) -> None:
        """Perform post-initialization checks.

        This checks that value of the variable matches its specified
        tensor dimensions, as well as putting `floats` and `int` values
        into a numpy array.
        """
        if self.value is not None and not isinstance(self.value, np.ndarray):
            self.value = np.array(self.value)
        if self.value is not None and self.value.shape != self.vtype:
            raise ValueError(
                f'Incompatible value assigned to vertex {self.name}.'
            )

    def set_value(self, value: np.ndarray | None) -> None:
        """Set this variable to a specific value."""
        if value is None:
            self.value = value
            return
        if value.shape != self.vtype:
            raise ValueError(
                f'Incompatible value of shape {value.shape} assigned to '
                + f'vertex of type {self.vtype}.'
            )
        self.value = value


class Operation(Hyperedge):
    """Represents a numerical operation in a neural network.

    Subclasses :py:class:`Hyperedge`, inheriting it's attributes
    as well as defining a new :py:attr:`operation` attribute.

    Attributes:
        operation: The numerical operation performed by this hyperedge.
        reverse_derivative: The reverse derivative of :py:attr:`operation`,
                            specified manually.
    """

    def __init__(self, operation: Callable[[list[np.ndarray | None]],
                                           list[np.ndarray | None]],
                 sources: list[int],
                 targets: list[int],
                 reverse_derivative: Callable[[list[np.ndarray | None]],
                                              list[np.ndarray | None]] | None
                 = None,
                 label: str | None = None,
                 identity: bool = False) -> None:
        """Initialize an :py:class:`Operation`."""
        self.operation = operation
        self.reverse_derivative = reverse_derivative
        super().__init__(sources, targets, label, identity)

    @classmethod
    def create_identity(cls, sources: list[int],
                        targets: list[int]) -> Self:
        """Create an identity operation."""
        return cls(lambda x: x, sources, targets,
                   reverse_derivative=lambda xs: [xs[1]] or [None],
                   label='id', identity=True)


class NeuralNetwork(BaseHypergraph[Variable, Operation]):
    """A hypergraph representing a neural network.

    Vertices send values to their target hyperedges, which perform
    computations on their recieved values and transmit them to their target
    vertices. This occurs from the inputs to the outputs of the hypergraph.
    """

    vertex_class = Variable
    edge_class = Operation

    def topological_sort(self) -> list[int]:
        """Sorts hyperedges for performing computation.

        The hyperedges are sorted into a list such that, provided values
        for all input variables are provided, they can be applied in sequence
        with the computations performed by all previous hyperedges in the list
        ensuring all the source variables of the current hyperedge have been
        evaluated.

        Returns:
            sorted_edges: A list of hyperedge identifiers sorted to allow
                          forward or backward computation.
        """
        # Track which variables have been evaluated to determine which
        # hyperedges would be ready to perform their operation
        # given the edges currently in sorted_edges had been evaluated.
        # Input (output) vertices are assumed to start evaluated.
        evaluated_variables: set[int] = set(self.inputs)

        # Hyperedges with no source vertices are immediately ready to evaluate
        sorted_edges = [edge_id for edge_id, edge in self.edges.items()
                        if len(edge.sources) == 0]
        # Register appropriate vertices as evaluated
        for edge_id in sorted_edges:
            edge = self.edges[edge_id]
            evaluated_variables |= set(edge.targets)

        # TODO: proof that this terminates
        while len(evaluated_variables) < len(self.vertices):
            new_evaluated_variables: set[int] = set()
            # The only edges that would be ready to evaluate would have
            # all their source vertices evaluated, so only check the targets
            # of evaluated vertices to save checking all the edges
            for vertex_id in evaluated_variables:
                edge_ids = self.vertices[vertex_id].targets
                for edge_id in edge_ids:
                    # Skip edges already in sorted_edges
                    # (they would have already been evaluated)
                    if edge_id in sorted_edges:
                        continue
                    edge = self.edges[edge_id]
                    vertex_ids = edge.sources
                    # Eagerly add edges ready to evaluate to sorted_edges
                    if all(vertex_id in evaluated_variables
                           for vertex_id in vertex_ids):
                        sorted_edges.append(edge_id)
                        # Register variables that would now be evaluated
                        new_evaluated_variables |= set(edge.targets)
            # If no new variables have been evaluated, the next loop iteration
            # will be identical, and so the loop itself won't terminate.
            if len(new_evaluated_variables) == 0:
                raise RuntimeError(
                    'Could not sort hyperedges. Is the hypergraph acyclic?'
                )
            # Once all edges that were ready to evaluate at the beginning of
            # of this loop iteration have been added to sorted_edges, update
            # evaluated_variables for another loop
            evaluated_variables |= new_evaluated_variables

        return sorted_edges

    def compute(self):
        """Perform a computation through the network.

        This can only be done if input variable values are specified.
        """
        # Sort the edges in a valid sequence to evaluate them in
        edges_to_compute = self.topological_sort()

        # Perform the operations in sequence
        for edge_id in edges_to_compute:
            edge = self.edges[edge_id]
            source_values = [self.vertices[source].value
                             for source in edge.sources]
            target_values = edge.operation(source_values)
            for target, value in zip(edge.targets, target_values):
                self.vertices[target].set_value(value)

    def reverse_derivative(self) -> NeuralNetwork:
        """Return the reverse derivative of this neural network hypergraph.

        The reverse derivative will compute the derivatives of each
        variable with respect to the outputs of the original graph via
        reverse-mode differentiation.

        Edges of type A -> B are mapped to edges of type A X B -> A

        Returns:
            reverse_derivative: The reverse derivative of this NN hypergraph.
        """
        reverse_derivative = NeuralNetwork()

        # Keep track of which variable in the reverse derivative hypergraph
        # evalutates the derivative of each variable in the original
        vmap: dict[int, int] = {}

        # First create the variables that correspond to derivatives
        for vertex_id, vertex in self.vertices.items():
            new_vertex = Variable(vertex.vtype, name=f'd{vertex.name}')
            new_vertex_id = reverse_derivative.add_vertex(new_vertex)
            vmap[vertex_id] = new_vertex_id

        # Second, create the edges and additional input vertices
        for edge in self.edges.values():
            # Reverse derivative operations for all edges must be defined
            if edge.reverse_derivative is None:
                raise ValueError(
                    f'No reverse derivative specified for {edge}'
                )
            # Create additional input vertices
            new_vertices = [
                Variable(self.vertices[vertex_id].vtype,
                         name=self.vertices[vertex_id].name)
                for vertex_id in edge.sources
            ]
            new_inputs = [
                reverse_derivative.add_vertex(new_vertex)
                for new_vertex in new_vertices
            ]
            reverse_derivative.inputs += new_inputs
            # Create reverse derivative hyperedges
            new_edge = Operation(
                edge.reverse_derivative,
                # source and targets are reversed
                new_inputs +
                [vmap[target_id] for target_id in edge.targets],
                [vmap[source_id] for source_id in edge.sources],
                label=f'R[{edge.label}]',
            )
            reverse_derivative.add_edge(new_edge)

        # Set inputs and outputs of reverse derivative hypergraph that
        # correspond to outputs and inputs of original hypergraph
        reverse_derivative.inputs += [vmap[output_id]
                                      for output_id in self.outputs]
        reverse_derivative.outputs = [vmap[input_id]
                                      for input_id in self.inputs]

        return reverse_derivative
