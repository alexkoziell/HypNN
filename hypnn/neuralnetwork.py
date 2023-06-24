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
from typing import Callable

import numpy as np

from hypnn.hypergraph import Vertex, Hyperedge, Hypergraph


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
            raise ValueError('Incompatible value assigned to vertex.')

    def set_value(self, value: np.ndarray) -> None:
        """Set this variable to a specific value."""
        if value.shape != self.vtype:
            raise ValueError('Incompatible value assigned to vertex.')
        self.value = value


class Operation(Hyperedge):
    """Represents a numerical operation in a neural network.

    Subclasses :py:class:`Hyperedge`, inheriting it's attributes
    as well as defining a new :py:attr:`operation` attribute.

    Attributes:
        operation: The numerical operation performed by this hyperedge.
    """

    def __init__(self, operation: Callable[[list[np.ndarray]],
                                           list[np.ndarray]],
                 sources: list[int],
                 targets: list[int],
                 label: str | None = None,
                 identity: bool = False) -> None:
        """Initialize an :py:class:`Operation`."""
        self.operation = operation
        super().__init__(sources, targets, label, identity)

    @staticmethod
    def create_identity(sources: list[int],
                        targets: list[int]) -> Operation:
        """Return an identity operation.

        Args:
            sources: A list of integer identifiers for vertices
                directed to the hyperedge.
            targets: A list of integer identifiers for vertices
                    directed from the hyperedge.

        Returns:
            operation: An identity operation.
        """
        return Operation(lambda x: x, sources, targets,
                         label='id', identity=True)


class NeuralNetwork(Hypergraph):
    """A hypergraph representing a neural network.

    Vertices send values to their target hyperedges, which
    perform computations on their recieved values and transmit
    them to their target vertices. This occurs from the inputs
    to the outputs of the hypergraph.
    """

    vertex_type = Variable
    edge_type = Operation
