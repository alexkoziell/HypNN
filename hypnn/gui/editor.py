#    Copyright 2024 Alexander Koziell-Pipe

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QToolBar, QWidget
)

from hypnn.gui.graphview import GraphView
from hypnn.hypergraph import Hypergraph, Vertex, Hyperedge


class Editor(QWidget):
    """Graph editor widget."""

    def __init__(self, graph: Hypergraph | None = None) -> None:
        super().__init__()

        # If not initialized with a hypergraph, start with a new one
        self.graph = Hypergraph() if graph is None else graph

        # GUI elements
        layout = QHBoxLayout()
        self.graph_view = GraphView()
        self.graph_view.set_graph(self.graph)
        layout.addWidget(self.graph_view)

        toolbar = QToolBar('toolbar')
        toolbar.setOrientation(Qt.Orientation.Vertical)

        add_vertex_action = QAction('Add Vertex', self)
        add_vertex_action.setStatusTip('Add a vertex to the hypergraph.')
        add_vertex_action.triggered.connect(self.add_vertex_clicked)
        toolbar.addAction(add_vertex_action)
        add_edge_action = QAction('Add Hyperedge', self)
        add_edge_action.setStatusTip('Add a hyperedge to the hypergraph.')
        add_edge_action.triggered.connect(self.add_hyperedge_clicked)
        toolbar.addAction(add_edge_action)
        add_wire_action = QAction('Add Wire', self)
        add_wire_action.setStatusTip('Add a wire to the hypergraph.')
        add_wire_action.triggered.connect(lambda _: print('Add wire clicked!'))

        toolbar.addAction(add_wire_action)
        layout.addWidget(toolbar)

        self.setLayout(layout)

    def set_graph(self, graph: Hypergraph) -> None:
        """Set the hypergraph being edited."""
        self.graph_view.set_graph(graph)

    def add_vertex_clicked(self) -> None:
        """Handle add vertex button action."""
        self.graph_view.add_vertex(Vertex())

    def add_hyperedge_clicked(self) -> None:
        """Handle add vertex button action."""
        self.graph_view.add_edge(Hyperedge([], [], 'f'))
