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
from PySide6.QtWidgets import (
    QMainWindow, QPlainTextEdit, QVBoxLayout, QWidget
)

from hypnn.gui.graphview import GraphView
from hypnn.hypergraph import Hypergraph, Hyperedge, Vertex


class MainWindow(QMainWindow):
    """Main window of HypNN."""
    def __init__(self):
        super().__init__()

        stress_test = Hypergraph()
        for _ in range(18):
            stress_test.add_vertex(Vertex())
        stress_test.add_edge(
            Hyperedge([0, 1, 2], [3, 12, 16, 4], 'f')
        )
        stress_test.add_edge(
            Hyperedge([6, 13, 5, 9, 14], [11, 10], 'g')
        )
        stress_test.add_edge(
            Hyperedge([4, 3, 15, 8], [5, 6, 7], 'h')
        )
        stress_test.add_edge(Hyperedge([12], [], 'e1'))
        stress_test.add_edge(Hyperedge([16, 17], [], 'e2'))
        stress_test.add_edge(Hyperedge([], [13, 17, 14], 's1'))
        stress_test.add_edge(Hyperedge([], [15], 's2'))
        stress_test.inputs = [2, 9, 8, 0, 1]
        stress_test.outputs = [10, 7, 11]

        self.setWindowTitle('HypNN')

        layout = QVBoxLayout()
        graph_view = GraphView()
        graph_view.set_graph(stress_test)
        layout.addWidget(graph_view)
        layout.addWidget(QPlainTextEdit())

        widget = QWidget()
        widget.setLayout(layout)

        self.setCentralWidget(widget)

        self.show()
