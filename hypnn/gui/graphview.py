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
"""Hypergraph view widget."""
from __future__ import annotations

from PySide6.QtCore import QPointF
from PySide6.QtGui import QBrush, QColor, QPainter
from PySide6.QtWidgets import (
    QGraphicsScene, QGraphicsView,
    QGraphicsEllipseItem, QGraphicsRectItem,
    QGraphicsTextItem
)

from hypnn.hypergraph import BaseHypergraph
from hypnn.gui.drawinfo import HypergraphDrawInfo


class GraphView(QGraphicsView):
    """Graphics View for Hypergraphs."""

    def __init__(self) -> None:
        self.graph_scene = GraphScene()
        super().__init__(self.graph_scene)
        self.setMouseTracking(True)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

    def set_graph(self, graph: BaseHypergraph) -> None:
        self.graph_scene.set_graph(graph)
        self.repaint()
        self.centerOn(0, 0)


class VertexItem(QGraphicsEllipseItem):
    """Vertex graphics item."""
    def __init__(self, x: float, y: float, radius: float, label: str | None) -> None:
        super().__init__(x, y, radius, radius)
        self.setBrush(QBrush(QColor(0, 0, 0)))

        # Add label as a text item
        if label is not None:
            self.textItem = QGraphicsTextItem(label)
            self.textItem.setDefaultTextColor(QColor(0, 0, 0))
            text_position = self.pos() + QPointF(0, -1.5 * self.rect().height())
            self.textItem.setPos(text_position)

    def refresh(self) -> None:
        # Update text position
        if hasattr(self, 'textItem'):
            text_position = self.pos() + QPointF(0, self.rect().height())
            self.textItem.setPos(text_position)


class GraphScene(QGraphicsScene):
    """Graphics Scene for Hypergraphs."""

    def __init__(self, x_scale: float = 50.0, y_scale: float = 50.0,
                 vertex_radius: float = 0.125) -> None:
        super().__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.vertex_radius = vertex_radius

    def set_graph(self, graph: BaseHypergraph) -> None:
        """Set the graph displayed in the graphics scene."""
        self.draw_info = HypergraphDrawInfo(graph)
        self.clear()
        self.add_items()

    def add_vertices(self) -> None:
        """Add vertices to the graph scene."""
        for vertex_id, vertex_draw_info in self.draw_info.vertices.items():
            x = vertex_draw_info.x * self.x_scale
            y = vertex_draw_info.y * self.y_scale
            radius = self.vertex_radius * (self.x_scale + self.y_scale)
            if vertex_draw_info.label is not None:
                label = vertex_draw_info.label()
            else:
                label = None
            self.addItem(VertexItem(x, y, radius, label))

    def add_items(self) -> None:
        """Add graphics items for the current graph."""
        self.add_vertices()

