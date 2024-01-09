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

from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QTransform
from PySide6.QtWidgets import (
    QGraphicsScene, QGraphicsSceneMouseEvent, QGraphicsView,
    QGraphicsEllipseItem, QGraphicsPathItem, QGraphicsRectItem,
    QGraphicsTextItem
)

from hypnn.hypergraph import BaseHypergraph, Hyperedge, Vertex
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
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.repaint()
        self.centerOn(0, 0)

    def add_vertex(self, vertex: Vertex) -> None:
        """Add a vertex to the graph view."""
        self.graph_scene.add_vertex(vertex)

    def add_edge(self, edge: Hyperedge) -> None:
        """Add an edge to the graph view"""
        self.graph_scene.add_edge(edge)


class VertexItem(QGraphicsEllipseItem):
    """Vertex graphics item."""
    def __init__(self, x: float, y: float, radius: float, label: str | None,
                 identifier: int) -> None:
        super().__init__(-radius, -radius, 2 * radius, 2 * radius)
        self.setBrush(QBrush(QColor('black')))
        self.setPos(x, y)
        self.identifier = identifier

        # Add label as a text item
        if label is not None:
            self.textItem = QGraphicsTextItem(label)
            self.textItem.setDefaultTextColor(QColor(0, 0, 0))
            text_position = self.pos() + QPointF(0,
                                                 -1.5 * self.rect().height())
            self.textItem.setPos(text_position)

    def refresh(self) -> None:
        # Update text position
        if hasattr(self, 'textItem'):
            text_position = self.pos() + QPointF(0, self.rect().height())
            self.textItem.setPos(text_position)


class EdgeItem(QGraphicsRectItem):
    """Hyperedge graphics item."""

    def __init__(self, x: float, y: float, width: float, height: float,
                 label: str | None,
                 sources: list[VertexItem], targets: list[VertexItem],
                 identifier: int,
                 identity: bool = False) -> None:
        super().__init__(0, 0, width, height)
        self.setPos(x, y)
        self.setBrush(QBrush(QColor('#6DB9EF')))
        self.height = height
        self.sources = sources
        self.targets = targets
        self.identity = identity
        self.label = label
        self.identifier = identifier 

    def paint(self, painter: QPainter, option, widget=None) -> None:
        super().paint(painter, option, widget)
        painter.drawText(self.boundingRect(),
                         Qt.AlignmentFlag.AlignCenter,
                         self.label)  # type: ignore


class WireItem(QGraphicsPathItem):
    """Wire graphics item."""

    def __init__(self, source: EdgeItem | VertexItem,
                 target: EdgeItem | VertexItem,
                 x_shift: float, y_shift: float) -> None:
        super().__init__()
        if not (
            isinstance(source, VertexItem) and isinstance(target, EdgeItem)
            or isinstance(source, EdgeItem) and isinstance(target, VertexItem)
        ):
            raise ValueError(
                'Wire must have a vertex at one end and an edge at the other.'
            )
        self.source = source
        self.target = target
        self.x_shift = x_shift
        self.y_shift = y_shift
        self.refresh()

    def calculate_path(self) -> QPainterPath:
        """Calculate the cubic bezier curve to draw this edge."""
        vertex_to_edge = isinstance(self.source, VertexItem)
        vertex_item = self.source if vertex_to_edge else self.target
        edge_item = self.target if vertex_to_edge else self.source
        if not isinstance(edge_item, EdgeItem):
            raise ValueError('Source/target type incorrect')

        if vertex_to_edge:
            start_x, start_y = vertex_item.pos().x(), vertex_item.pos().y()
            end_x = edge_item.pos().x()
            end_y = edge_item.pos().y() + self.y_shift
            dx = abs(start_x - end_x)
        else:
            start_x = edge_item.pos().x() + self.x_shift
            start_y = edge_item.pos().y() + self.y_shift
            end_x, end_y = vertex_item.pos().x(), vertex_item.pos().y()
            dx = abs(start_x - end_x)

        # Create the Path object for the cubic Bezier curve
        path = QPainterPath()
        path.moveTo(start_x, start_y)  # start point
        path.cubicTo(start_x + dx * 0.4, start_y,  # control point 1
                     end_x - dx * 0.4, end_y,  # control point 2
                     end_x, end_y)  # end point
        return path

    def refresh(self) -> None:
        path = self.calculate_path()
        self.setPath(path)


class GraphScene(QGraphicsScene):
    """Graphics Scene for Hypergraphs."""

    def __init__(self, x_scale: float = 50.0, y_scale: float = 50.0,
                 vertex_radius: float = 0.125, box_width: float = 1,
                 box_height: float = 1) -> None:
        super().__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.vertex_radius = vertex_radius
        self.box_width = box_width
        self.box_height = box_height

        self.drag_start = QPointF(0, 0)
        self.drag_items = []

        self.setBackgroundBrush(QBrush(QColor('white')))

        self.edges: dict[int, EdgeItem] = {}
        self.vertices: dict[int, VertexItem] = {}

    def set_graph(self, graph: BaseHypergraph) -> None:
        """Set the graph displayed in the graphics scene."""
        self.draw_info = HypergraphDrawInfo(graph)
        self.clear()
        self.place_items()

    def add_vertex(self, vertex: Vertex) -> int:
        """Add a vertex to the draw info and add to the scene."""
        vertex_id = self.draw_info.add_vertex(vertex)
        self.place_vertex(vertex_id)
        return vertex_id

    def add_edge(self, edge: Hyperedge) -> int:
        """Add an edge to the draw info and add to the scene."""
        edge_id = self.draw_info.add_edge(edge)
        self.place_edge(edge_id)
        return edge_id

    def place_vertex(self, vertex_id) -> None:
        """Place a vertex from the draw info into the scene."""
        vertex_draw_info = self.draw_info.vertices[vertex_id]
        x = vertex_draw_info.x * self.x_scale
        y = vertex_draw_info.y * self.y_scale
        radius = self.vertex_radius * (self.x_scale + self.y_scale) / 2
        label = vertex_draw_info.label
        vertex_item = VertexItem(x, y, radius, label, vertex_id)
        self.vertices[vertex_id] = vertex_item
        self.addItem(vertex_item)
        if hasattr(vertex_item, 'textItem'):
            self.addItem(vertex_item.textItem)

    def place_vertices(self) -> None:
        """Place all vertices from the draw info into the scene."""
        for vertex_id, vertex_draw_info in self.draw_info.vertices.items():
            x = vertex_draw_info.x * self.x_scale
            y = vertex_draw_info.y * self.y_scale
            radius = self.vertex_radius * (self.x_scale + self.y_scale) / 2
            label = vertex_draw_info.label
            vertex_item = VertexItem(x, y, radius, label, vertex_id)
            self.vertices[vertex_id] = vertex_item
            self.addItem(vertex_item)
            if hasattr(vertex_item, 'textItem'):
                self.addItem(vertex_item.textItem)

    def place_edge(self, edge_id) -> None:
        """Place an edge from the draw info into the scene."""
        edge_draw_info = self.draw_info.edges[edge_id]
        width = self.x_scale * self.box_width
        height = self.y_scale * self.box_height
        if edge_draw_info.identity:
            width /= 10
            height /= 10

        x_shift = width / 2
        y_shift = height / 2

        x = edge_draw_info.x * self.x_scale - x_shift
        y = edge_draw_info.y * self.y_scale - y_shift

        edge = self.draw_info.graph.edges[edge_id]
        sources = [self.vertices[vertex_id] for vertex_id in edge.sources]
        targets = [self.vertices[vertex_id] for vertex_id in edge.targets]

        edge_item = EdgeItem(x, y, width, height,
                             edge_draw_info.label,
                             sources, targets,
                             edge_id,
                             edge_draw_info.identity)
        self.edges[edge_id] = edge_item
        self.addItem(edge_item)

    def place_edges(self) -> None:
        """Place all hyperedges from the draw info into the scene."""
        for edge_id, edge_draw_info in self.draw_info.edges.items():
            width = self.x_scale * self.box_width
            height = self.y_scale * self.box_height
            if edge_draw_info.identity:
                width /= 10
                height /= 10

            x_shift = width / 2
            y_shift = height / 2

            x = edge_draw_info.x * self.x_scale - x_shift
            y = edge_draw_info.y * self.y_scale - y_shift

            edge = self.draw_info.graph.edges[edge_id]
            sources = [self.vertices[vertex_id] for vertex_id in edge.sources]
            targets = [self.vertices[vertex_id] for vertex_id in edge.targets]

            edge_item = EdgeItem(x, y, width, height,
                                 edge_draw_info.label,
                                 sources, targets,
                                 edge_id,
                                 edge_draw_info.identity)
            self.edges[edge_id] = edge_item
            self.addItem(edge_item)

    def place_wires_for_edge(self, edge_id: int) -> None:
        """Place all wires for an edge in the graph scene."""
        edge_item = self.edges[edge_id]

        x_shift = 0 if edge_item.identity else self.box_width * self.x_scale

        for vertex_items, vertex_to_edge in ((edge_item.sources, True),
                                             (edge_item.targets, False)):
            num_ports = len(vertex_items)
            for i, vertex_item in enumerate(vertex_items):
                if num_ports == 1:
                    y_shift = edge_item.height / 2
                else:
                    # a bit of playing around with scaling
                    # and numbers to get lines straight
                    y_shift = edge_item.height * (1 / 1.6) * (
                        i / (num_ports - 1) + 0.3
                    )

                source: EdgeItem | VertexItem = (vertex_item if vertex_to_edge
                                                 else edge_item)
                target: EdgeItem | VertexItem = (edge_item if vertex_to_edge
                                                 else vertex_item)
                self.addItem(WireItem(source, target, x_shift, y_shift))

    def place_wires(self) -> None:
        """Place all wires from the draw info in the graph scene."""
        # Given that every wire has an edge at one of its end points, we are
        # guaranteed to find all the wires by iterating over all the edges
        for edge_id in self.draw_info.edges:
            self.place_wires_for_edge(edge_id)

    def place_items(self) -> None:
        """Place graphics items from the draw info into the scene."""
        self.place_vertices()
        self.place_edges()
        self.place_wires()
        self.setSceneRect(self.sceneRect().x() - self.x_scale,
                          self.sceneRect().y() - self.y_scale,
                          self.sceneRect().width() + self.x_scale * 1.5,
                          self.sceneRect().height() + self.y_scale * 1.5)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Handle mouse press for dragging items."""
        self.drag_start = event.scenePos()
        for item in self.items(event.scenePos(), deviceTransform=QTransform()):
            if item and (isinstance(item, EdgeItem)
                         or isinstance(item, VertexItem)):
                self.drag_items = [(item, item.scenePos())]
                break

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Handle mouse move for dragging items."""
        pos = event.scenePos()
        grid_size_x = self.x_scale / 25
        grid_size_y = self.y_scale / 25
        dx = round((pos.x() - self.drag_start.x()) / grid_size_x) * grid_size_x
        dy = round((pos.y() - self.drag_start.y()) / grid_size_y) * grid_size_y

        for item, pos in self.drag_items:
            new_x = pos.x() + dx
            new_y = pos.y() + dy
            if isinstance(item, VertexItem):
                draw_info = self.draw_info.vertices[item.identifier]
            elif isinstance(item, EdgeItem):
                draw_info = self.draw_info.edges[item.identifier]
            else:
                continue
            draw_info.x = new_x
            draw_info.y = new_y
            item.setPos(QPointF(pos.x() + dx, pos.y() + dy))

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Clear all items being dragged when mouse is released."""
        self.drag_items = []
