import sys

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QLabel, QMainWindow, QPlainTextEdit, QVBoxLayout, QWidget
)


# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    """Main window of HypNN."""
    def __init__(self):
        super().__init__()

        self.setWindowTitle('HypNN')

        layout = QVBoxLayout()
        layout.addWidget(QLabel('Graph display goes here.'))
        layout.addWidget(QPlainTextEdit())

        widget = QWidget()
        widget.setLayout(layout)

        self.setCentralWidget(widget)

        self.show()
