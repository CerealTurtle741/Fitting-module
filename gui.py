import sys
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QApplication, QMainWindow,
                               QWidget, QGridLayout, QMenu, QMenuBar)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class Gui(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('Gui')

        container: QWidget = QWidget()
        layout: QGridLayout = QGridLayout(container)
        self.setCentralWidget(container)

        menubar: QMenuBar = self.menuBar()

        # Menu bar
        file_menu: QMenu = menubar.addMenu('File')

        # File menu
        exit_action: QAction = file_menu.addAction('Exit')
        exit_action.triggered.connect(self.close)

        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)

        self.ax = self.figure.add_subplot(111)

        self.canvas.draw()

        layout.addWidget(self.canvas)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Gui()
    window.show()
    app.exec()
