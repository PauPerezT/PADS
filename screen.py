from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QSizePolicy
from PySide6.QtCore import QSize, Qt


# Main Window of demo dashboard
class Screen(QMainWindow):

    def __init__(self, app):
        super().__init__()
        self.app = app

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
