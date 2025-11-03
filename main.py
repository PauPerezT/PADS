import sys
from PySide6.QtWidgets import QApplication
from dashboard import Dashboard

app = QApplication(sys.argv)

# Create demo dashboard
home_screen = Dashboard(app)
home_screen.show()
home_screen.showFullScreen()

app.exec()
