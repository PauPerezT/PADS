from PySide6.QtWidgets import QWidget, QLabel, QHBoxLayout, QPushButton, QSizePolicy, QGridLayout
from PySide6.QtGui import QPixmap, QFont
from PySide6 import QtCore


# Section for emotion analysis final result of selected audio section in form of smiley and result text
class EmotionAnalysisSection(QGridLayout):
    def __init__(self, audio_signal_base, audio_signal_ctrl):
        super().__init__()
        self.audio_signal_base = audio_signal_base
        self.audio_signal_ctrl = audio_signal_ctrl

        self.label_title = QLabel()
        self.label_smiley = QLabel()
        self.label_result = QLabel()
        self.pixmap_smiley = QPixmap()

        self.setup_emotion_analysis_section()

    def setup_emotion_analysis_section(self):
        if len(self.audio_signal_base) > 1 and len(self.audio_signal_ctrl) > 1:
            font = QFont()
            font.setPointSize(15)
            font.setBold(True)

            self.label_title.setText('Emotions Analysis Result:')
            self.label_title.setFont(font)

            self.label_title.setAlignment(QtCore.Qt.AlignCenter)
            self.addWidget(self.label_title, 1, 0)

            self.pixmap_smiley.load('./Assets/myHappy.jpg')
            self.label_smiley.setPixmap(self.pixmap_smiley)
            self.label_smiley.setAlignment(QtCore.Qt.AlignCenter)
            self.addWidget(self.label_smiley, 2, 0)

            self.label_result.setText('Happy')
            font.setPointSize(30)
            self.label_result.setFont(font)
            self.label_result.setAlignment(QtCore.Qt.AlignCenter)
            self.addWidget(self.label_result, 3, 0)
        else:
            self.label_title.hide()
            self.removeWidget(self.label_title)

            self.label_smiley.hide()
            self.removeWidget(self.label_smiley)

            self.label_result.hide()
            self.removeWidget(self.label_result)
