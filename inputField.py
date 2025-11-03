from PySide6.QtWidgets import QWidget, QLineEdit, QLabel, QGridLayout
from PySide6.QtGui import QIntValidator, QFont, QRegularExpressionValidator
from PySide6.QtCore import QRegularExpression


# Input fields and hint texts for start time and end time
class InputField(QWidget):
    def __init__(self):
        super().__init__()
        self.line_edit_moving_window_start_time = QLineEdit(parent=self, )
        self.line_edit_moving_window_end_time = QLineEdit(parent=self, )

        self.label_instruction_text_start_time = QLabel()
        self.label_instruction_text_end_time = QLabel()

        self.window_size = 500

        self.entered_start_time_ms = 0
        self.entered_end_time_ms = 0
        self.entered_start_frame_index = 0
        self.entered_end_frame_index = 0

        self.setup_input_field()

    def setup_input_field(self):
        label_moving_window_start_time = QLabel("Moving Window Start Time (seconds):")
        self.line_edit_moving_window_start_time.setValidator(QRegularExpressionValidator(QRegularExpression("[0-9]+")))
        self.line_edit_moving_window_start_time.textChanged.connect(self.start_time_changed)

        self.label_instruction_text_start_time.setStyleSheet('color: red')
        font_instruction_text = self.font()
        font_instruction_text.setPointSize(12)
        self.label_instruction_text_start_time.setFont(font_instruction_text)
        instruction_text = "Please enter only integer numbers (seconds)"

        label_moving_window_end_time = QLabel("Moving Window End Time (seconds):")
        self.line_edit_moving_window_end_time.setValidator(QRegularExpressionValidator(QRegularExpression("[0-9]+")))
        self.line_edit_moving_window_end_time.textChanged.connect(self.end_time_changed)

        self.label_instruction_text_end_time.setStyleSheet('color: red')
        self.label_instruction_text_end_time.setFont(font_instruction_text)

        grid_layout = QGridLayout()
        grid_layout.addWidget(label_moving_window_start_time, 0, 0)
        grid_layout.addWidget(self.line_edit_moving_window_start_time, 1, 0)
        grid_layout.addWidget(self.label_instruction_text_start_time, 2, 0)
        self.label_instruction_text_start_time.setText(instruction_text)

        grid_layout.addWidget(label_moving_window_end_time, 3, 0)
        grid_layout.addWidget(self.line_edit_moving_window_end_time, 4, 0)
        grid_layout.addWidget(self.label_instruction_text_end_time, 5, 0)
        self.label_instruction_text_end_time.setText(instruction_text)

        self.setLayout(grid_layout)

    def start_time_changed(self):
        self.label_instruction_text_start_time.setText("")
        if not self.line_edit_moving_window_start_time.text():
            input_start_time = 0
        else:
            input_start_time = int(float(self.line_edit_moving_window_start_time.text()))

        input_start_time_ms = input_start_time * 1000
        # window_start_time_ms = approximate_window_start(input_start_time_ms)
        self.entered_start_time_ms = input_start_time_ms
        print("Entered window start time by user (ms):", self.entered_start_time_ms)
        self.entered_start_frame_index = input_start_time_ms // 250
        print("Start frame index:", self.entered_start_frame_index)

    def end_time_changed(self):
        self.label_instruction_text_end_time.setText("")
        if not self.line_edit_moving_window_end_time.text():
            input_end_time = 0
        else:
            input_end_time = int(float(self.line_edit_moving_window_end_time.text()))

        input_end_time_ms = input_end_time * 1000
        # window_end_time_ms = approximate_window_end(input_end_time_ms)
        self.entered_end_time_ms = input_end_time_ms
        print("Entered window end time by user:", self.entered_end_time_ms)
        self.entered_end_frame_index = input_end_time_ms // 250
        print("End frame index:", self.entered_end_frame_index)

# def approximate_window_start(input_start_time_ms):
#     if input_start_time_ms % 250 != 0:
#         adjusted_start_time_ms = ((input_start_time_ms % 250) * 250)
#         if adjusted_start_time_ms > input_start_time_ms:
#             adjusted_start_time_ms = adjusted_start_time_ms - self.window_size
#         print("Window adjusted start time (ms):", adjusted_start_time_ms)
#
#
# def approximate_window_end(input_end_time_ms):
#     if input_end_time_ms % 250 != 0:
#         adjusted_end_time_ms = ((input_end_time_ms % 250) * 250)
#         if adjusted_end_time_ms < input_end_time_ms:
#             adjusted_end_time_ms = adjusted_end_time_ms + self.window_size
#
#         print("Window adjusted end time (ms):", adjusted_end_time_ms)
