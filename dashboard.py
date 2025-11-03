from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, \
    QSizePolicy, QGridLayout, QLabel, QFileDialog, QStackedLayout, QLineEdit
from PySide6 import QtCore
from PySide6.QtCore import QStringListModel, QRegularExpression
from PySide6.QtGui import QIntValidator, QFont, QRegularExpressionValidator
from screen import Screen
from radarPlot import RadarPlot
from barPlot import BarPlot
from emotionAnalysisSection import EmotionAnalysisSection
import soundfile as sf
from os import getcwd
from audioRecorder import record_audio
from audioSignalPlot import AudioSignalPlot
from getEmotions import *
from matplotlib import pyplot as plt
import math

# Main Window of demo dashboard
class Dashboard(Screen):
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.setWindowTitle("Speech Emotions Recognition Demo")
        self.grid_layout = QGridLayout()
        self.main_widget.setLayout(self.grid_layout)

        self.complete_audio_signal_base = [0]
        self.complete_audio_signal_ctrl = [0]

        self.complete_audio_file_path_base = ""
        self.complete_audio_file_path_ctrl = ""

        self.sampling_freq_base = 16000
        self.sampling_freq_ctrl = 16000

        self.button_record_audio = QPushButton()

        self.line_edit_moving_window_start_time = QLineEdit(parent=self, )
        self.line_edit_moving_window_end_time = QLineEdit(parent=self, )

        self.label_instruction_text_start_time = QLabel()
        self.label_instruction_text_end_time = QLabel()
        self.input_instruction_text = "Please enter only integer numbers (seconds)"

        self.window_size = 500

        self.entered_start_time = 0
        self.entered_end_time = 0
        self.entered_start_frame_index = 0
        self.entered_end_frame_index = 0

        self.setup_audio_grid()
        self.setup_input_field()
        self.setup_more_stats()
        self.setup_emotions_plots(None)
        self.setup_emotions_analysis(self.complete_audio_signal_base, self.complete_audio_signal_ctrl)

    # Change text of record audio button when recording
    def update_record_button(self, is_recording):
        if is_recording:
            self.button_record_audio.setText("Recording")
        else:
            self.button_record_audio.setText("Record Audio")

    # Setup load base audio, load control audio, record control audio and signal time plots
    def setup_audio_grid(self):
        button_load_audio_base = QPushButton()
        button_load_audio_base.setText("Load Base Audio File")
        button_load_audio_base.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        button_load_audio_base.clicked.connect(self.button_load_audio_base_clicked)

        button_load_audio_ctrl = QPushButton()
        button_load_audio_ctrl.setText("Load Control Audio File")
        button_load_audio_ctrl.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        button_load_audio_ctrl.clicked.connect(self.button_load_audio_ctrl_clicked)

        self.update_record_button(False)
        self.button_record_audio.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.button_record_audio.clicked.connect(self.button_record_audio_clicked)

        self.grid_layout.addWidget(button_load_audio_base, 0, 0)

        buttons_audio_ctrl = QHBoxLayout()
        buttons_audio_ctrl.addWidget(button_load_audio_ctrl)
        buttons_audio_ctrl.addWidget(self.button_record_audio)

        self.grid_layout.addLayout(buttons_audio_ctrl, 0, 1)

        widget_plot_audio_base = AudioSignalPlot(self.complete_audio_signal_base, self.complete_audio_signal_ctrl,
                                                 'blue')
        self.grid_layout.addWidget(widget_plot_audio_base.canvas, 1, 0)

        widget_plot_audio_ctrl = AudioSignalPlot(self.complete_audio_signal_base, self.complete_audio_signal_ctrl,
                                                 'indianred')
        self.grid_layout.addWidget(widget_plot_audio_ctrl.canvas, 1, 1)

    # Setup input fields and hint texts for start time and end time
    def setup_input_field(self):
        label_moving_window_start_time = QLabel("Moving Window Start Time (seconds):")
        self.line_edit_moving_window_start_time.setValidator(QRegularExpressionValidator(QRegularExpression("[0-9]+")))
        self.line_edit_moving_window_start_time.textChanged.connect(self.start_time_changed)

        self.label_instruction_text_start_time.setStyleSheet('color: red')
        font_instruction_text = self.font()
        font_instruction_text.setPointSize(12)
        self.label_instruction_text_start_time.setFont(font_instruction_text)

        label_moving_window_end_time = QLabel("Moving Window End Time (seconds):")
        self.line_edit_moving_window_end_time.setValidator(QRegularExpressionValidator(QRegularExpression("[0-9]+")))
        self.line_edit_moving_window_end_time.textChanged.connect(self.end_time_changed)

        self.label_instruction_text_end_time.setStyleSheet('color: red')
        self.label_instruction_text_end_time.setFont(font_instruction_text)

        grid_layout_input_field = QGridLayout()
        grid_layout_input_field.addWidget(label_moving_window_start_time, 0, 0)
        grid_layout_input_field.addWidget(self.line_edit_moving_window_start_time, 1, 0)
        grid_layout_input_field.addWidget(self.label_instruction_text_start_time, 2, 0)
        self.label_instruction_text_start_time.setText(self.input_instruction_text)

        grid_layout_input_field.addWidget(label_moving_window_end_time, 3, 0)
        grid_layout_input_field.addWidget(self.line_edit_moving_window_end_time, 4, 0)
        grid_layout_input_field.addWidget(self.label_instruction_text_end_time, 5, 0)
        self.label_instruction_text_end_time.setText(self.input_instruction_text)

        self.grid_layout.addLayout(grid_layout_input_field, 0, 2)

    # Event start time changes: Update hint text and dashboard plots
    def start_time_changed(self):
        if not self.line_edit_moving_window_start_time.text():
            input_start_time = 0
            self.label_instruction_text_start_time.setText(self.input_instruction_text)
        else:
            input_start_time = int(float(self.line_edit_moving_window_start_time.text()))
            self.label_instruction_text_start_time.setText("")

        self.entered_start_time = input_start_time
        print("Entered window start time by user (s):", self.entered_start_time)

        self.update_plots()

    # Event end time changes: Update hint text and dashboard plots
    def end_time_changed(self):
        if not self.line_edit_moving_window_end_time.text():
            input_end_time = 0
            self.label_instruction_text_end_time.setText(self.input_instruction_text)
        else:
            input_end_time = int(float(self.line_edit_moving_window_end_time.text()))
            self.label_instruction_text_end_time.setText("")

        self.entered_end_time = input_end_time
        print("Entered window end time by user (s):", self.entered_end_time)

        self.update_plots()

    # Select audio section of the entire loaded/recorded signal according to entered start and end times
    def select_audio_section(self, complete_audio_signal_amplitude, sampling_freq):
        if self.entered_end_time == 0 and self.entered_start_time == 0:
            selected_audio_section = complete_audio_signal_amplitude
        elif self.entered_end_time <= self.entered_start_time:
            selected_audio_section = [0]
            print("Invalid entered end time:", self.entered_end_time)
            print("Because it is less than or equal to start time:", self.entered_start_time)
        elif self.entered_end_time > len(complete_audio_signal_amplitude) / sampling_freq:
            selected_audio_section = complete_audio_signal_amplitude[int(self.entered_start_time * sampling_freq):]
        else:
            selected_audio_section = complete_audio_signal_amplitude[int(self.entered_start_time
                                                                         * sampling_freq):int(
                self.entered_end_time * sampling_freq) + 1]
            print("Selected audio section from", self.entered_start_time, "to",
                  self.entered_end_time, "seconds")
            print("Selected audio section from index", int(self.entered_start_time * sampling_freq), "to",
                  int(self.entered_end_time * sampling_freq))

        return selected_audio_section

    # Open base audio file and plot audio signal in time domain
    def button_load_audio_base_clicked(self):
        selected_audio_file = open_audio_file()
        self.complete_audio_file_path_base = selected_audio_file
        audio_signal, self.sampling_freq_base = sf.read(selected_audio_file)

        print("Loaded base audio signal sampling frequency:", self.sampling_freq_base)

        self.complete_audio_signal_base = audio_signal.tolist()
        selected_audio_signal_amplitude = self.select_audio_section(self.complete_audio_signal_base,
                                                                    self.sampling_freq_base)
        audio_signal_index = [i for i, _ in enumerate(selected_audio_signal_amplitude)]
        audio_signal_time = [j / self.sampling_freq_base for j in audio_signal_index]
        audio_signal_time_axis = [x + self.entered_start_time for x in audio_signal_time]
        print("x-axis starts from:", audio_signal_time_axis[0])
        widget_plot_audio_base = AudioSignalPlot(audio_signal_time_axis, selected_audio_signal_amplitude, 'blue')
        self.grid_layout.addWidget(widget_plot_audio_base.canvas, 1, 0)

        self.setup_emotions_plots(audio_signal_time_axis)

    # Open control audio file and plot audio signal in time domain
    def button_load_audio_ctrl_clicked(self):
        selected_audio_file = open_audio_file()
        self.complete_audio_file_path_ctrl = selected_audio_file
        audio_signal, self.sampling_freq_ctrl = sf.read(selected_audio_file)

        print("Loaded control audio signal sampling frequency:", self.sampling_freq_ctrl)

        self.complete_audio_signal_ctrl = audio_signal.tolist()
        selected_audio_signal_amplitude = self.select_audio_section(self.complete_audio_signal_ctrl,
                                                                    self.sampling_freq_ctrl)
        audio_signal_index = [i for i, _ in enumerate(selected_audio_signal_amplitude)]
        audio_signal_time = [j / self.sampling_freq_ctrl for j in audio_signal_index]
        audio_signal_time_axis = [x + self.entered_start_time for x in audio_signal_time]
        print("x-axis starts from:", audio_signal_time_axis[0])
        widget_plot_audio_ctrl = AudioSignalPlot(audio_signal_time_axis, selected_audio_signal_amplitude, 'indianred')

        self.grid_layout.addWidget(widget_plot_audio_ctrl.canvas, 1, 1)

        self.setup_emotions_plots(audio_signal_time_axis)

    # Record control audio signal and then save it and plot it in time domain
    def button_record_audio_clicked(self):
        self.update_record_button(True)
        recorded_audio_file = record_audio(self.sampling_freq_ctrl)
        self.complete_audio_file_path_ctrl = recorded_audio_file
        self.update_record_button(False)

        audio_signal, self.sampling_freq_ctrl = sf.read(recorded_audio_file)

        print("Recorded control audio signal sampling frequency:", self.sampling_freq_ctrl)

        self.complete_audio_signal_ctrl = audio_signal.tolist()
        selected_audio_signal_amplitude = self.select_audio_section(self.complete_audio_signal_ctrl,
                                                                    self.sampling_freq_ctrl)
        audio_signal_index = [i for i, _ in enumerate(selected_audio_signal_amplitude)]
        audio_signal_time = [j / self.sampling_freq_ctrl for j in audio_signal_index]
        audio_signal_time_axis = [x + self.entered_start_time for x in audio_signal_time]
        print("x-axis starts from:", audio_signal_time_axis[0])
        widget_plot_audio_ctrl = AudioSignalPlot(audio_signal_time_axis, selected_audio_signal_amplitude, 'indianred')

        self.grid_layout.addWidget(widget_plot_audio_ctrl.canvas, 1, 1)

        self.setup_emotions_plots(audio_signal_time_axis)

    # Setup button for predicted emotions probabilities across the entire base and control audio signals
    def setup_more_stats(self):
        grid_predicted_emotion_results = QGridLayout()

        label_more_stats = QLabel("More Statistics:")
        label_more_stats.setAlignment(QtCore.Qt.AlignCenter)
        font_more_stats_text = self.font()
        font_more_stats_text.setPointSize(15)
        font_more_stats_text.setBold(True)
        label_more_stats.setFont(font_more_stats_text)

        button_signal_analysis = QPushButton()
        button_signal_analysis.setText("Predicted Emotions Probabilities")
        button_signal_analysis.setFixedSize(QtCore.QSize(220, 150))
        button_signal_analysis.clicked.connect(self.button_signal_analysis_clicked)

        grid_predicted_emotion_results.addWidget(label_more_stats, 0, 0)
        grid_predicted_emotion_results.addWidget(button_signal_analysis, 1, 0)

        self.grid_layout.addLayout(grid_predicted_emotion_results, 1, 2)

    # Setup radar plot and bar plot of both base and control audio signals
    def setup_emotions_plots(self, audio_signal_time_array):
        if audio_signal_time_array == None:
            posteriros_start_index = 0
            posteriros_end_index = 0
        else:
            posteriros_start_index, posteriros_end_index = match_posteriros_to_time(audio_signal_time_array)
        
        bar_plot = get_bar_plot(self.complete_audio_file_path_base, self.complete_audio_file_path_ctrl, 
        posteriros_start_index, posteriros_end_index)
        radar_plot = get_radar_plot(self.complete_audio_file_path_base, self.complete_audio_file_path_ctrl, 
        posteriros_start_index, posteriros_end_index)
        
        self.grid_layout.addWidget(bar_plot.browser, 2, 0)
        self.grid_layout.addWidget(radar_plot.browser, 2, 1)

    # Setup emotion analysis final result of selected audio section in form of smiley and result text
    def setup_emotions_analysis(self, audio_signal_base, audio_signal_ctrl):
        emotion_analysis_section = EmotionAnalysisSection(audio_signal_base, audio_signal_ctrl)
        self.grid_layout.addLayout(emotion_analysis_section, 2, 2)

    # Show figure window for plotting predicted emotions probabilities across the entire base and control audio signals
    def button_signal_analysis_clicked(self):
        figure, axes = plt.subplots(2)
        figure.suptitle('Predicted Emotions Probabilities')

        if self.complete_audio_file_path_base:
            active_arousal_posteriors_base = get_arousal_active_posteriors(self.complete_audio_file_path_base)
            strong_dominance_posteriors_base = get_dominance_strong_posteriors(self.complete_audio_file_path_base)
            positive_valence_posteriors_base = get_valence_positive_posteriors(self.complete_audio_file_path_base)

            base_audio_signal_index = [i for i, _ in enumerate(self.complete_audio_signal_base)]
            base_audio_signal_adjusted_signal_axis = match_time_axis_to_posteriors(base_audio_signal_index, active_arousal_posteriors_base)
            base_audio_signal_time_axis_seconds = [j / self.sampling_freq_base for j in base_audio_signal_adjusted_signal_axis]

            axes[0].plot(base_audio_signal_time_axis_seconds, active_arousal_posteriors_base, color="red",
                         label="Active Arousal")
            axes[0].plot(base_audio_signal_time_axis_seconds, strong_dominance_posteriors_base,
                         color="black", label="Strong Dominance")
            axes[0].plot(base_audio_signal_time_axis_seconds, positive_valence_posteriors_base, color="blue",
                         label="Positive Valence")
            axes[0].set(xlabel="Time (s)", ylabel="Probability")
            axes[0].set(title="Base Signal")

            passive_arousal_posteriors_base = get_arousal_passive_posteriors(self.complete_audio_file_path_base)
            weak_dominance_posteriors_base = get_dominance_weak_posteriors(self.complete_audio_file_path_base)
            negative_valence_posteriors_base = get_valence_negative_posteriors(self.complete_audio_file_path_base)

            axes[0].plot(base_audio_signal_time_axis_seconds, passive_arousal_posteriors_base, color="red", linestyle='dotted',
                         label="Passive Arousal")
            axes[0].plot(base_audio_signal_time_axis_seconds, weak_dominance_posteriors_base,
                         color="black", linestyle='dotted', label="Weak Dominance")
            axes[0].plot(base_audio_signal_time_axis_seconds, negative_valence_posteriors_base, color="blue", linestyle='dotted',
                         label="Negative Valence")
            axes[0].legend(loc="upper right")

        if self.complete_audio_file_path_ctrl:
            active_arousal_posteriors_ctrl = get_arousal_active_posteriors(self.complete_audio_file_path_ctrl)
            strong_dominance_posteriors_ctrl = get_dominance_strong_posteriors(self.complete_audio_file_path_ctrl)
            positive_valence_posteriors_ctrl = get_valence_positive_posteriors(self.complete_audio_file_path_ctrl)
            
            ctrl_audio_signal_index = [i for i, _ in enumerate(self.complete_audio_signal_ctrl)]
            ctrl_audio_signal_adjusted_signal_axis = match_time_axis_to_posteriors(ctrl_audio_signal_index, active_arousal_posteriors_ctrl)
            ctrl_audio_signal_time_axis_seconds = [j / self.sampling_freq_ctrl for j in ctrl_audio_signal_adjusted_signal_axis]

            axes[1].plot(ctrl_audio_signal_time_axis_seconds, active_arousal_posteriors_ctrl, color="red",
                         label="Active Arousal")
            axes[1].plot(ctrl_audio_signal_time_axis_seconds, strong_dominance_posteriors_ctrl,
                         color="black", label="Strong Dominance")
            axes[1].plot(ctrl_audio_signal_time_axis_seconds,positive_valence_posteriors_ctrl, color="blue",
                         label="Positive Valence")
            axes[1].set(xlabel="Time (s)", ylabel="Probability")
            axes[1].set(title="Control Signal")

            passive_arousal_posteriors_base = get_arousal_passive_posteriors(self.complete_audio_file_path_ctrl)
            weak_dominance_posteriors_base = get_dominance_weak_posteriors(self.complete_audio_file_path_ctrl)
            negative_valence_posteriors_base = get_valence_negative_posteriors(self.complete_audio_file_path_ctrl)

            axes[1].plot(ctrl_audio_signal_time_axis_seconds, passive_arousal_posteriors_base, color="red", linestyle='dotted',
                         label="Passive Arousal")
            axes[1].plot(ctrl_audio_signal_time_axis_seconds, weak_dominance_posteriors_base,
                         color="black", linestyle='dotted', label="Weak Dominance")
            axes[1].plot(ctrl_audio_signal_time_axis_seconds, negative_valence_posteriors_base, color="blue", linestyle='dotted',
                         label="Negative Valence")
            axes[1].legend(loc="upper right")

        plt.tight_layout()
        figure.show()

    # Update dashboard plots in case of start or end time change by user
    def update_plots(self):
        selected_audio_signal_base = self.select_audio_section(self.complete_audio_signal_base,
                                                               self.sampling_freq_base)
        base_audio_signal_index = [i for i, _ in enumerate(selected_audio_signal_base)]
        base_audio_signal_time = [j / self.sampling_freq_base for j in base_audio_signal_index]
        audio_signal_time_axis = [x + self.entered_start_time for x in base_audio_signal_time]
        widget_plot_audio_base = AudioSignalPlot(audio_signal_time_axis, selected_audio_signal_base, 'blue')
        self.grid_layout.addWidget(widget_plot_audio_base.canvas, 1, 0)

        selected_audio_signal_ctrl = self.select_audio_section(self.complete_audio_signal_ctrl,
                                                               self.sampling_freq_ctrl)
        ctrl_audio_signal_index = [i for i, _ in enumerate(selected_audio_signal_ctrl)]
        ctrl_audio_signal_time = [j / self.sampling_freq_ctrl for j in ctrl_audio_signal_index]
        audio_signal_time_axis = [x + self.entered_start_time for x in ctrl_audio_signal_time]
        widget_plot_audio_ctrl = AudioSignalPlot(audio_signal_time_axis, selected_audio_signal_ctrl, 'indianred')
        self.grid_layout.addWidget(widget_plot_audio_ctrl.canvas, 1, 1)

        self.setup_emotions_plots(audio_signal_time_axis)
        #self.update_emotions_analysis(selected_audio_signal_base, selected_audio_signal_ctrl)

# Getter function for radar plot
def get_radar_plot(audio_file_path_base, audio_file_path_ctrl, posteriros_start_index, posteriros_end_index):
    radar_plot = RadarPlot(audio_file_path_base, audio_file_path_ctrl, posteriros_start_index, posteriros_end_index)
    return radar_plot


# Getter function for bar plot
def get_bar_plot(audio_file_path_base, audio_file_path_ctrl, posteriros_start_index, posteriros_end_index):
    bar_plot = BarPlot(audio_file_path_base, audio_file_path_ctrl, posteriros_start_index, posteriros_end_index)
    return bar_plot


# Open dialog to open the loaded audio file from file system
def open_audio_file():
    dialog_load_audio_file = QFileDialog()
    dialog_load_audio_file.setDirectory(getcwd())
    dialog_load_audio_file.setFileMode(QFileDialog.ExistingFile)
    dialog_load_audio_file.setNameFilter("Audio files (*.wav)")
    string_list_selected_audio_file = QStringListModel()

    if dialog_load_audio_file.exec():
        string_list_selected_audio_file = dialog_load_audio_file.selectedFiles()
        print("Selected Audio File:", string_list_selected_audio_file)

    return string_list_selected_audio_file[0]

def match_time_axis_to_posteriors(time_axis, posteriors):
    time_axis_length = len(time_axis)
    posteriors_length = len(posteriors)
    time_frame_length = time_axis_length//posteriors_length
    time_axis_updated = []
    
    if posteriors_length !=0:
        posteriors_index = [i for i, _ in enumerate(posteriors)]
        time_axis_updated = [i * time_frame_length for i in posteriors_index]
    
    return time_axis_updated

def match_posteriros_to_time(time_array_in_second):
    start_time_sec = math.floor(time_array_in_second[0])
    end_time_sec = math.floor(time_array_in_second[-1])

    sequence_step = 0.5
    posterior_start_index = int(start_time_sec / sequence_step)
    posterior_end_index = int(end_time_sec / sequence_step)

    return posterior_start_index, posterior_end_index
