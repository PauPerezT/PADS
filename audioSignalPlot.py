from PySide6 import QtCore
from PySide6.QtWidgets import QWidget, QLabel, QHBoxLayout, QPushButton, QSizePolicy, QGridLayout
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


# Audio signal plot in time domain
class AudioSignalPlot(QWidget):
    def __init__(self, audio_signal_time_ms, audio_signal_amplitude, plot_color):
        super().__init__()
        self.audio_signal_time_ms = audio_signal_time_ms
        self.audio_signal_amplitude = audio_signal_amplitude
        self.plot_color = plot_color

        self.setup_audio_signal_plot()

    def setup_audio_signal_plot(self):
        figure_audio_signal = plt.figure(figsize=(6, 2))
        plt.plot(self.audio_signal_time_ms, self.audio_signal_amplitude, color=self.plot_color)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.tight_layout()

        self.canvas = FigureCanvas(figure_audio_signal)
