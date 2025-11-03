import soundfile as sf
import sounddevice as sd

import numpy
from scipy.io.wavfile import write

from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from os import getcwd

from PySide6.QtWidgets import QFileDialog
from PySide6.QtCore import QStringListModel


# Record and save audio file with the given sampling frequency
def record_audio(sampling_freq):
    time_seconds = 5

    sd.default.samplerate = sampling_freq
    sd.default.channels = 1

    recorded_audio_signal = sd.rec(int(time_seconds * sampling_freq))
    print("Starting audio recording")

    sd.wait()

    saved_audio_file = save_audio_file()

    print("Finished the recording:", saved_audio_file)

    write(saved_audio_file, sampling_freq, recorded_audio_signal)

    return saved_audio_file


# Open dialog to save the recorded audio signal in file system
def save_audio_file():
    dialog_save_audio_file = QFileDialog()
    dialog_save_audio_file.setDirectory(getcwd())
    dialog_save_audio_file.setFileMode(QFileDialog.AnyFile)
    string_list_saved_audio_file = dialog_save_audio_file.getSaveFileName(None, "Save File", "", "Audio files (*.wav)")

    return string_list_saved_audio_file[0]
