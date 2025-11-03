from PySide6 import QtCore
from PySide6.QtWidgets import QWidget, QLabel, QHBoxLayout, QPushButton, QSizePolicy, QGridLayout
import plotly.graph_objects as graph
from PySide6 import QtWebEngineWidgets
from getEmotions import *


# Radar plot of predicted emotions of audio signals
class RadarPlot(QWidget):
    def __init__(self, audio_file_path_base, audio_file_path_ctrl, posteriros_start_index, posteriros_end_index):
        super().__init__()
        self.audio_file_path_base = audio_file_path_base
        self.audio_file_path_ctrl = audio_file_path_ctrl
        self.posteriros_start_index= posteriros_start_index
        self.posteriros_end_index= posteriros_end_index

        self.browser = QtWebEngineWidgets.QWebEngineView(self)
        self.setup_radar_plot()

    def setup_radar_plot(self):
        emotions = ['Active Arousal', 'Passive Arousal', 'Strong Dominance', 'Weak Dominance', 'Positive Valence', 'Negative Valence']
        base_signal_emotions_prob = []
        ctrl_signal_emotions_prob = []

        if  self.audio_file_path_base:
            base_signal_emotions_prob = [get_arousal_active_posterior_mean(self.audio_file_path_base, self.posteriros_start_index, self.posteriros_end_index),   
            get_arousal_passive_posterior_mean(self.audio_file_path_base, self.posteriros_start_index, self.posteriros_end_index), 
            get_dominance_strong_posterior_mean(self.audio_file_path_base, self.posteriros_start_index, self.posteriros_end_index),
            get_dominance_weak_posterior_mean(self.audio_file_path_base, self.posteriros_start_index, self.posteriros_end_index),
            get_valence_positive_posterior_mean(self.audio_file_path_base, self.posteriros_start_index, self.posteriros_end_index),
            get_valence_negative_posterior_mean(self.audio_file_path_base, self.posteriros_start_index, self.posteriros_end_index)]

        if  self.audio_file_path_ctrl:
            ctrl_signal_emotions_prob = [get_arousal_active_posterior_mean(self.audio_file_path_ctrl, self.posteriros_start_index, self.posteriros_end_index),   
            get_arousal_passive_posterior_mean(self.audio_file_path_ctrl, self.posteriros_start_index, self.posteriros_end_index), 
            get_dominance_strong_posterior_mean(self.audio_file_path_ctrl, self.posteriros_start_index, self.posteriros_end_index),
            get_dominance_weak_posterior_mean(self.audio_file_path_ctrl, self.posteriros_start_index, self.posteriros_end_index),
            get_valence_positive_posterior_mean(self.audio_file_path_ctrl, self.posteriros_start_index, self.posteriros_end_index),
            get_valence_negative_posterior_mean(self.audio_file_path_ctrl, self.posteriros_start_index, self.posteriros_end_index)]

        fig = graph.Figure()

        fig.add_trace(graph.Scatterpolar(
            r=base_signal_emotions_prob,
            theta=emotions,
            fill='toself',
            name='Base Signal'
        ))
        fig.add_trace(graph.Scatterpolar(
            r=ctrl_signal_emotions_prob,
            theta=emotions,
            fill='toself',
            name='Control Signal'
        ))

        fig.update_layout(
            title={
                'text': "Emotion Radar Plot",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            autosize=True
        )

        fig.update_polars(gridshape='linear')

        self.browser.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Expanding)
        self.browser.setHtml(fig.to_html(include_plotlyjs='cdn'))
