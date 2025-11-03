from PySide6 import QtCore
from PySide6.QtWidgets import QWidget, QLabel, QHBoxLayout, QPushButton, QSizePolicy, QGridLayout
import plotly.graph_objects as graph
from PySide6 import QtWebEngineWidgets
from getEmotions import *


# Bar plot of predicted emotions of audio signals
class BarPlot(QWidget):
    def __init__(self, audio_file_path_base, audio_file_path_ctrl, posteriros_start_index, posteriros_end_index):
        super().__init__()
        self.audio_file_path_base = audio_file_path_base
        self.audio_file_path_ctrl = audio_file_path_ctrl
        self.posteriros_start_index= posteriros_start_index
        self.posteriros_end_index= posteriros_end_index

        self.browser = QtWebEngineWidgets.QWebEngineView(self)
        self.setup_bar_plots()

    def setup_bar_plots(self):
        emotions = ['Arousal', 'Dominance', 'Valence']

        base_signal_emotions = []
        base_signal_color = "blue"

        ctrl_signal_emotions = []
        ctrl_signal_color = "indianred"

        if  self.audio_file_path_base:
            base_signal_emotions = [get_arousal_active_posterior_mean(self.audio_file_path_base, self.posteriros_start_index, self.posteriros_end_index),
                                    get_dominance_strong_posterior_mean(self.audio_file_path_base, self.posteriros_start_index, self.posteriros_end_index),
                                    get_valence_positive_posterior_mean(self.audio_file_path_base, self.posteriros_start_index, self.posteriros_end_index)]
        
        if  self.audio_file_path_ctrl:
            ctrl_signal_emotions = [get_arousal_active_posterior_mean(self.audio_file_path_ctrl, self.posteriros_start_index, self.posteriros_end_index),
                                    get_dominance_strong_posterior_mean(self.audio_file_path_ctrl, self.posteriros_start_index, self.posteriros_end_index),
                                    get_valence_positive_posterior_mean(self.audio_file_path_ctrl, self.posteriros_start_index, self.posteriros_end_index)]

        fig = graph.Figure()

        fig.add_trace(graph.Bar(
            name="Base Signal",
            x=emotions,
            y=base_signal_emotions,
            marker_color=base_signal_color
        ))

        fig.add_trace(graph.Bar(
            name="Control Signal",
            x=emotions,
            y=ctrl_signal_emotions,
            marker_color=ctrl_signal_color
        ))

        fig.update_layout(
            title={
                'text': "Emotion Bar Plot",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            yaxis_range=[0, 100],
            showlegend=True
        )

        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='grey', griddash="dash", dtick=33, nticks=2)

        self.browser.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Expanding)
        self.browser.setHtml(fig.to_html(include_plotlyjs='cdn'))
