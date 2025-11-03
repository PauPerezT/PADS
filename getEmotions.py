from random import random
from statistics import mean
import os
import numpy as np
from example_main_lightning import PADS_Models


def get_model_output(audio_file_path):
    model_output = PADS_Models(audio_file_path, cuda=False, sig_set = True)
    return model_output

def get_arousal_active_posteriors(audio_file_path):
    model_output = get_model_output(audio_file_path)
    list_arousal_posteriors = model_output["arousal_post"]
    list_active_arousal_posteriors = np.array(list_arousal_posteriors)[:,1]
    return list_active_arousal_posteriors*100

def get_arousal_active_posterior_mean(audio_file_path, posteriros_start_index, posteriros_end_index):
    arousal_posteriors = get_arousal_active_posteriors(audio_file_path)[posteriros_start_index:posteriros_end_index]
    return mean(arousal_posteriors)

def get_arousal_passive_posteriors(audio_file_path):
    model_output = get_model_output(audio_file_path)
    list_arousal_posteriors = model_output["arousal_post"]
    list_passive_arousal_posteriors = np.array(list_arousal_posteriors)[:,0]
    return list_passive_arousal_posteriors*100

def get_arousal_passive_posterior_mean(audio_file_path, posteriros_start_index, posteriros_end_index):
    arousal_posteriors = get_arousal_passive_posteriors(audio_file_path)[posteriros_start_index:posteriros_end_index]
    return mean(arousal_posteriors)

def get_dominance_strong_posteriors(audio_file_path):
    model_output = get_model_output(audio_file_path)
    list_dominance_posteriors = model_output["dominance_post"]
    list_strong_dominance_posteriors = np.array(list_dominance_posteriors)[:,1]
    return list_strong_dominance_posteriors*100

def get_dominance_strong_posterior_mean(audio_file_path, posteriros_start_index, posteriros_end_index):
    dominance_posteriors = get_dominance_strong_posteriors(audio_file_path)[posteriros_start_index:posteriros_end_index]
    return mean(dominance_posteriors)

def get_dominance_weak_posteriors(audio_file_path):
    model_output = get_model_output(audio_file_path)
    list_dominance_posteriors = model_output["dominance_post"]
    list_weak_dominance_posteriors = np.array(list_dominance_posteriors)[:,0]
    return list_weak_dominance_posteriors*100

def get_dominance_weak_posterior_mean(audio_file_path, posteriros_start_index, posteriros_end_index):
    dominance_posteriors = get_dominance_weak_posteriors(audio_file_path)[posteriros_start_index:posteriros_end_index]
    return mean(dominance_posteriors)

def get_valence_positive_posteriors(audio_file_path):
    model_output = get_model_output(audio_file_path)
    list_valence_posteriors = model_output["valence_post"]
    list_positive_valence_posteriors = np.array(list_valence_posteriors)[:,1]
    return list_positive_valence_posteriors*100

def get_valence_positive_posterior_mean(audio_file_path, posteriros_start_index, posteriros_end_index):
    valence_posteriors = get_valence_positive_posteriors(audio_file_path)[posteriros_start_index:posteriros_end_index]
    return mean(valence_posteriors)

def get_valence_negative_posteriors(audio_file_path):
    model_output = get_model_output(audio_file_path)
    list_valence_posteriors = model_output["valence_post"]
    list_negative_valence_posteriors = np.array(list_valence_posteriors)[:,0]
    return list_negative_valence_posteriors*100

def get_valence_negative_posterior_mean(audio_file_path, posteriros_start_index, posteriros_end_index):
    valence_posteriors = get_valence_negative_posteriors(audio_file_path)[posteriros_start_index:posteriros_end_index]
    return mean(valence_posteriors)

##################


def get_predicted_happiness(audio_signal):
    audio_signal_happiness_prob = [random() for a in audio_signal]
    return audio_signal_happiness_prob


def get_predicted_sadness(audio_signal):
    audio_signal_sadness_prob = [random() for a in audio_signal]
    return audio_signal_sadness_prob


def get_predicted_anger(audio_signal):
    audio_signal_anger_prob = [random() for a in audio_signal]
    return audio_signal_anger_prob


def get_predicted_frustration(audio_signal):
    audio_signal_frustration_prob = [random() for a in audio_signal]
    return audio_signal_frustration_prob


def get_predicted_neutrality(audio_signal):
    audio_signal_neutrality_prob = [random() for a in audio_signal]
    return audio_signal_neutrality_prob


##############


def get_predicted_arousal_base_mock(audio_signal):
    return [0.9, 0.8, 0.9, 0.7, 0.9]


def get_predicted_dominance_base_mock(audio_signal):
    return [0.1, 0.2, 0.3, 0.2, 0.1]


def get_predicted_valence_base_mock(audio_signal):
    return [0.4, 0.5, 0.5, 0.4, 0.4]


def get_predicted_happiness_base_mock(audio_signal):
    return [0.9, 0.7, 0.8, 0.6, 0.9]


def get_predicted_sadness_base_mock(audio_signal):
    return [0.1, 0.0, 0.1, 0.1, 0.0]


def get_predicted_anger_base_mock(audio_signal):
    return [0.1, 0.0, 0.1, 0.0, 0.0]


def get_predicted_frustration_base_mock(audio_signal):
    return [0.1, 0.0, 0.1, 0.0, 0.0]


def get_predicted_neutrality_base_mock(audio_signal):
    return [0.4, 0.5, 0.5, 0.4, 0.6]


def get_predicted_arousal_ctrl_mock(audio_signal):
    return [0.8, 0.7, 0.8, 0.6, 0.8]


def get_predicted_dominance_ctrl_mock(audio_signal):
    return [0.0, 0.1, 0.2, 0.1, 0.0]


def get_predicted_valence_ctrl_mock(audio_signal):
    return [0.3, 0.4, 0.4, 0.3, 0.3]


def get_predicted_happiness_ctrl_mock(audio_signal):
    return [0.8, 0.6, 0.7, 0.5, 0.8]


def get_predicted_sadness_ctrl_mock(audio_signal):
    return [0.1, 0.0, 0.0, 0.0, 0.0]


def get_predicted_anger_ctrl_mock(audio_signal):
    return [0.1, 0.0, 0.1, 0.0, 0.0]


def get_predicted_frustration_ctrl_mock(audio_signal):
    return [0.1, 0.1, 0.1, 0.0, 0.0]


def get_predicted_neutrality_ctrl_mock(audio_signal):
    return [0.3, 0.3, 0.2, 0.4, 0.5]
