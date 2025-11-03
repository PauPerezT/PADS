import numpy as np 
import os
import torch
from torch.utils.data import Dataset
from sklearn.utils import class_weight
import pandas as pd
from prepareAudiosAndLabels import matchLabelsToAudios

# Speech dataset class for training and validation of audio files features and labels
class SpeechDataset(Dataset):
    
    def __init__(self, files_features, files_labels, dataType):
        self.files_features=list(files_features.values())
        self.files_labels=list(files_labels.values())
        print("***** Length of audio files features in", dataType, ":", len(self.files_features), " *****")
        print("***** Length of audio files labels in", dataType, ":", len(self.files_labels), " *****")

        self.files_features, self.files_labels = self.reshape_data(files_features)

    def reshape_data(self, files_features_dict):
        files_features_reshaped =[]
        files_labels_reshaped = []
        file_index= 0
        for file_feats in self.files_features:
            file_feats=np.array(file_feats)
            if file_feats.shape[0] == 0:
                file_index += 1
                continue
            files_features_reshaped.extend(file_feats)

            file_labels_repeated = np.repeat(self.files_labels[file_index], file_feats.shape[0])
            files_labels_reshaped.extend(file_labels_repeated)
            file_index += 1

        files_features_reshaped = np.array(files_features_reshaped)
        files_labels_reshaped = np.array(files_labels_reshaped)

        return files_features_reshaped, files_labels_reshaped

    def __getitem__(self, index):
        
        if torch.is_tensor(index):
            index = index.tolist()

        frame_tensor = torch.tensor(self.files_features[index])
        label_tensor = torch.tensor(self.files_labels[index])

        return (frame_tensor, label_tensor)

    def __len__(self):
        return len(self.files_labels)
    
    def class_weight(self):
        labels_vector=np.hstack(self.files_labels)
        weighted_labels = class_weight.compute_class_weight('balanced', classes=np.unique(labels_vector), y=labels_vector)
        self.classWeights=torch.Tensor(weighted_labels)
        return  self.classWeights

# Obtain filtered labels csv dataset of audio files for the given data type (Training, Validation or Testing)
def readCsvFileForLabels(labels_csv, dataType):
    filtered_csv=[]
    
    if dataType == 'Training':
        training_files_ids = ['01F', '01M', '02F', '02M']
        training_csv = labels_csv[labels_csv['ID'].isin(training_files_ids)]
        filtered_csv = training_csv
    
    elif dataType == 'Validation':
        val_files_ids = ['03F', '03M', '04F', '04M']
        val_csv = labels_csv[labels_csv['ID'].isin(val_files_ids)]
        filtered_csv = val_csv

    elif dataType == 'Testing':
        test_files_ids = ['05F', '05M']
        test_csv = labels_csv[labels_csv['ID'].isin(test_files_ids)]
        filtered_csv = test_csv

    else:
        print("Error: valid data set names: Training, Validation, Testing")
        
    return filtered_csv

# Obtain filtered labels csv dataset for the given emotional dimension
def get_emotional_dimension_scores(csv_file, emotional_dimension):

    filtered_csv=[]
    
    if emotional_dimension == 'arousal':
        arousal_csv = csv_file[['ID', 'File_Name', 'arousal']]
        filtered_csv = arousal_csv
    
    elif emotional_dimension == 'dominance':
        dominance_csv = csv_file[['ID', 'File_Name', 'dominance']]
        filtered_csv = dominance_csv

    elif emotional_dimension == 'valence':
        valence_csv = csv_file[['ID', 'File_Name', 'valence']]
        filtered_csv = valence_csv

    else:
        print("Error: valid emotional dimensions: arousal, dominance, valence")

    return filtered_csv

# Obtain class label (0 or 1) from the given emotion score
def emotional_dim_labels(csv_emotional_dim_scores):
    dict_labels = {}
    np_emotional_dim_scores = np.array(csv_emotional_dim_scores)
    np_class_labels = np.where(np_emotional_dim_scores[:,2] < 3, 0, 1)
    tensor_class_labels = torch.from_numpy(np_class_labels)

    for file_index in range(len(np_class_labels)):
        dict_labels[np_emotional_dim_scores[file_index, 1]] = tensor_class_labels[file_index]

    return dict_labels

# Get training features and class weights of the given emotional dimension
def get_train_dataset(path_labels_csv, features_tensors, emotional_dimension):
    labels_csv = pd.read_csv(path_labels_csv+'metadata_IEMOCAP.csv', sep=',')
    updated_labels_csv = matchLabelsToAudios(labels_csv, features_tensors)
    
    dataType = 'Training'
    dict_training_feats = features_tensors[dataType]
    csv_training = readCsvFileForLabels(updated_labels_csv, dataType)
    
    training_scores = get_emotional_dimension_scores(csv_training, emotional_dimension)

    dict_training_labels = emotional_dim_labels(training_scores)

    training_data = SpeechDataset(dict_training_feats, dict_training_labels, dataType)
    weigths = training_data.class_weight()

    return training_data, weigths

# Get validation features of the given emotional dimension
def get_val_dataset(path_labels_csv, features_tensors, emotional_dimension):
    labels_csv = pd.read_csv(path_labels_csv+'metadata_IEMOCAP.csv', sep=',')
    updated_labels_csv = matchLabelsToAudios(labels_csv, features_tensors)
    
    dataType = 'Validation'
    dict_val_feats = features_tensors[dataType]
    csv_val = readCsvFileForLabels(updated_labels_csv, dataType)
    
    val_scores = get_emotional_dimension_scores(csv_val, emotional_dimension)

    dict_val_labels = emotional_dim_labels(val_scores)

    val_data = SpeechDataset(dict_val_feats, dict_val_labels, dataType)

    return val_data

# Get testing features of the given emotional dimension
def get_test_dataset(path_labels_csv, features_tensors, emotional_dimension):
    labels_csv = pd.read_csv(path_labels_csv+'metadata_IEMOCAP.csv', sep=',')
    updated_labels_csv = matchLabelsToAudios(labels_csv, features_tensors)
    
    dataType = 'Testing'
    dict_test_feats = features_tensors[dataType]
    csv_test = readCsvFileForLabels(updated_labels_csv, dataType)
    
    test_scores = get_emotional_dimension_scores(csv_test, emotional_dimension)

    dict_test_labels = emotional_dim_labels(test_scores)

    test_data = SpeechDataset(dict_test_feats, dict_test_labels, dataType)

    return test_data

