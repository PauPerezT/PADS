import pandas as pd
import numpy as np 


# Remove labels of audio files that do not exist in the given sessions audio files
def matchLabelsToAudios(labels_csv, dict_features_tensors):
    label_files = labels_csv[['File_Name']]
    updated_csv= labels_csv
    dict_audio_files = dict_features_tensors.values()

    audio_files_names = []
    for dict_file in dict_audio_files: 
        files = dict_file.keys()
        audio_files_names.extend(key for key in files)

    audio_files_names = np.array(list(audio_files_names))

    for index, label_file_name in label_files.iterrows():
        if label_file_name[0] not in audio_files_names:
            updated_csv = updated_csv.drop([index])
            
        
    return updated_csv

