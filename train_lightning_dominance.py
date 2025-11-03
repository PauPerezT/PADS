import torch as t
from lightning import Trainer
from model.model_first_selfCNN_8_3_GRU_lightning import  SelfConvData, SelfConv
import argparse

from Spectrum_Rep import getFeaturesTensors
from normalizeFeatures import normalizeFeaturesTensors
from prepareData import get_train_dataset, get_val_dataset

emotional_dimension = 'dominance'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--audio_path', default= r"./audios/",help='Path to load audio files to compute the spectrograms and mfccs')
    parser.add_argument('--features_path', default='./feats/', help='Path to save the spectrograms and mfccs i.e. features')
    parser.add_argument('--labels_csv_path', default='./labels/', help='Path to load the CSV file containing labels of audio files')
    parser.add_argument('--checkpoints_path', default='./checkpoints/', help='Path to save checkpoints after training')

    args = parser.parse_args()

    path_audio=args.audio_path 
    path_feats=args.features_path
    path_labels_csv=args.labels_csv_path
    path_checkpoints=args.checkpoints_path

    num_epochs = 200
    lr = 0.001
    bs=100
    input_shape = (bs,3,100,64)
    num_gpus = 1

    if bs==1:
        params = {'batch_size': bs,
              'shuffle': True,
              'drop_last':True}
    else:
        params = {'batch_size': bs,
          'shuffle': True,
          'drop_last':True,
          'num_workers':7,#8 or 16 acc. to HPC
          'pin_memory':True}

    features, path_tensors = getFeaturesTensors(path_audio, path_feats)
    feats_normalized = normalizeFeaturesTensors(features)

    train_data_set,weight = get_train_dataset(path_labels_csv, feats_normalized, emotional_dimension)
    train_data = t.utils.data.DataLoader(train_data_set, **params)

    dev_data_set = get_val_dataset(path_labels_csv, feats_normalized, emotional_dimension)
    dev_data = t.utils.data.DataLoader(dev_data_set, **params)

    model_data = SelfConvData(train_data=train_data, val_data=dev_data)
    model = SelfConv(class_weights=weight, learning_rate= lr, nc=3, input_shape = input_shape)
    
    trainer = Trainer(num_nodes=num_gpus,max_epochs=num_epochs, fast_dev_run=False, default_root_dir=path_checkpoints)
    trainer.fit(model, datamodule=model_data)
