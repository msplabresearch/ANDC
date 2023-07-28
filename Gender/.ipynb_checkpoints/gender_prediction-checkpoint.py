from __future__ import print_function, division
import os, sys
import torch
import numpy as np
from torch.autograd import Variable
import librosa
from scipy.stats import mode
import csv
import scipy
from keras.preprocessing.sequence import pad_sequences
from model import LSTMnet
import random
from tqdm import tqdm
from csv import writer
from pathlib import Path
import json




# Ignore warnings & Fix random seed
import warnings
warnings.filterwarnings("ignore")
random.seed(999)


def Spec_and_Phase(fpath):
    signal, rate  = librosa.load(fpath, sr=16000)
    signal = signal/np.max(abs(signal)) # Restrict value between [-1,1]
    F = librosa.stft(signal, n_fft=512, hop_length=256, win_length=512, window=scipy.signal.hamming)
    spec = np.abs(F)
    phase = np.angle(F)
    spec = np.reshape(spec.T,(spec.shape[1],spec.shape[0]))
    return spec

# Split Original batch Data into Small-Chunk batch Data Structure with padding
def SmallChunkSplitData(data, FrameSize):  
    start = 0
    Start = [0]
    End = []
    Split_Data = []
    if len(data)>=FrameSize:
        equeal_division_data = data[:(int(len(data)/FrameSize))*FrameSize]
        split_data = np.split(equeal_division_data,int(len(data)/FrameSize))           
        left_data = data[len(equeal_division_data):]
        if len(left_data)!=0:            
            pad_left_data = pad_sequences(left_data.T, maxlen=FrameSize ,dtype='float', padding='post', truncating='post')
            pad_left_data = pad_left_data.T                         
            Split_Data = Split_Data + split_data + [pad_left_data]
            Start.append(start+len(split_data)+1)
            End.append(start+len(split_data)+1)
            start = start+len(split_data)+1
        else:
            Split_Data = Split_Data + split_data
            Start.append(start+len(split_data))
            End.append(start+len(split_data))
            start = start+len(split_data)  
    else:
        left_data = data
        pad_left_data = pad_sequences(left_data.T, maxlen=FrameSize ,dtype='float', padding='post', truncating='post')
        pad_left_data = pad_left_data.T
        Split_Data = Split_Data + [pad_left_data]        
        Start.append(start+1)
        End.append(start+1)
        start = start+1 
    return np.array(Split_Data)

def prediction_folder(input_path):
    F_Name = []
    Pred_Rsl = []
    for root, directories, files in os.walk(input_path):
        # print(root)
        # files = sorted(files)
        # print(files)
        print('Gender Predictions')
        for filename in tqdm(files):
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            if '.wav' in filepath:
                try:
                    data = Spec_and_Phase(filepath)
                    chunk_data = SmallChunkSplitData(data, FrameSize=65)
                    # Data to torch & float for model input
                    chunk_data = torch.from_numpy(chunk_data)
                    chunk_data = chunk_data.float().to(device)
                    # Pred-chunk-labels for chunk data
                    pred_label = model(chunk_data)
                    pred_label = (np.round( (Variable(pred_label).data).cpu().numpy() )).reshape(-1)
                    # Output Results
                    F_Name.append(filename)
                    Pred_Rsl.append(mode(pred_label)[0][0])   # output voting result only 
                except:
                    print('Cannot Predict: '+filename)
    return F_Name, Pred_Rsl

###############################################################################
#import ANDC general arguments
file_dir = os.path.dirname(os.path.realpath(__file__))
utils_dir = os.path.join(Path(file_dir).parents[0], 'utils')
sys.path.append(utils_dir)
from andc_args import get_args


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")      




# Settings  
Training_epochs = 15
feat_type = 'Spec'
MODEL_STRUCT = 'LSTM'
LOADING_PATH = os.path.join(file_dir, feat_type+'_'+MODEL_STRUCT+'_epoch'+str(Training_epochs)+'.pt.tar')



# Loading Model Parameters
model = LSTMnet(input_dim=257, hidden_dim=150, output_dim=1, num_layers=2)
model.load_state_dict(torch.load(LOADING_PATH, map_location=torch.device(device)))
model = model.to(device) # if wants to predict on torch.tensor data
model.eval()





args = get_args()
root = args.root
output_location = os.path.join(root, "Outputs")

with open(os.path.join(os.path.join(output_location), "Short_files.json"), "r") as openfile:
    audio_files = json.load(openfile)
    
    
    
for filename, f_info in tqdm(audio_files.items()):
    f_path = f_info['filepaths']['wav']
    # music_info = f_info['gender']
    data = Spec_and_Phase(f_path)
    chunk_data = SmallChunkSplitData(data, FrameSize=65)
    # Data to torch & float for model input
    chunk_data = torch.from_numpy(chunk_data)
    chunk_data = chunk_data.float().to(device)
    # Pred-chunk-labels for chunk data
    pred_label = model(chunk_data)
    pred_label = (np.round( (Variable(pred_label).data).cpu().numpy() )).reshape(-1)
    # Output Results
    gender_pred = float(mode(pred_label)[0][0])   # output voting result only 
    if gender_pred == 1:
        gender_pred = 'Male'
    elif gender_pred == 0:
        gender_pred = 'Female'
    f_info['gender'] = gender_pred
    
json_object = json.dumps(audio_files, indent=4)
with open(os.path.join(output_location, "Short_files.json"), "w") as outfile:
    outfile.write(json_object)