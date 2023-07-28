# -*- coding: UTF-8 -*-
# Local modules
import os, sys
import argparse
# 3rd-Party Modules
import numpy as np
import pickle as pk
import glob
from tqdm import tqdm
import warnings
import librosa
from multiprocessing import Pool
warnings.filterwarnings('ignore')
# PyTorch Modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from pathlib import Path
from tqdm import tqdm
import json



    
#import ANDC general arguments
file_dir = os.path.dirname(os.path.realpath(__file__))
utils_dir = os.path.join(Path(file_dir).parents[2], 'utils')
sys.path.append(utils_dir)
from andc_args import get_args



args = get_args()
root = args.root
output_location = os.path.join(root, "Outputs")

with open(os.path.join(os.path.join(output_location), "Short_files.json"), "r") as openfile:
    audio_files = json.load(openfile)
    
out_path = os.path.join(output_location, "Short_split_w2v_large_feats")

def extract_wav(wav_path):
    raw_wav, _ = librosa.load(wav_path, sr=16000)
    return raw_wav


seed = 0

model_dir = os.path.join(file_dir, "model") # dir of wav2vec2.0 model

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
torch.manual_seed(seed)
np.random.seed(seed)


with open(os.path.join(model_dir, 'train_norm_stat.pkl'), 'rb') as f:
    wav_mean, wav_std = pk.load(f)
    

if not os.path.exists(out_path):
    os.mkdir(out_path)  
    
    
# Load model
wav2vec_model= Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-robust")
del wav2vec_model.encoder.layers[12:]
wav2vec_model.load_state_dict(torch.load(model_dir+"/final_wav2vec.pt", map_location=torch.device('cpu')))
# wav2vec_model.cuda()
wav2vec_model.eval()



print("facebook/wav2vec2-large-robust inference: ")
for filename, f_info in tqdm(audio_files.items()):
    f_path = f_info['filepaths']['wav']
    f_wav = extract_wav(f_path)
    
    x = torch.from_numpy((f_wav - wav_mean) / wav_std).unsqueeze(0)
    
    
    
    w2v = wav2vec_model(x).last_hidden_state
    w2v = w2v.squeeze(0).cpu().detach().numpy()  
    
    
    save_path = os.path.join(out_path, filename+'.npy')
    np.save(save_path, w2v)
    f_info['filepaths']['w2v_large_feats'] = save_path
    
    
print("Writing json file")
json_object = json.dumps(audio_files, indent=4)
with open(os.path.join(output_location, "Short_files.json"), "w") as outfile:
    outfile.write(json_object)    