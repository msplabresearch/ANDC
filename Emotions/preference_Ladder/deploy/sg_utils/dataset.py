import torch
import torch.nn as nn
import torch.utils as torch_utils
import numpy as np
import sys
from multiprocessing import Pool
from tqdm import tqdm
from .normalizer import get_norm_stat_for_melspec
import librosa
from . import normalizer
import pickle as pk
import json
import os

class WavSet(torch_utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        super(WavSet, self).__init__()
        self.wav_list = kwargs.get("wav_list", args[0]) # (N, D, T)
        self.lab_list = kwargs.get("lab_list", args[1])
        self.utt_list = kwargs.get("utt_list", args[2])

        self.print_dur = kwargs.get("print_dur", False)
        self.lab_type = kwargs.get("lab_type", False)
        self.norm_method = kwargs.get("norm_method", "t-norm")

        self.wav_mean = kwargs.get("wav_mean", None)
        self.wav_std = kwargs.get("wav_std", None)

        # check max duration
        self.max_dur = np.min([np.max([len(cur_wav) for cur_wav in self.wav_list]), 12*16000])
        
        if self.norm_method == "t-norm":
            if self.wav_mean is None or self.wav_std is None:
                self.wav_mean, self.wav_std = normalizer.get_norm_stat_for_wav(self.wav_list)
        
    def save_norm_stat(self, norm_stat_file):
        with open(norm_stat_file, 'wb') as f:
            pk.dump((self.wav_mean, self.wav_std), f)
            
    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        cur_wav = self.wav_list[idx][:self.max_dur]
        cur_dur = len(cur_wav)
        if self.norm_method == "t-norm":
            cur_wav = (cur_wav - self.wav_mean) / (self.wav_std+0.000001)
        elif self.norm_method == "u-norm":
            cur_wav = (cur_wav - np.mean(cur_wav)) / (np.std(cur_wav)+0.000001)
        
        if self.lab_type == "dimensional":
            cur_lab = self.lab_list[idx]
            ## MSP-Podcast
            cur_lab = (cur_lab - 1) / (7-1)
            ## MSP-IMPROV
            # cur_lab[0] = (((cur_lab[0])-3)*(-1))+3
            # cur_lab = (cur_lab - 1) / (5-1)
        result = (cur_wav, cur_lab)
        if self.print_dur:
            result = (cur_wav, cur_lab, cur_dur)
        return result

def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    total_wav = []
    total_lab = []
    total_dur = []
    for wav, lab, dur, in batch:
        total_wav.append(torch.Tensor(wav))
        total_lab.append(lab)
        total_dur.append(dur)
    total_wav = nn.utils.rnn.pad_sequence(total_wav, batch_first=True)
    
    total_lab = torch.Tensor(total_lab)
    max_dur = np.max(total_dur)
    attention_mask = torch.zeros(total_wav.shape[0], max_dur)
    for data_idx, dur in enumerate(total_dur):
        attention_mask[data_idx,:dur] = 1
    ## compute mask
    return total_wav, total_lab, attention_mask


def cut_dataset(long_wav, utt_id, duration=16000*10):
    if type(duration) == str:
        duration = 16000 * float(duration.replace("s",""))
    wav_dur = len(long_wav)
    seg_num = int(len(long_wav)/duration)
    seg_list = []
    timestamp = []
    utt_ids = []
    for sidx_i in range(seg_num):
        sidx = sidx_i * duration
        eidx = (sidx_i+1) * duration
        cur_seg = long_wav[sidx:eidx] 
        seg_list.append(cur_seg)
        timestamp.append([int(sidx/16000), int(eidx/16000)])
        utt_ids.append(utt_id)
    
    return seg_list, timestamp, utt_ids
