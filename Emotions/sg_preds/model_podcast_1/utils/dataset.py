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
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class MelSetForGAN(torch_utils.data.Dataset):
    def __cut_segment__(self, cur_spec, max_dur, step_size):
        """
        cur_spec: (128, T)
        """
        segment_list = []
        sample_dur = cur_spec.shape[1]
        for sidx in range(0, sample_dur-max_dur, step_size):
            cur_sample=cur_spec[:,sidx:sidx+max_dur]
            segment_list.append(cur_sample)
        return segment_list

    def __init__(self, *args, **kwargs):
        super(MelSetForGAN, self).__init__()
        mel_list = kwargs.get("feat_list", args[0]) # (N, D, T)
        self.max_dur = kwargs.get("max_dur", 1500)
        self.step_size = kwargs.get("step_size", 500)
        
        self.feat_mean, self.feat_var = get_norm_stat_for_melspec(mel_list, feat_dim=128)
        self.feat_list = []
        for cur_spec in mel_list:
            self.feat_list.extend(self.__cut_segment__(cur_spec, self.max_dur, self.step_size))
    
    def __len__(self):
        return len(self.feat_list)

    def __getitem__(self, idx):
        feat_mat = self.feat_list[idx]
        feat_mat = feat_mat.T
        feat_mat = (feat_mat - self.feat_mean) / np.sqrt(self.feat_var + 0.000001)
        feat_mat = feat_mat.T
        return feat_mat

class WavSet(torch_utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        super(WavSet, self).__init__()
        self.wav_list = kwargs.get("wav_list", args[0]) # (N, D, T)
        self.lab_list = kwargs.get("lab_list", args[1])
        self.utt_list = kwargs.get("utt_list", args[2])
        self.print_dur = kwargs.get("print_dur", False)
        self.lab_type = kwargs.get("lab_type", False)

        self.wav_mean = kwargs.get("wav_mean", None)
        self.wav_std = kwargs.get("wav_std", None)

        self.label_config = kwargs.get("label_config", None)

        ## Assertion
        if self.lab_type == "categorical":
            assert len(self.label_config.get("emo_type", [])) != 0, "Wrong emo_type in config file"
        elif self.lab_type == "dimensional":
            assert self.label_config.get("max_score", None) != None and self.label_config.get("min_score", None) != None, \
            "You need to specify maximum and minimum attribute score in config file"
            self.max_lab_score =  self.label_config["max_score"]
            self.min_lab_score =  self.label_config["min_score"]
            self.flip_aro = str2bool(self.label_config.get("flip_aro", False))
        
        # check max duration
        self.max_dur = np.min([np.max([len(cur_wav) for cur_wav in self.wav_list]), 12*16000])
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
        cur_wav = (cur_wav - self.wav_mean) / (self.wav_std+0.000001)
        
        if self.lab_type == "dimensional":
            cur_lab = self.lab_list[idx]
            if self.flip_aro:
                cur_lab[0] = 6 - (cur_lab[0])
            cur_lab = (cur_lab - self.min_lab_score) / (self.max_lab_score-self.min_lab_score)
            
        elif self.lab_type == "categorical":
            cur_lab = self.lab_list[idx]


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
    for wav, lab, dur in batch:
        total_wav.append(torch.Tensor(wav))
        total_lab.append(lab)
        total_dur.append(dur)
    total_wav = nn.utils.rnn.pad_sequence(total_wav, batch_first=True)
    
    total_lab = torch.Tensor(total_lab)
    max_dur = np.max(total_dur)
    attention_mask = torch.zeros(total_wav.shape[0], max_dur)
    for dur in total_dur:
        attention_mask[:,:dur] = 1
    ## compute mask
    return total_wav, total_lab, attention_mask