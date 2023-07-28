import os
import csv
import glob
import json
import numpy as np
from . import utterance

"""
All DataManager classes should follow the following interface:
    1. Input must be a configuration dictionary
    2. All classes must have an assert function that checks if the input is
        valid for designated corpus type
        1) Configuration dictionary must have the following keys:
            - "audio": directory of the audio files
            - "label": path of the label files
    3. Must have a function that returns a list of following items:
        1) List of utterance IDs
        2) List of features
        3) List of categorical labels
        4) List of dimensional labels
"""


def load_env(env_path):
    with open(env_path, 'r') as f:
        env_dict = json.load(f)
    return env_dict

class DataManager:
    def __load_env__(self, env_path):
        with open(env_path, 'r') as f:
            env_dict = json.load(f)
        return env_dict

    def __init__(self, env_path):
        self.env_dict=self.__load_env__(env_path)
        self.msp_label_dict = None

    def get_wav_path(self, split_type=None, wav_loc=None, lbl_loc=None , *args, **kwargs):
        wav_root=wav_loc
        if split_type == None:
            wav_list = glob.glob(os.path.join(wav_root, "*.wav"))
        else:
            utt_list = self.get_utt_list(split_type,  lbl_loc)
            wav_list = [os.path.join(wav_root, utt_id) for utt_id in utt_list]
        
        wav_list.sort()
        return wav_list

    # def get_feature_path(self, corpus_type, feature_type, env_type, split_type, *args, **kwargs):
    #     raw_data_indicator = self.env_dict["RawDataIndicator"]
    #     feat_root = raw_data_indicator["root"]
    #     cid = raw_data_indicator["corpus"][corpus_type]
    #     fid = raw_data_indicator["feature"][feature_type]
    #     sid = raw_data_indicator["data_split"][split_type]

    #     if env_type in ["noisy", "augmented"]:
    #         snr = kwargs.get("snr", None)
    #         assert snr in ["10db", "5db", "0db"], print("Invalid SNR value")
    #         eid = raw_data_indicator["environment"][env_type][snr]
    #     else:
    #         eid = raw_data_indicator["environment"][env_type]
        

    #     feature_dir = os.path.join(feat_root, cid, fid, eid, sid)
    #     return feature_dir

    def get_utt_list(self, split_type, lbl_loc):
        label_path = lbl_loc
        utt_list=[]
        sid = self.env_dict["data_split_type"][split_type]
        with open(label_path, 'r') as f:
            f.readline()
            csv_reader = csv.reader(f)
            for row in csv_reader:
                utt_id = row[0]
                stype = row[-1]
                if stype == sid:
                    utt_list.append(utt_id)
        utt_list.sort()
        return utt_list

    def __load_msp_cat_label_dict__(self,lbl_loc):
        label_path = lbl_loc
        self.msp_label_dict=dict()
        emo_class_list =  self.get_categorical_emo_class()
        print(emo_class_list)
        with open(label_path, 'r') as f:
            header = f.readline().split(",")
            emo_idx_list = []
            for emo_class in emo_class_list:
                emo_idx_list.append(header.index(emo_class))
            # a_idx = header.index("angry")
            # s_idx = header.index("sad")
            # n_idx = header.index("neutral")
            # h_idx = header.index("happy")

            csv_reader = csv.reader(f)
            for row in csv_reader:
                utt_id = row[0]
                self.msp_label_dict[utt_id]=dict()
                cur_emo_lab = []
                for emo_idx in emo_idx_list:
                    cur_emo_lab.append(float(row[emo_idx]))
                self.msp_label_dict[utt_id]=cur_emo_lab


    def __load_msp_dim_label_dict__(self, lbl_loc):
        label_path = lbl_loc
        self.msp_label_dict=dict()
        
        with open(label_path, 'r') as f:
            header = f.readline().split(",")
            aro_idx = header.index("EmoAct")
            dom_idx = header.index("EmoDom")
            val_idx = header.index("EmoVal")

            csv_reader = csv.reader(f)
            for row in csv_reader:
                utt_id = row[0]
                self.msp_label_dict[utt_id]=dict()
                self.msp_label_dict[utt_id]=[float(row[aro_idx]), float(row[dom_idx]), float(row[val_idx])]
    
    def get_msp_labels(self, utt_list, lab_type=None,lbl_loc=None):
        if lab_type == "categorical":
            self.__load_msp_cat_label_dict__(lbl_loc)
        elif lab_type == "dimensional":
            self.__load_msp_dim_label_dict__(lbl_loc)
        return np.array([self.msp_label_dict[utt_id] for utt_id in utt_list])

    def get_categorical_emo_class(self):
        return self.env_dict["categorical"]["emo_type"]
    def get_categorical_emo_num(self):
        cat_list = self.get_categorical_emo_class()
        return len(cat_list)

    def get_label_config(self, label_type):
        assert label_type in ["categorical", "dimensional"]
        return self.env_dict[label_type]
        