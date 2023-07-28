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


# class DataManager():
#     def __load_config__(self, config_path):
#         with open(config_path, 'r') as f:
#             config_dict = json.load(f)
#         return config_dict
#     def __init__(self, *args, **kwargs):
#         config_path = kwargs.get("config_path", None)
#         self.config_dict=self.__load_config__(config_path)
#     def get_utts(self, *args, **kwargs):
#         raise NotImplementedError
#     def get_wav_paths(self, *args, **kwargs):
#         raise NotImplementedError
#     def get_feat_paths(self, *args, **kwargs):
#         raise NotImplementedError
#     def get_labels(self, *args, **kwargs):
#         raise NotImplementedError

# class MSP_Podcast_DataManager(DataManager):
#     def __check_env_validity__(self):
#         # version = self.config_dict.get("version", None)
#         audio_path = self.config_dict.get("audio_path", None)
#         label_path = self.config_dict.get("label_path", None)
#         if None in [audio_path, label_path]:
#             raise ValueError("Invalid environment configuration")

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.sample_num = kwargs.get("sample_num", None)

#     def get_utts(self, *args, **kwargs):
#         """
#         Input: split type
#             - If split type is None, return all utterances
#         Output: list of utterance IDs included in input split type
#         """
#         split_type = kwargs.get("split_type", None)    
#         if split_type != None:
#             split_id = self.config_dict["data_split_type"][split_type]

#         label_path = self.config_dict["label_path"]
#         utt_list=[]
#         with open(label_path, 'r') as f:
#             f.readline()
#             csv_reader = csv.reader(f)
#             for row in csv_reader:
#                 utt_id = row[0]
#                 stype = row[-1]
#                 if split_type != None:
#                     if stype == split_id:
#                         utt_list.append(utt_id) # retrun utterance IDs given split type
#                 else:
#                     utt_list.append(utt_id) # return all utterances
#         utt_list.sort()
#         return utt_list
    
#     def get_wav_paths(self, *args, **kwargs):
#         """
#         Input: split type
#             - If split type is None, return all utterances
#         Output: list of utterance IDs included in input split type
#         """
#         env_type = kwargs.get("env_type", "clean")  
#         split_type = kwargs.get("split_type", None)    
#         if split_type != None:
#             split_id = self.config_dict["data_split_type"][split_type]

#         label_path = self.config_dict["label_path"]
#         utt_list=[]
#         with open(label_path, 'r') as f:
#             f.readline()
#             csv_reader = csv.reader(f)
#             for row in csv_reader:
#                 utt_id = row[0]
#                 stype = row[-1]
#                 if split_type != None:
#                     if stype == split_id:
#                         utt_list.append(utt_id) # retrun utterance IDs given split type
#                 else:
#                     utt_list.append(utt_id) # return all utterances
#         utt_list.sort()
#         return utt_list

class DataManager:
    def __load_env__(self, env_path):
        with open(env_path, 'r') as f:
            env_dict = json.load(f)
        return env_dict

    def __init__(self, env_path):
        self.env_dict=self.__load_env__(env_path)
        self.msp_label_dict = None

    def get_wav_path(self, corpus_type, env_type, split_type=None, *args, **kwargs):
        raw_data_indicator = self.env_dict["RawDataIndicator"]
        cid = raw_data_indicator["corpus"][corpus_type]
        if env_type == "noisy":
            snr = kwargs.get("snr", None)
            assert snr in ["10db", "5db", "0db"], print("Invalid SNR value")
            eid = raw_data_indicator["environment"][env_type][snr]
        else: 
            eid = raw_data_indicator["environment"][env_type]
        wav_root=self.env_dict[cid]["wav"][eid]
        if eid == "clean" or "viewmaker" in eid or "noisy" in eid:
            if split_type == None:
                wav_list = glob.glob(os.path.join(wav_root, "*.wav"))
            else:
                sid = raw_data_indicator["data_split"][split_type]
                utt_list = self.get_utt_list(corpus_type, sid)
                wav_list = [os.path.join(wav_root, utt_id) for utt_id in utt_list]
        
        else:
            wav_list = glob.glob(os.path.join(wav_root, "*.wav"))
        wav_list.sort()
        return wav_list

    def get_feature_path(self, corpus_type, feature_type, env_type, split_type, *args, **kwargs):
        raw_data_indicator = self.env_dict["RawDataIndicator"]
        feat_root = raw_data_indicator["root"]
        cid = raw_data_indicator["corpus"][corpus_type]
        fid = raw_data_indicator["feature"][feature_type]
        sid = raw_data_indicator["data_split"][split_type]

        if env_type in ["noisy", "augmented"]:
            snr = kwargs.get("snr", None)
            assert snr in ["10db", "5db", "0db"], print("Invalid SNR value")
            eid = raw_data_indicator["environment"][env_type][snr]
        else:
            eid = raw_data_indicator["environment"][env_type]
        

        feature_dir = os.path.join(feat_root, cid, fid, eid, sid)
        return feature_dir

    def get_utt_list(self, corpus_type, split_type=None):
        cid = self.env_dict["RawDataIndicator"]["corpus"][corpus_type]
        if split_type is not None:
            sid = self.env_dict["RawDataIndicator"]["data_split"][split_type]
        else:
            sid = None
        assert cid in ["MSP-Podcast"]
        if cid == "MSP-Podcast":
            label_path = self.env_dict[cid]["label"]
            utt_list=[]
            with open(label_path, 'r') as f:
                f.readline()
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    utt_id = row[0]
                    stype = row[-1]
                    if stype == sid or sid == None:
                        utt_list.append(utt_id)
            utt_list.sort()
            return utt_list
    # def get_utt_list(self, split_type):
    #     sid = self.env_dict["data_split_type"][split_type]
    #     label_path = self.env_dict["label_path"]
    #     utt_list=[]
    #     with open(label_path, 'r') as f:
    #         f.readline()
    #         csv_reader = csv.reader(f)
    #         for row in csv_reader:
    #             utt_id = row[0]
    #             stype = row[-1]
    #             if stype == sid:
    #                 utt_list.append(utt_id)
    #     utt_list.sort()
    #     return utt_list

    def __load_msp_label_dict__(self):
        label_path = self.env_dict["MSP-Podcast"]["label"]
        # label_path = self.env_dict["label_path"]
        self.msp_label_dict=dict()
        
        with open(label_path, 'r') as f:
            header = f.readline().split(",")
            aro_idx = header.index("EmoAct")
            dom_idx = header.index("EmoDom")
            val_idx = header.index("EmoVal")

            csv_reader = csv.reader(f)
            for row in csv_reader:
                self.msp_label_dict[row[0]]=dict()
                self.msp_label_dict[row[0]]=[float(row[aro_idx]), float(row[dom_idx]), float(row[val_idx])]

    def get_msp_labels(self, utt_list, lab_type=None):
        if self.msp_label_dict == None:
            self.__load_msp_label_dict__()
        return np.array([self.msp_label_dict[utt_id] for utt_id in utt_list])
