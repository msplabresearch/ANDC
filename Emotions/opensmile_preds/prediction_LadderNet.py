#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:51:58 2020

@author: winston
"""
from scipy.io import loadmat
import os, sys
import numpy as np
from LadderNet_model import ladder_network_multi
import csv
import tensorflow as tf
from csv import writer
from tqdm import tqdm
from pathlib import Path
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#import ANDC general arguments
file_dir = os.path.dirname(os.path.realpath(__file__))
utils_dir = os.path.join(Path(file_dir).parents[1], 'utils')
sys.path.append(utils_dir)
from andc_args import get_args

args = get_args()
root = args.root
output_location = os.path.join(root, "Outputs")

with open(os.path.join(os.path.join(output_location, "Short_files.json")), "r") as openfile:
    audio_files = json.load(openfile)
    
# Parameters
batch_size = 256
epochs = 50

# Data/Label Dir
# root_dir = '/home/podcast/Desktop/MSP_Podcast_FILTER/emotions_feats_and_preds/Features/OpenSmile_func_IS13ComParE/feat_mat/'
meta_path = os.path.join(file_dir,'LadderNet','NormTerm')
Feat_mean_All = loadmat(os.path.join(meta_path,'feat_norm_means.mat'))['normal_para']
Feat_std_All = loadmat(os.path.join(meta_path,'feat_norm_stds.mat'))['normal_para']  
Label_mean_act = loadmat(os.path.join(meta_path,'act_norm_means.mat'))['normal_para'][0][0]
Label_std_act = loadmat(os.path.join(meta_path,'act_norm_stds.mat'))['normal_para'][0][0]
Label_mean_dom = loadmat(os.path.join(meta_path,'dom_norm_means.mat'))['normal_para'][0][0]
Label_std_dom = loadmat(os.path.join(meta_path,'dom_norm_stds.mat'))['normal_para'][0][0]
Label_mean_val = loadmat(os.path.join(meta_path,'val_norm_means.mat'))['normal_para'][0][0]
Label_std_val = loadmat(os.path.join(meta_path,'val_norm_stds.mat'))['normal_para'][0][0]


def TryToFloat(single_data):
    try:
        return float(single_data)
    except:
        return None
    
def LoadFeature(filename):
    content = open(filename, 'r').read()
    data = content.split('@data\n')[1].split('\n')
    data = filter(None, data)
    feature = [[TryToFloat(data_split) for data_split in d.split(',') \
                if TryToFloat(data_split)!=None] for d in data]
    return np.array(feature)


# Regression Task => Prediction & De-Normalize Target
model_path = file_dir+'/LadderNet/Model/LadderNN_MTL_model[epoch'+str(epochs)+'-batch'+str(batch_size)+'].hdf5'
model, _ = ladder_network_multi(num_nodes=256, noise=0.3, alpha=0, beta=0, gamma=0, seeding=0)
with tf.device('/cpu:0'):
    model.load_weights(model_path)
    
pred_type = 'LadderNet'
for filename, f_info in tqdm(audio_files.items()):
    f_path = f_info['filepaths']['opensmile_hld']
    data = LoadFeature(f_path)
    data = (data-Feat_mean_All)/Feat_std_All    # Feature Normalization  
    data = data.reshape(-1)
    # Bounded NormFeat Range -3~3 and assign NaN to 0
    data[np.isnan(data)]=0
    data[data>3]=3
    data[data<-3]=-3

    # Recording prediction time cost
    _, pred_act, pred_dom, pred_val, _, _, _ = model.predict(np.expand_dims(data, axis=0))
    # Output prediction results    
    pred_act, pred_dom, pred_val = np.mean(pred_act), np.mean(pred_dom), np.mean(pred_val)
    
    # Regression Task => Prediction & De-Normalize Target
    pred_act = (Label_std_act*pred_act)+Label_mean_act
    pred_dom = (Label_std_dom*pred_dom)+Label_mean_dom
    pred_val = (Label_std_val*pred_val)+Label_mean_val
    
    f_info[pred_type] = {}
    f_info[pred_type]['act'] = [pred_act,'Not_Used']
    f_info[pred_type]['dom'] = [pred_dom,'Not_Used']
    f_info[pred_type]['val'] = [pred_val,'Not_Used']
    
print("Saving updated json file")
json_object = json.dumps(audio_files, indent=4)
with open(os.path.join(output_location, "Short_files.json"), "w") as outfile:
    outfile.write(json_object)    