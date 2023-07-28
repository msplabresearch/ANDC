# -*- coding: UTF-8 -*-
# Local modules
import os
import sys
import argparse
# 3rd-Party Modules
import numpy as np
import pickle as pk
from tqdm import tqdm

# PyTorch Modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import json
# Self-Written Modules
sys.path.append(os.getcwd())
import sg_utils
import net
from pathlib import Path
import gc

# Save predictions
import pandas as pd


def main(args):
    
    #import ANDC general arguments
    file_dir = os.path.dirname(os.path.realpath(__file__))
    # utils_dir = os.path.join(Path(file_dir).parents[1], 'utils')
    # sys.path.append(utils_dir)
    # from andc_args import get_args


    root = args.root
    output_location = os.path.join(root, "Outputs")

    with open(os.path.join(os.path.join(output_location), "Short_files.json"), "r") as openfile:
        audio_files = json.load(openfile)
    
    
    sg_utils.set_deterministic(args.seed)
    
    
    to_be_processed = os.path.join(output_location,  'Short_split_file')
    # if args.train_type == "manually_finetuned":
    model_path = os.path.join(file_dir, args.model_path)
    
    modelWrapper = net.ModelWrapper(args) # Change this to use custom model
    modelWrapper.init_model()
    modelWrapper.load_model(model_path)
    modelWrapper.set_eval()
    
    fnames = os.listdir(to_be_processed)
    if (len(fnames) % 2000) != 0:
        iter = len(fnames) // 2000
        iter += 1
    else:
        iter = len(fnames) // 2


    for n in range(iter):
        locs = []
        for fname in fnames[n*2000:2000+(n*2000)]:
            locs.append(os.path.join(to_be_processed, fname))
        
        
        
        config_dict = sg_utils.load_env(args.conf_path)
        assert config_dict.get("config_root", None) != None, "No config_root in config/conf.json"
        assert config_dict.get(args.corpus_type, None) != None, "Change config/conf.json"
        config_path = os.path.join(config_dict["config_root"], config_dict[args.corpus_type])
        sg_utils.print_config_description(config_path)
        
        DataManager=sg_utils.DataManager(config_path)
        lab_type = args.label_type
        print(lab_type)
        print(args.label_learning)
        
        if args.label_type == "dimensional":
            assert args.output_num == 3
        elif args.label_type == "categorical":
            emo_num = DataManager.get_categorical_emo_num()
            assert args.output_num == emo_num
            
        test_wav_path = locs
        test_utts = fnames[n*2000:2000+(n*2000)]#[:10]
        test_wav_path.sort()
        test_utts.sort()
        test_wavs = sg_utils.WavExtractor(test_wav_path).extract()
        ###################################################################################################
        with open(args.model_path+"/train_norm_stat.pkl", 'rb') as f:
            wav_mean, wav_std = pk.load(f)
        test_set = sg_utils.WavSet(test_wavs, test_utts, 
            print_dur=True, lab_type=lab_type, print_utt=True,
            wav_mean = wav_mean, wav_std = wav_std, 
            label_config = DataManager.get_label_config(lab_type)
        )


        lm = sg_utils.LogManager()
        if args.label_type == "dimensional":
            lm.alloc_stat_type_list(["test_aro", "test_dom", "test_val"])
        elif args.label_type == "categorical":
            lm.alloc_stat_type_list(["test_loss", "test_acc"])

        batch_size=args.batch_size
        test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=sg_utils.collate_fn_padd, shuffle=False)


        with torch.no_grad():
            total_pred = [] 
            # total_y = []
            total_utts = []
            for xy_pair in tqdm(test_loader):
                x = xy_pair[0]
                y = xy_pair[1]
                mask = xy_pair[2]
                utt_ids = xy_pair[3]

                
                pred = modelWrapper.feed_forward(x, attention_mask=mask, eval=True)
                pred = torch.sigmoid(pred).cpu().tolist()
                # print(pred,'PREDICTION')
                total_pred.append(pred)
                # total_y.append(y)
                total_utts.append(utt_ids)
                # print(utt_ids)
                # print(pred)
                for i, audio_file in enumerate(utt_ids):
                    basename = audio_file.split('.')[0]
                    audio_info = audio_files[basename]
                    audio_info['fear_CREAMA_pred'] = [pred[i][0],'Not_Used']
                    
                

        # from csv import writer
        print("Saving updated json file")
        json_object = json.dumps(audio_files, indent=4)
        with open(os.path.join(output_location, "Short_files.json"), "w") as outfile:
            outfile.write(json_object)
  

        lm.print_stat()
        del total_pred
        del test_set
        del test_loader
        del test_wavs
        gc.collect()


if __name__ == "__main__":
    # Inputs for the main function
    parser = argparse.ArgumentParser()

    # Experiment Arguments
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        default='cuda',
        type=str)
    parser.add_argument(
        '--seed',
        default=0,
        type=int)
    parser.add_argument(
        '--conf_path',
        default="config/conf.json",
        type=str)
    
    
    parser.add_argument(
        '--root',
        default="./",
        type=str)

    # Data Arguments
    parser.add_argument(
        '--data_type',
        default="clean",
        type=str)
    parser.add_argument(
        '--corpus_type',
        default="podcast_v1.7",
        type=str)
    parser.add_argument(
        '--snr',
        default=None,
        type=str)
    parser.add_argument(
        '--model_type',
        default="wav2vec2",
        type=str)
    parser.add_argument(
        '--label_type',
        choices=['dimensional', 'categorical'],
        default='categorical',
        type=str)

    parser.add_argument(
        '--chunk_hidden_dim',
        default=256,
        type=int)
    parser.add_argument(
        '--chunk_window',
        default=50,
        type=int)
    parser.add_argument(
        '--chunk_num',
        default=11,
        type=int)
    
    # Model Arguments
    parser.add_argument(
        '--model_path',
        default=None,
        type=str)
    parser.add_argument(
        '--output_num',
        default=4,
        type=int)
    parser.add_argument(
        '--batch_size',
        default=128,
        type=int)
    parser.add_argument(
        '--hidden_dim',
        default=256,
        type=int)
    parser.add_argument(
        '--num_layers',
        default=3,
        type=int)
    parser.add_argument(
        '--epochs',
        default=100,
        type=int)
    parser.add_argument(
        '--lr',
        default=1e-5,
        type=float)
    parser.add_argument(
        '--noise_dur',
        default="30m",
        type=str)
    
    # Label Learning Arguments
    parser.add_argument(
        '--label_learning',
        default="multi-label",
        type=str)

    parser.add_argument(
        '--corpus',
        default="USC-IEMOCAP",
        type=str)
    parser.add_argument(
        '--num_classes',
        default="four",
        type=str)
    parser.add_argument(
        '--label_rule',
        default="M",
        type=str)
    parser.add_argument(
        '--partition_number',
        default="1",
        type=str)
    parser.add_argument(
        '--data_mode',
        default="primary",
        type=str)



    args = parser.parse_args()

    # Call main function
    main(args)


    gc.collect()