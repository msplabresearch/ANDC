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
from transformers import Wav2Vec2Processor, Wav2Vec2Model, WavLMModel

# Self-Written Modules
sys.path.append(os.getcwd())
import sg_utils
import net

def main(args):
    sg_utils.set_deterministic(args.seed)

    audio_path = './data/' + args.corpus + '/Audios'

    if args.corpus[:11] == 'MSP-PODCAST':
        label_path = '/home/podcast/SER/Deployment/NEW_exp/model_podcast/data/MSP-PODCAST1.10/Partitioned_data_Primary_Emotion/labels_fear.csv'
    elif args.corpus[:11] == 'CREMA-D':
        label_path = '/media/lucas/LSG_SSD/model_podcast/data/labels_crema_disgust.csv'
            
  

    config_dict = sg_utils.load_env(args.conf_path)
    assert config_dict.get("config_root", None) != None, "No config_root in config/conf.json"
    # assert config_dict.get(args.corpus_type, None) != None, "Change config/conf.json"
    config_path = os.path.join(config_dict["config_root"], config_dict[args.corpus_type])
    sg_utils.print_config_description(config_path)

    # Make model directory
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)


    # Initialize dataset
    DataManager=sg_utils.DataManager(config_path)
    lab_type = args.label_type
    print(lab_type)
    if args.label_type == "dimensional":
        assert args.output_num == 3

    if args.label_type == "categorical":
        emo_num = DataManager.get_categorical_emo_num()
        print(emo_num)
        assert args.output_num == emo_num

    snum=1000000000000000
    train_wav_path = DataManager.get_wav_path(split_type="train",wav_loc=audio_path, lbl_loc=label_path)[:snum]
    # print(train_wav_path)
    train_utts = DataManager.get_utt_list("train",lbl_loc=label_path)[:snum]
    train_labs = DataManager.get_msp_labels(train_utts, lab_type=lab_type,lbl_loc=label_path)
    train_wavs = sg_utils.WavExtractor(train_wav_path).extract()

    dev_wav_path = DataManager.get_wav_path(split_type="dev",wav_loc=audio_path,lbl_loc=label_path)[:snum]
    # print(dev_wav_path)
    dev_utts = DataManager.get_utt_list("dev",lbl_loc=label_path)[:snum]
    dev_labs = DataManager.get_msp_labels(dev_utts, lab_type=lab_type,lbl_loc=label_path)
    dev_wavs = sg_utils.WavExtractor(dev_wav_path).extract()
    ###################################################################################################

    train_set = sg_utils.WavSet(train_wavs, train_labs, train_utts, 
        print_dur=True, lab_type=lab_type,
        label_config = DataManager.get_label_config(lab_type)
    )
    dev_set = sg_utils.WavSet(dev_wavs, dev_labs, dev_utts, 
        print_dur=True, lab_type=lab_type,
        wav_mean = train_set.wav_mean, wav_std = train_set.wav_std,
        label_config = DataManager.get_label_config(lab_type)
    )
    train_set.save_norm_stat(model_path+"/train_norm_stat.pkl")
    
    total_dataloader={
        "train": DataLoader(train_set, batch_size=args.batch_size, collate_fn=sg_utils.collate_fn_padd, shuffle=True),
        # "train": DataLoader(dev_set, batch_size=args.batch_size, collate_fn=sg_utils.collate_fn_padd, shuffle=True),
        "dev": DataLoader(dev_set, batch_size=args.batch_size, collate_fn=sg_utils.collate_fn_padd, shuffle=False)
        # "dev": DataLoader(dev_set, batch_size=args.batch_size, collate_fn=sg_utils.collate_fn_padd, shuffle=False)
    }

    # Initialize model
    modelWrapper = net.ModelWrapper(args) # Change this to use custom model
    modelWrapper.init_model()
    modelWrapper.init_optimizer()

    # Initialize loss function
    lm = sg_utils.LogManager()
    if args.label_type == "dimensional":
        lm.alloc_stat_type_list(["train_aro", "train_dom", "train_val",
            "dev_aro", "dev_dom", "dev_val"])
    elif args.label_type == "categorical":
        lm.alloc_stat_type_list(["train_loss", "train_acc", "dev_loss", "dev_acc"])

    epochs=args.epochs
    scaler = GradScaler()
    min_epoch = 0
    min_loss = 99999999999
    temp_dev = 99999999999
    losses_train, losses_dev = [], []
    for epoch in range(epochs):
        print("Epoch:",epoch)
        lm.init_stat()
        modelWrapper.set_train()
        for xy_pair in tqdm(total_dataloader["train"]):
            x = xy_pair[0]
            y = xy_pair[1]
            mask = xy_pair[2]

            x=x.cuda(non_blocking=True).float()
            y=y.cuda(non_blocking=True).float()
            mask=mask.cuda(non_blocking=True).float()

            
            with autocast():
                ## Feed-forward
                pred = modelWrapper.feed_forward(x, attention_mask=mask)
                # print(pred,'PREDICTION')
                
                ## Calculate loss
                total_loss = 0.0
                if args.label_type == "dimensional":
                    ccc = sg_utils.CCC_loss(pred, y)
                    loss = 1.0-ccc
                    total_loss += loss[0] + loss[1] + loss[2]
                elif args.label_type == "categorical":
                    # print(pred)
                    # print(y,'y')
                    loss = sg_utils.BCE_category(pred, y)
                    #loss = sg_utils.CE_category(pred, y)
                    total_loss += loss
                    ma, m1 = sg_utils.scores(pred, y)
                    

            ## Backpropagation
            modelWrapper.backprop(total_loss)

            # Logging
            if args.label_type == "dimensional":
                lm.add_torch_stat("train_aro", ccc[0])
                lm.add_torch_stat("train_dom", ccc[1])
                lm.add_torch_stat("train_val", ccc[2])
            elif args.label_type == "categorical":
                lm.add_torch_stat("train_loss", loss)
                lm.add_torch_stat("train_acc", m1)

        modelWrapper.set_eval()

        with torch.no_grad():
            total_pred = [] 
            total_y = []
            for xy_pair in tqdm(total_dataloader["dev"]):
                x = xy_pair[0]
                y = xy_pair[1]
                mask = xy_pair[2]

                x=x.cuda(non_blocking=True).float()
                y=y.cuda(non_blocking=True).float()
                mask=mask.cuda(non_blocking=True).float()
                # print(x, 'INPUUUUUT')
                pred = modelWrapper.feed_forward(x, attention_mask=mask, eval=True)
                # print(pred,'PREDICTION')
                # print(y,'GT')
                total_pred.append(pred)
                total_y.append(y)

            total_pred = torch.cat(total_pred, 0)
            total_y = torch.cat(total_y, 0)
        
        if args.label_type == "dimensional":
            ccc = sg_utils.CCC_loss(total_pred, total_y)            
            lm.add_torch_stat("dev_aro", ccc[0])
            lm.add_torch_stat("dev_dom", ccc[1])
            lm.add_torch_stat("dev_val", ccc[2])
        elif args.label_type == "categorical":
            loss = sg_utils.BCE_category(pred, y)

            ma, m1 = sg_utils.scores(pred, y)
            lm.add_torch_stat("dev_loss", loss)
            lm.add_torch_stat("dev_acc", m1)


        lm.print_stat()
        if args.label_type == "dimensional":
            dev_loss = 3.0 - lm.get_stat("dev_aro") - lm.get_stat("dev_dom") - lm.get_stat("dev_val")
        elif args.label_type == "categorical":
            dev_loss = lm.get_stat("dev_loss")
            tr_loss = lm.get_stat("train_loss")
            losses_dev.append(dev_loss)
            losses_train.append(tr_loss)
        if min_loss > dev_loss:
            min_epoch = epoch
            min_loss = dev_loss
        
        if float(dev_loss) < float(temp_dev):
            temp_dev = float(dev_loss)
            print('better dev loss found:' + str(float(dev_loss)) + ' saving model')
            modelWrapper.save_model(epoch)
    print("Save",end=" ")
    print(min_epoch, end=" ")
    print("")

    with open(model_path+'/train_loss.txt', 'w') as f:
        for item in losses_train:
            f.write("%s\n" % item)
    
    with open(model_path+'/dev_loss.txt', 'w') as f:
        for item in losses_dev:
            f.write("%s\n" % item)

    
    # print("Loss",end=" ")
    # if args.label_type == "dimensional":
    #     print(3.0-min_loss, end=" ")
    # elif args.label_type == "categorical":
    #     print(min_loss, end=" ")
    # print("")
    # modelWrapper.save_final_model(min_epoch, remove_param=False)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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
        '--sample_num',
        default=None,
        type=int)
    parser.add_argument(
        '--conf_path',
        default="config/conf.json",
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

    # Chunk Arguments
    parser.add_argument(
        '--use_chunk',
        default=False,
        type=str2bool)
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
        default="output",
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
