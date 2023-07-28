import torch
from torch import nn
import sys
from src import models
from src.utils import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pre_trained_src.models as pre_model
from sklearn.metrics import f1_score
from src.eval_metrics import *
import random
import scipy.io as sio
from scipy.io import savemat, loadmat
import csv
import json 
from tqdm import tqdm

def initiate(hyp_params, model_dir):#, train_loader, valid_loader, test_loader):

    dev = {True:'cuda', False:'cpu'}
    device = torch.device(dev[torch.cuda.is_available()])
    
    output_dim = hyp_params.output_dim
    
    model_p = pre_model.SSL_MODEL(model_args=hyp_params)
    model_path = os.path.join(model_dir, 'ssl_pretrain_aud_auxnet_till400.pth')
    model_params = load_model(hyp_params, name=model_path)
    
    model_p.load_state_dict(model_params.state_dict(), strict=False)
    
    model = getattr(models, hyp_params.model)(model_p, hyp_params)
    
    model = model.to(device)


    
    
    criterion = getattr(nn, hyp_params.criterion)()

    settings = {'model': model,
                'criterion': criterion
                }

    return train_model(settings, hyp_params, model_dir)#, train_loader, valid_loader, test_loader)

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
    
    
def train_model(settings, args, model_dir):#, train_loader, valid_loader, test_loader):
    model = settings['model']
    # optimizer = settings['optimizer']
    # criterion = settings['criterion']    
    # scheduler = settings['scheduler']
    
    dev = {True:'cuda', False:'cpu'}
    device = torch.device(dev[torch.cuda.is_available()])


    def model_eval(model, criterion,  test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0
    
        results = []
        truths = []

        with torch.no_grad():
            for i_batch, (batch_X, batch_Y) in enumerate(loader):
                sample_ind, audio, vision = batch_X
                ground_truth = batch_Y.squeeze(dim=-1)            
            
                audio, ground_truth = audio.to(device), ground_truth.to(device)
                
                model.zero_grad()
                
                audio, ground_truth = audio.cuda(0), ground_truth.cuda(0)
                ground_truth = ground_truth.long()
                        
                batch_size = audio.size(0)
                
                net = nn.DataParallel(model) if batch_size > 10 else model
                preds_va, preds_a= net(audio)
                
                preds = (preds_va + preds_a) / 2
                
                
                labels = reshaping_lbl(batch_Y)                
                lbl = torch.Tensor(labels)
                ground_truth = lbl.long()


                crit = criterion(preds, ground_truth).item()

                mulc = crit * batch_size
                total_loss = total_loss + mulc

                results.append(preds)
                truths.append(ground_truth)
                
        avg_loss = total_loss / (args.n_test if test else args.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths
        
        
    def reshaping_lbl(x):
        
        ground_truth = x.squeeze(-1)  
        detached_gt = ground_truth.clone().detach()
        gt_arr = detached_gt.reshape(len(detached_gt),len(detached_gt[0][0]))
        labels = [np.where(val==1)[0][0] for val in gt_arr ]
        
        return labels


    model_path = os.path.join(model_dir, 'saud_finetunrelax_v2_5_aux2_45.pth')
    model = load_model(args, name=model_path)
    model.eval()

    dev = {True:'cuda', False:'cpu'}
    device = torch.device(dev[torch.cuda.is_available()])


    
    root = args.root
    output_location = os.path.join(root, "Outputs")

    with open(os.path.join(os.path.join(output_location), "Short_files.json"), "r") as openfile:
        audio_files = json.load(openfile)


    emos =  [ 'fear', 'sad', 'disgust', 'neutral', 'anger', 'happy']

    stats = []
    File_Name_fear,File_Name_sad,File_Name_disg,File_Name_neut,File_Name_ang,File_Name_hap = [],[],[],[],[],[]
    Pred_Senti_fear,Pred_Senti_sad,Pred_Senti_disg,Pred_Senti_neut,Pred_Senti_ang,Pred_Senti_hap = [],[],[],[],[],[]
    scores_senti_fear,scores_senti_sad,scores_senti_disg,scores_senti_neut,scores_senti_ang,scores_senti_hap = [],[],[],[],[],[]

    A_mean = sio.loadmat(os.path.join(model_dir,'aud_mean.mat'))['A_mean']

    A_std = sio.loadmat(os.path.join(model_dir,'aud_std.mat'))['A_std']

    A_mean = A_mean[0].astype(np.float32)

    A_std = A_std[0].astype(np.float32)
    


    print("Music/speech inference: ")
    for filename, f_info in tqdm(audio_files.items()):
        f_path = f_info['filepaths']['opensmile_lld']
        audio = LoadFeature(f_path)[:,1:].astype(np.float32)
    
    # for aud in audios:

        # audio = (sio.loadmat(root + '/' + aud )['Audio_data'])[:,1:].astype(np.float32)
        
        audio = (audio - A_mean)/(A_std + 1e-18)

        audio = torch.tensor([audio]).cpu().detach()

        audio = audio.to(device)
        
        # print(audio.size())
        model.zero_grad()
        
        # audio = audio.cuda(0)
        
        net = model
        preds_va, preds_a= net(audio)
        
        preds = (preds_va + preds_a) / 2

        m = nn.Softmax(dim=1)

        output = m(preds)

        score = output.cpu().detach().tolist()[0]
        # print(score)
        # ranking = ranking[::-1][0]

        # score = (np.array(output.cpu().detach())[0])[ranking]
        # emo = emos[ranking]
        f_info['opensmile_ssl'] = {}
        f_info['opensmile_ssl']['fear'] = [score[0],'Not_Used']
        f_info['opensmile_ssl']['sad'] = [score[1],'Not_Used']
        f_info['opensmile_ssl']['disgust'] = [score[2],'Not_Used']
        f_info['opensmile_ssl']['neutral'] = [score[3],'Not_Used']
        f_info['opensmile_ssl']['anger'] = [score[4],'Not_Used']
        f_info['opensmile_ssl']['happiness'] = [score[5],'Not_Used']
        
    print("Saving updated json file")
    json_object = json.dumps(audio_files, indent=4)
    with open(os.path.join(output_location, "Short_files.json"), "w") as outfile:
        outfile.write(json_object)
    
    print('Classifier Done!')