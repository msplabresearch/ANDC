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



def initiate(hyp_params, train_loader, valid_loader, test_loader):

    dev = {True:'cuda', False:'cpu'}
    device = torch.device(dev[torch.cuda.is_available()])
    
    output_dim = hyp_params.output_dim
    
    model_p = pre_model.SSL_MODEL(model_args=hyp_params)
    model_params = load_model(hyp_params, name='ssl_pretrain_aud_auxnet_till400')
    
    model_p.load_state_dict(model_params.state_dict(), strict=False)
    
    # vec = [512, 260, 6]
    # for param in model_p.parameters():
    #    param.requires_grad = False
  

    # print(model_p)
    model = getattr(models, hyp_params.model)(model_p, hyp_params)
    
    model = model.to(device)

    # print('***'*20)
    # print(model)

    
    
    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', 
                                  patience=hyp_params.when, factor=0.1, verbose=True)
    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler}
    # print(9090)
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)

def train_model(settings, args, train_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']    
    scheduler = settings['scheduler']
    
    dev = {True:'cuda', False:'cpu'}
    device = torch.device(dev[torch.cuda.is_available()])

    def model_train(model, optimizer, criterion):
        epoch_loss = 0
        model.train()
        for i_batch, (batch_X, batch_Y) in enumerate(train_loader):

            sample_ind, audio, vision = batch_X
            labels = reshaping_lbl(batch_Y)
            lbl = torch.Tensor(labels)
            
            audio, lbl = audio.to(device), lbl.to(device)
            
            model.zero_grad()
            
            audio, ground_truth = audio.cuda(0), lbl.cuda(0)
            ground_truth = lbl.long()
            ground_truth = ground_truth.long()
            
            batch_size = audio.size(0)
    
            net = nn.DataParallel(model) if batch_size > 10 else model

            # else:
            preds_va, preds_a = net(audio)

            ground_truth = ground_truth.view(-1)
            
            lossva = criterion(preds_va, ground_truth)
            lossa = criterion(preds_a, ground_truth)
            
            wva, wa = .5, .5
            
            loss = wva * lossva + wa * lossa
            
            # loss = criterion(preds, ground_truth)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            epoch_loss += loss.item() * batch_size

                
                
        return epoch_loss / args.n_train

    def model_eval(model, criterion, test=False):
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

    best_valid = 1e8
    for epoch in range(1, args.num_epochs+1):
        start = time.time()
        model_train(model, optimizer, criterion)
        val_loss, _, _ = model_eval(model, criterion, test=False)
        test_loss, _, _ = model_eval(model, criterion, test=True)
        
        end = time.time()
        time_interval = end-start
        scheduler.step(val_loss)

        # print("-"*25)
        # print('Epoch {:2d}/{:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, args.num_epochs, time_interval, val_loss, test_loss))
        # print("-"*25)
        
        if val_loss < best_valid:
            print(f"Saved model at saved_models/{args.name}.pt!")
            save_model(args, model, name=args.name)
            best_valid = val_loss

            print("-"*25)
            print('Epoch {:2d}/{:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, args.num_epochs, time_interval, val_loss, test_loss))
            print("-"*25)
            
    model = load_model(args, name=args.name)
    _, results, truths = model_eval(model, criterion, test=True)

    scores(results, truths)

    sys.stdout.flush()
