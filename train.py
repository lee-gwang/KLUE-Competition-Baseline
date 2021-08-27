# - train.py - #

# ------------------------
#  Import library
# ------------------------
import numpy as np
import os
import pickle
import sys
import pandas as pd
import re
import cv2
import argparse
# utils
from utils import load_data, get_learning_rate
from dataloader import KlueDataSet
# torch
import torch
import torch.cuda.amp as amp
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *

import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau, MultiStepLR, OneCycleLR
#

import math
import torch
from torch.optim.optimizer import Optimizer, required
import torch_optimizer as optim
from collections import defaultdict
import itertools as it

import tqdm
import random
#import time
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch

# transformer
import transformers
from transformers import XLMPreTrainedModel, XLMRobertaModel, XLMRobertaConfig, XLMRobertaTokenizer
from transformers import XLMRobertaForSequenceClassification, BertForSequenceClassification
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification, DistilBertForSequenceClassification, XLNetForSequenceClassification,\
XLMRobertaForSequenceClassification, XLMForSequenceClassification, RobertaForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

# ------------------------
#  Arguments
# ------------------------
parser = argparse.ArgumentParser(description='LG')
# setting
parser.add_argument('--debug', action='store_true', help='debugging mode')
parser.add_argument('--amp', action='store_true', help='mixed precision')
parser.add_argument('--gpu', type=str, default= '0,1', help='gpu')
parser.add_argument('--dir_', type=str, default= './saved_models/', help='dir')
parser.add_argument('--max_len', type=int, default= 33, help='max len')
# training
parser.add_argument('--epochs', type=int, default= 10, help='')
parser.add_argument('--start_lr', type=float, default= 1e-5, help='start learning rate')
parser.add_argument('--min_lr', type=float, default= 1e-6, help='min learning rate')
parser.add_argument('--batch_size', type=int, default= 32, help='')
parser.add_argument('--weight_decay', type=float, default= 1e-6, help='')
# model
parser.add_argument('--pt',  type=str, default= 'klue/roberta-base', help='huggingface models')
# else
parser.add_argument('--exp_name', type=str, default= 'experiment', help='experiments name')
parser.add_argument('--num_workers', type=int, default= 8, help='')
parser.add_argument('--seed', type=int, default= 42, help='')

args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    
def set_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True # for faster training, but not deterministic


def do_valid(net, valid_loader):

    val_loss = 0
    target_lst = []
    pred_lst = []
    logit = []
    loss_fn = nn.CrossEntropyLoss()

    net.eval()
    start_timer = timer()
    for t, data in enumerate(tqdm.tqdm(valid_loader)):
        ids  = data['ids'].to(device)
        mask  = data['mask'].to(device)
        tokentype = data['token_type_ids'].to(device)
        target = data['targets'].to(device)

        with torch.no_grad():
            if args.amp:
                with amp.autocast():
                    # output
                    output = net(ids, mask)[0]
                    # loss
                    loss = loss_fn(output, target)

            else:
                output = net(ids, mask)[0]
                loss = loss_fn(output, target)
            
            val_loss += loss
            target_lst.extend(target.detach().cpu().numpy())
            pred_lst.extend(output.argmax(dim=1).tolist())
            logit.extend(output.tolist())
            
        val_mean_loss = val_loss / len(valid_loader)
        validation_score = f1_score(y_true=target_lst, y_pred=pred_lst, average='macro')
        validation_acc = accuracy_score(y_true=target_lst, y_pred=pred_lst)
        

    return val_mean_loss, validation_score, validation_acc, logit

def run_train(folds=3):
    out_dir = args.dir_+ f'/fold{folds}/{args.exp_name}/'
    os.makedirs(out_dir, exist_ok=True)
    
    # load dataset
    train, test = load_data()    
    with open(f'./data/{args.pt}/train_data_{args.max_len}.pickle', 'rb') as f:
        train_data = pickle.load(f)
    with open(f'./data/{args.pt}/test_data_{args.max_len}.pickle', 'rb') as f:
        test_data = pickle.load(f)    
    
    # split fold
    for n_fold in range(5):
        if n_fold != folds:
            print(f'{n_fold} fold pass'+'\n')
            continue
            
        if args.debug:
            train = train.sample(1000).copy()
        
        trn_idx = train[train['fold']!=n_fold]['id'].values
        val_idx = train[train['fold']==n_fold]['id'].values
    

        train_dict = {'input_ids' : train_data['input_ids'][trn_idx] , 'attention_mask' : train_data['attention_mask'][trn_idx] , 
                      'token_type_ids' : train_data['token_type_ids'][trn_idx], 'targets' : train_data['targets'][trn_idx]}
        val_dict = {'input_ids' : train_data['input_ids'][val_idx] , 'attention_mask' : train_data['attention_mask'][val_idx] , 
                      'token_type_ids' : train_data['token_type_ids'][val_idx], 'targets' : train_data['targets'][val_idx]}

        ## dataset ------------------------------------
        train_dataset = KlueDataSet(data = train_dict)
        valid_dataset = KlueDataSet(data = val_dict)
        trainloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                 num_workers=8, shuffle=True, pin_memory=True)
        validloader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, 
                                 num_workers=8, shuffle=False, pin_memory=True)

        ## net ----------------------------------------
        scaler = amp.GradScaler()
        if 'xlm-roberta' in args.pt:
            net = XLMRobertaForSequenceClassification.from_pretrained(args.pt, num_labels = 7) 
        
        elif 'klue/roberta' in args.pt:
            net = RobertaForSequenceClassification.from_pretrained(args.pt, num_labels = 7) 
        else:
            net = BertForSequenceClassification.from_pretrained(args.pt, num_labels = 7) 

        net.to(device)
        if len(args.gpu)>1:
            net = nn.DataParallel(net)

        # ------------------------
        # loss
        # ------------------------
        loss_fn = nn.CrossEntropyLoss()

        # ------------------------
        #  Optimizer
        # ------------------------
        optimizer = optim.Lookahead(optim.RAdam(filter(lambda p: p.requires_grad,net.parameters()), lr=args.start_lr), alpha=0.5, k=5)

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = len(trainloader)*args.epochs)
        
        
        # ----
        start_timer = timer()
        best_score = 0

        for epoch in range(1, args.epochs+1):
            train_loss = 0
            valid_loss = 0

            target_lst = []
            pred_lst = []
            lr = get_learning_rate(optimizer)
            print(f'-------------------')
            print(f'{epoch}epoch start')
            print(f'-------------------'+'\n')
            print(f'learning rate : {lr : .6f}')
            for t, data in enumerate(tqdm.tqdm(trainloader)):

                # one iteration update  -------------
                ids  = data['ids'].to(device)
                mask  = data['mask'].to(device)
                tokentype = data['token_type_ids'].to(device)
                target = data['targets'].to(device)

                # ------------
                net.train()
                optimizer.zero_grad()


                if args.amp:
                    with amp.autocast():
                        # output
                        output = net(ids, mask)[0]

                        # loss
                        loss = loss_fn(output, target)
                        train_loss += loss


                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                else:
                    # output
                    output = net(ids, mask)[0]

                    # loss
                    loss = loss_fn(output, target)
                    train_loss += loss

                    # update
                    loss.backward()
                    optimizer.step()


                # for calculate f1 score
                target_lst.extend(target.detach().cpu().numpy())
                pred_lst.extend(output.argmax(dim=1).tolist())


                if scheduler is not None:
                    scheduler.step() 
            train_loss = train_loss / len(trainloader)
            train_score = f1_score(y_true=target_lst, y_pred=pred_lst, average='macro')
            train_acc = accuracy_score(y_true=target_lst, y_pred=pred_lst)

            # validation
            valid_loss, valid_score, valid_acc, _ = do_valid(net, validloader)


            if valid_acc > best_score:
                best_score = valid_acc
                best_epoch = epoch
                best_loss = valid_loss

                torch.save(net.state_dict(), out_dir + f'/{folds}f_{epoch}e_{best_score:.4f}_s.pth')
                print('best model saved'+'\n')


            print(f'train loss : {train_loss:.4f}, train f1 score : {train_score : .4f}, train acc : {train_acc : .4f}'+'\n')
            print(f'valid loss : {valid_loss:.4f}, valid f1 score : {valid_score : .4f}, valid acc : {valid_acc : .4f}'+'\n')


        print(f'best valid loss : {best_loss : .4f}'+'\n')
        print(f'best epoch : {best_epoch }'+'\n')
        print(f'best accuracy : {best_score : .4f}'+'\n')
   
if __name__ == '__main__':
    for i in [0,1,2,3,4]:
        run_train(folds=i)