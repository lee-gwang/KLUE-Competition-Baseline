import torch.cuda.amp as amp
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import tqdm
import argparse
import zipfile
import os
import cv2
import time
import pickle
# utils
from utils import load_data
from dataloader import KlueDataSet
from torch.utils.data import DataLoader

# transformer
import transformers
from transformers import XLMPreTrainedModel, XLMRobertaModel, XLMRobertaConfig, XLMRobertaTokenizer
from transformers import XLMRobertaForSequenceClassification, BertForSequenceClassification
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification, DistilBertForSequenceClassification, XLNetForSequenceClassification,\
XLMRobertaForSequenceClassification, XLMForSequenceClassification, RobertaForSequenceClassification
# ------------------------
#  Arguments
# ------------------------
parser = argparse.ArgumentParser(description='LG')
# setting
parser.add_argument('--debug', action='store_true', help='debugging mode')
parser.add_argument('--amp', action='store_true', help='mixed precision')
parser.add_argument('--gpu', type=str, default= '0,1', help='gpu')
parser.add_argument('--max_len', type=int, default= 33, help='max len')
parser.add_argument('--model_path', type=str, default= 'Your model path', help='saved model path')
parser.add_argument('--batch_size', type=int, default= 32, help='')
# model
parser.add_argument('--pt',  type=str, default= 'klue/roberta-base', help='huggingface models')
# else
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")


# ------------------------
#  Inference
# ------------------------
def do_predict(net, valid_loader):
    
    val_loss = 0
    pred_lst = []
    logit=[]
    net.eval()
    for t, data in enumerate(tqdm.tqdm(valid_loader)):
        ids  = data['ids'].to(device)
        mask  = data['mask'].to(device)
        tokentype = data['token_type_ids'].to(device)

        with torch.no_grad():
            if args.amp:
                with amp.autocast():
                    # output
                    output = net(ids, mask)[0]

            else:
                output = net(ids, mask)
             
            pred_lst.extend(output.argmax(dim=1).tolist())
            logit.extend(output.tolist())
            
    return pred_lst,logit

def run_predict(model_path):
    ## dataset ------------------------------------
    # load
    with open(f'./data/{args.pt}/test_data_{args.max_len}.pickle', 'rb') as f:
        test_dict = pickle.load(f)
        
    print('test load')
    test_dataset = KlueDataSet(data = test_dict, test=True)
    testloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, 
                             num_workers=8, shuffle=False, pin_memory=True)
    print('set testloader')
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

    f = torch.load(model_path)
    net.load_state_dict(f, strict=True)  # True
    print('load saved models')

    # predict
    preds, logit = do_predict(net, testloader) # If you want to ensemble, you can use logit.
    print('complete predict')

    # make submission
    sub = pd.read_csv("./data/sample_submission.csv")
    sub['topic_idx'] = preds
    sub.to_csv('./submission/final_submission.csv', index=False)
    print('make submission')
    
    return 
     
if __name__ == '__main__':
    run_predict(args.model_path)
