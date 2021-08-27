import pandas as pd
import os
import tqdm
import argparse
import numpy as np
import pickle
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
parser.add_argument('--pt', type=str, default= 'klue/roberta-base', help='pretrained model')
parser.add_argument('--max_len', type=int, default= 33, help='max len')
args = parser.parse_args()


def bert_tokenizer(sent, MAX_LEN, tokenizer):
    
    encoded_dict=tokenizer.encode_plus(
    text = sent, 
    add_special_tokens=True, 
    max_length=MAX_LEN, 
    pad_to_max_length=True, 
    return_attention_mask=True,
    truncation = True)
    
    input_id=encoded_dict['input_ids']
    attention_mask=encoded_dict['attention_mask']
    #token_type_id = encoded_dict['token_type_ids']
    token_type_id = 0
    
    return input_id, attention_mask, token_type_id

def preprocessing_train():
    
    pt = args.pt
    tokenizer = AutoTokenizer.from_pretrained(args.pt)
    
    MAX_LEN = args.max_len
    train = pd.read_csv('./data/train_data.csv')
    train=train[['title','topic_idx']]

    input_ids =[]
    attention_masks =[]
    token_type_ids =[]
    train_data_labels = []

    for train_sent, train_label in tqdm.tqdm(zip(train['title'], train['topic_idx'])):
        try:
            input_id, attention_mask,_ = bert_tokenizer(train_sent, MAX_LEN=MAX_LEN, tokenizer=tokenizer)

            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            token_type_ids.append(0)
            #########################################
            train_data_labels.append(train_label)

        except Exception as e:
            print(e)
            pass

    train_input_ids=np.array(input_ids, dtype=int)
    train_attention_masks=np.array(attention_masks, dtype=int)
    train_token_type_ids=np.array(token_type_ids, dtype=int)
    ###########################################################
    train_inputs=(train_input_ids, train_attention_masks, train_token_type_ids)
    train_labels=np.asarray(train_data_labels, dtype=np.int32)

    # save
    train_data = {}

    train_data['input_ids'] = train_input_ids
    train_data['attention_mask'] = train_attention_masks
    train_data['token_type_ids'] = train_token_type_ids
    train_data['targets'] = np.asarray(train_data_labels, dtype=np.int32)
    
    os.makedirs(f'./data/{pt}/', exist_ok=True)
    with open(f'./data/{pt}/train_data_{MAX_LEN}.pickle', 'wb') as f:
        pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)

def preprocessing_test():
    
    pt = args.pt
    tokenizer = AutoTokenizer.from_pretrained(args.pt)
    MAX_LEN = args.max_len
    
    test = pd.read_csv('./data/test_data.csv')
    test=test[['title']]
    
    input_ids =[]
    attention_masks =[]
    token_type_ids =[]

    for test_sent in tqdm.tqdm(test['title']):
        try:
            input_id, attention_mask,_ = bert_tokenizer(test_sent, MAX_LEN=MAX_LEN, tokenizer=tokenizer)

            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            token_type_ids.append(0)
            #########################################

        except Exception as e:
            print(e)
            pass

    test_input_ids=np.array(input_ids, dtype=int)
    test_attention_masks=np.array(attention_masks, dtype=int)
    test_token_type_ids=np.array(token_type_ids, dtype=int)
    ###########################################################
    test_inputs=(test_input_ids, test_attention_masks, test_token_type_ids)


    # save
    test_data = {}

    test_data['input_ids'] = test_input_ids
    test_data['attention_mask'] = test_attention_masks
    test_data['token_type_ids'] = test_token_type_ids
    
    os.makedirs(f'./data/{pt}/', exist_ok=True)
    with open(f'./data/{pt}/test_data_{MAX_LEN}.pickle', 'wb') as f:
        pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    preprocessing_train()
    preprocessing_test()