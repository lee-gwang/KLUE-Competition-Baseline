from torch.optim.optimizer import Optimizer
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
        lr +=[ param_group['lr'] ]

    assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr

def load_data():
    train=pd.read_csv('./data/train_data.csv')
    test=pd.read_csv('./data/test_data.csv')
    
    #
    train=train[['title','topic_idx']]
    test=test[['title']]
    #
    
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    train['fold'] = -1
    for n_fold, (_,v_idx) in enumerate(skf.split(train, train['topic_idx'])):
        train.loc[v_idx, 'fold']  = n_fold
    train['id'] = [x for x in range(len(train))]
    
    return train, test
