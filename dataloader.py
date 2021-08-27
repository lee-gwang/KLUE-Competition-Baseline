import torch
from torch.utils.data.dataset import Dataset
# ------------------------
#  Datasets
# ------------------------
class KlueDataSet(Dataset):
    
    def __init__(self, data, test=False):
        
        self.data = data
        self.test = test
        
    def __len__(self):
        
        return self.data['input_ids'].shape[0]
    
    def __getitem__(self,idx):
        
        ids = torch.tensor(self.data['input_ids'][idx], dtype=torch.long)
        mask = torch.tensor(self.data['attention_mask'][idx], dtype=torch.long)
        token_type_ids = torch.tensor(self.data['token_type_ids'][idx], dtype=torch.long)
         
            
        if self.test:
            return {
                'ids': ids,
                'mask': mask,
                'token_type_ids': token_type_ids
            }
        
        else:
            target = torch.tensor(self.data['targets'][idx],dtype=torch.long)

            return {
                    'ids': ids,
                    'mask': mask,
                    'token_type_ids': token_type_ids,
                    'targets': target
                }