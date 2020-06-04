import torch
import torch.nn as nn
import torch.nn.functional as F

class MathDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.input_ids = dataset['input_ids']
        self.attention_mask = dataset['attention_mask']
        self.token_type_ids = dataset['token_type_ids']

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        token_type_ids = self.token_type_ids[idx]
        item = MathDataset({'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
                })
        return item
    def __len__(self):
        return len(self.input_ids)
