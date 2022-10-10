import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


site = 'KDM'
data_path = './data/{site}/'.format(site=site)

class MySet(Dataset):
    def __init__(self, file_name):
        super(MySet, self).__init__()
        self.content = open(os.path.join(data_path, file_name)).readlines()
        rec = json.loads(self.content[0])
        forward_seq = rec['forward']
        bkward_seq = rec['backward']
        indices = np.arange(len(self.content))
        val_indices = np.random.choice(indices, len(self.content) // 5)

        self.val_indices = set(val_indices.tolist())

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        rec = json.loads(self.content[idx])
        if idx in self.val_indices:
            rec['is_train'] = 0
        else:
            rec['is_train'] = 1
        return rec

def collate_fn(recs):
    forward = list(map(lambda x: x['forward'], recs))
    backward = list(map(lambda x: x['backward'], recs))

    def to_tensor_dict(recs):
        values = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['values'], r)), recs)))
        masks = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['masks'], r)), recs)))
        deltas = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['deltas'], r)), recs)))
        forwards = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['forwards'], r)), recs)))
        evals = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['evals'], r)), recs)))
        eval_masks = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['eval_masks'], r)), recs)))
        return {'values': values, 'forwards': forwards, 'masks': masks, 'deltas': deltas, 'evals':evals, 'eval_masks':eval_masks}

    ret_dict = {'forward': to_tensor_dict(forward), 'backward': to_tensor_dict(backward)}

    ret_dict['labels'] = torch.FloatTensor(list(map(lambda x: x['label'], recs)))
    ret_dict['masks_y'] = torch.FloatTensor(list(map(lambda x: x['masks_y'], recs)))

    ret_dict['delta_y'] = torch.FloatTensor(list(map(lambda x: x['delta_y'], recs)))
    ret_dict['eval_masks_y'] = torch.FloatTensor(list(map(lambda x: x['eval_masks_y'], recs)))

    ret_dict['eval_label'] = torch.FloatTensor(list(map(lambda x: x['eval_label'], recs)))
    return ret_dict

def get_loader(batch_size = 64, method='cust', thre='0.1', shuffle = True):
    data_set = MySet('wifi_biseq_{method}_{thre}.json'.format(method=method, thre=thre))
    data_iter = DataLoader(dataset = data_set, \
                              batch_size = batch_size, \
                              num_workers = 4, \
                              shuffle = shuffle, \
                              pin_memory = True, \
                              collate_fn = collate_fn
    )

    return data_iter
def get_val_loader(batch_size = 64, shuffle = True):
    val_data_set = MySet('wifi_biseq_val.json')
    val_data_iter = DataLoader(dataset = val_data_set, \
                              batch_size = batch_size, \
                              num_workers = 4, \
                              shuffle = shuffle, \
                              pin_memory = True, \
                              collate_fn = collate_fn
    )
    return val_data_iter
