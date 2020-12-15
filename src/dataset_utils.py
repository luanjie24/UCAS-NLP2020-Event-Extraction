# coding=utf-8
"""

"""
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, features, mode):
        self.nums = len(features)

        self.token_ids = [torch.tensor(example.token_ids).long() for example in features]
        self.attention_masks = [torch.tensor(example.attention_masks).float() for example in features]
        self.token_type_ids = [torch.tensor(example.token_type_ids).long() for example in features]

        self.labels = None
        # if mode == 'train':
        #     self.labels = [torch.tensor(example.labels) for example in features]
        self.labels = [torch.tensor(example.labels) for example in features]

    def __len__(self):
        return self.nums


class TriggerDataset(BaseDataset):
    def __init__(self,
                 features,
                 mode,
                 use_distant_trigger=False):
        super(TriggerDataset, self).__init__(features, mode)

        self.distant_trigger = None

        if use_distant_trigger:
            self.distant_trigger = [torch.tensor(example.distant_trigger_label).long()
                                    for example in features]

    def __getitem__(self, index):
        data = {'token_ids': self.token_ids[index],
                'attention_masks': self.attention_masks[index],
                'token_type_ids': self.token_type_ids[index],
                'labels':self.labels[index]}

        if self.distant_trigger is not None:
            data['distant_trigger'] = self.distant_trigger[index]

        if self.labels is not None:
            data['labels'] = self.labels[index]

        return data


class RoleDataset(BaseDataset):
    def __init__(self,
                 features,
                 mode,
                 use_trigger_distance=False):
        super(RoleDataset, self).__init__(features, mode)
        self.trigger_distance = None

        self.trigger_label = [torch.tensor(example.trigger_loc).long() for example in features]
        if use_trigger_distance:
            self.trigger_distance = [torch.tensor(example.trigger_distance).long()
                                     for example in features]

    def __getitem__(self, index):
        data = {'token_ids': self.token_ids[index],
                'attention_masks': self.attention_masks[index],
                'token_type_ids': self.token_type_ids[index],
                'trigger_index': self.trigger_label[index],
                'labels':self.labels[index]}

        # if self.trigger_distance is not None:
        #     data['trigger_distance'] = self.trigger_distance[index]

        # if self.labels is not None:
        #     data['labels'] = self.labels[index]

        return data


class AttributionDataset(BaseDataset):
    def __init__(self,
                 features,
                 mode):
        super(AttributionDataset, self).__init__(features, mode)

        self.trigger_label = [torch.tensor(example.trigger_loc).long() for example in features]

        self.pooling_masks = [torch.tensor(example.pooling_masks).float() for example in features]

    def __getitem__(self, index):
        data = {'token_ids': self.token_ids[index],
                'attention_masks': self.attention_masks[index],
                'token_type_ids': self.token_type_ids[index],
                'trigger_index': self.trigger_label[index],
                'pooling_masks': self.pooling_masks[index]}

        if self.labels is not None:
            data['labels'] = self.labels[index]

        return data


def build_dataset(num, features, mode):
    assert num in [1, 2,3], 'task mismatch'

    if num == 1:
        dataset = TriggerDataset(features, mode)
    elif num in [2, 3]:
        dataset = RoleDataset(features, mode)

    else:
        dataset = AttributionDataset(features, mode)

    return dataset
