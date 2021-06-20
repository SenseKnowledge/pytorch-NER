# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
from zhner.utils import read_clue_json

tag2id = {'O': 0,
          'B-address': 1, 'I-address': 2,
          'B-book': 3, 'I-book': 4,
          'B-company': 5, 'I-company': 6,
          'B-game': 7, 'I-game': 8,
          'B-government': 9, 'I-government': 10,
          'B-movie': 11, 'I-movie': 12,
          'B-name': 13, 'I-name': 14,
          'B-organization': 15, 'I-organization': 16,
          'B-position': 17, 'I-position': 18,
          'B-scene': 19, 'I-scene': 20}

id2tag = {0: 'O',
          1: 'B-address', 2: 'I-address',
          3: 'B-book', 4: 'I-book',
          5: 'B-company', 6: 'I-company',
          7: 'B-game', 8: 'I-game',
          9: 'B-government', 10: 'I-government',
          11: 'B-movie', 12: 'I-movie',
          13: 'B-name', 14: 'I-name',
          15: 'B-organization', 16: 'I-organization',
          17: 'B-position', 18: 'I-position',
          19: 'B-scene', 20: 'I-scene'}


def decode_tags_from_ids(batch_ids):
    batch_tags = []
    for ids in batch_ids:
        sequence_tags = []
        for id in ids:
            sequence_tags.append(id2tag[int(id)])
        batch_tags.append(sequence_tags)
    return batch_tags


class CLUEDataset(Dataset):
    """Pytorch Dataset for CLUE
    """

    def __init__(self, path_to_clue, tokenizer):
        self.data = read_clue_json(path_to_clue)
        self.tokenizer = tokenizer

    def collate_fn(self, batch):
        """collate_fn for 'torch.utils.data.DataLoader'
        """
        texts, labels = list(zip(*[[item[0], item[1]] for item in batch]))
        token = self.tokenizer(list(texts), padding=False, return_offsets_mapping=True)

        # align the label
        # Bert mat split a word 'AA' into 'A' and '##A'
        labels = [self._align_label(offset, label) for offset, label in zip(token['offset_mapping'], labels)]
        token = self.tokenizer.pad(token, padding=True, return_attention_mask=True)

        return torch.LongTensor(token['input_ids']), torch.ByteTensor(token['attention_mask']), self._pad(labels)

    @staticmethod
    def _align_label(offset, label):

        label_align = []
        for i, (start, end) in enumerate(offset):

            if start == end:
                label_align.append(tag2id['O'])
            else:
                # 1-N or N-1, default to use first original label as final label
                if i > 0 and offset[i - 1] == (start, end):
                    label_align.append(label[start:end][0].replace('B', 'O', 1))
                else:
                    label_align.append(label[start:end][0])
        return label_align

    @staticmethod
    def _pad(labels):
        max_len = max([len(label) for label in labels])
        labels = [(label + [tag2id['O']] * (max_len - len(label))) for label in labels]
        return torch.LongTensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        pkg = self.data[index]

        text = pkg['text']
        label = [tag2id[tag] for tag in pkg["label"]]

        return text, label
