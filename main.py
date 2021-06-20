import torch

from zhner.model import RoBERTa_BiLSTM_CRF
from zhner.dataset import CLUEDataset, DataLoader, id2tag, decode_tags_from_ids
from zhner.metrics import Score
from torch.optim import Adam
from tqdm import tqdm
from pprint import pprint

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
lr = 1e-3
batch_size = 128
accumulation_steps = 1
roberta_path = 'resource/RoBERTa_zh_Large_PyTorch'

lstm_hidden_dim = 768
lstm_dropout_rate = 0.1

model = RoBERTa_BiLSTM_CRF(roberta_path, len(id2tag),
                           lstm_hidden_dim=lstm_hidden_dim, lstm_dropout_rate=lstm_dropout_rate).to(device)
model.reset_parameters()
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

dataset = CLUEDataset('clue/train.json', model.tokenizer)
dataloader_train = DataLoader(dataset, collate_fn=dataset.collate_fn,
                              batch_size=batch_size, shuffle=True, drop_last=True)
dataset = CLUEDataset('clue/dev.json', model.tokenizer)
dataloader_valid = DataLoader(dataset, collate_fn=dataset.collate_fn,
                              batch_size=batch_size, shuffle=False, drop_last=False)

for _ in range(500):
    model.train()
    with tqdm(desc='Train', total=len(dataloader_train)) as t:
        for i, (input, mask, label) in enumerate(dataloader_train):
            input, mask, label = [_.to(device) for _ in (input, mask, label)]
            loss = model.loss(input, mask, label)
            loss.backward()
            t.update(1)
            t.set_postfix(loss=float(loss))

            if (1 + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

    model.eval()
    with torch.no_grad():
        score = Score()

        for i, (input, mask, label) in enumerate(tqdm(dataloader_valid, desc='Test')):
            input, mask, label = [_.to(device) for _ in (input, mask, label)]
            y_pred = model(input, mask)

            y_pred = decode_tags_from_ids(y_pred)
            y_true = decode_tags_from_ids(label)

            score.update(y_pred, y_true)

        pprint(score.compute())
