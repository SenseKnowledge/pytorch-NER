# -*- coding: utf-8 -*-
from torch import nn
from transformers import BertModel, BertTokenizerFast, BertConfig
from package.nn import ConditionalRandomField

import torch


class RoBERTa_BiLSTM_CRF(nn.Module):
    """RoBERTa + BiLSTM + CRF
    """

    def __init__(self, path_to_roberta, tag_set_size, lstm_hidden_dim=256, lstm_dropout_rate=0.1, freeze_roberta=True):
        super(RoBERTa_BiLSTM_CRF, self).__init__()

        model = BertModel.from_pretrained(path_to_roberta)

        if freeze_roberta:
            for param in model.parameters():
                param.requires_grad = False

        self.roberta = model

        self.tokenizer = BertTokenizerFast.from_pretrained(path_to_roberta)
        roberta_config = BertConfig.from_pretrained(path_to_roberta)

        self.bilstm = nn.LSTM(roberta_config.hidden_size, lstm_hidden_dim // 2,
                              num_layers=2, bidirectional=True, dropout=lstm_dropout_rate, batch_first=True)
        self.hidden2tag = nn.Linear(lstm_hidden_dim, tag_set_size)
        self.crf = ConditionalRandomField(tag_set_size)

        self.lstm_hidden_dim = lstm_hidden_dim

    def reset_parameters(self):
        self.crf.reset_parameters()

    def forward(self, input: torch.LongTensor, mask: torch.ByteTensor):

        x = self.roberta(input_ids=input, attention_mask=mask)[0]
        x, _ = self.bilstm(x)
        x = self.hidden2tag(x)
        return self.crf(x, mask)

    def loss(self, input: torch.LongTensor, mask: torch.ByteTensor, target: torch.LongTensor):

        x = self.roberta(input_ids=input, attention_mask=mask)[0]
        x, _ = self.bilstm(x)
        x = self.hidden2tag(x)
        return self.crf.neg_log_likelihood_loss(x, mask, target)
