# coding: UTF-8
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
from models.BaseModel import BaseModel
from pytorch_pretrained.configuration_utils import PretrainedConfig


class Config(PretrainedConfig):

    """配置参数"""
    def __init__(self, args, finetuning_task, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.model_name = 'TextRCNN'
        self.finetuning_task = finetuning_task
        self.use_word = args.use_word

        self.embedding_type = args.embedding_type
        self.embedding_path = os.path.join(args.model_name_or_path, args.embedding_type)
        self.class_list = [x.strip() for x in
                           open(os.path.join('dataset', self.finetuning_task, 'class.txt')).readlines()]  # 类别名单

        self.dropout = 1.0                                              # 随机失活
        self.num_labels = len(self.class_list)                          # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.pad_size = args.max_seq_length                             # 每句话处理成的长度(短填长切)
        self.hidden_size = 256                                          # 隐藏层大小
        self.num_layers = 1                                             # lstm层数
        if self.use_word:
            self.tokenizer = lambda x: x.split(' ')                     # 以空格隔开，word-level
        else:
            self.tokenizer = lambda x: [y for y in x]                   # char-level


'''Recurrent Convolutional Neural Networks for Text Classification'''


class Model(BaseModel):
    def __init__(self, config):
        super(Model, self).__init__(config)

        if config.embedding_type != 'random':
            embedding_pretrained = torch.tensor(np.load(config.embedding_path)["embeddings"].astype('float32'))
            embed = embedding_pretrained.size(1)
            self.embedding = nn.Embedding.from_pretrained(embedding_pretrained, freeze=False)
        else:
            embed = 300
            self.embedding = nn.Embedding(config.n_vocab, embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.hidden_size * 2 + embed, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, labels=None):
        embed = self.embedding(input_ids)  # [batch_size, seq_len, embeding]=[64, 32, 64]
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        outputs = self.fc(out)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(outputs.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + (outputs,)

        return outputs
