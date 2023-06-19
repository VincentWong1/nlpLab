# coding: UTF-8
import os
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np
from models.BaseModel import BaseModel
from pytorch_pretrained.configuration_utils import PretrainedConfig


class Config(PretrainedConfig):

    """配置参数"""
    def __init__(self, args, finetuning_task, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.model_name = 'TextRNN'
        self.finetuning_task = finetuning_task
        self.use_word = args.use_word

        self.embedding_type = args.embedding_type
        self.embedding_path = os.path.join(args.model_name_or_path, args.embedding_type)
        self.class_list = [x.strip() for x in
                           open(os.path.join('dataset', self.finetuning_task, 'class.txt')).readlines()]  # 类别名单

        self.dropout = 0.5                                              # 随机失活
        self.num_labels = len(self.class_list)                          # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.pad_size = args.max_seq_length                             # 每句话处理成的长度(短填长切)
        self.hidden_size = 128                                          # 隐藏层大小
        self.num_layers = 2                                             # lstm层数
        if self.use_word:
            self.tokenizer = lambda x: x.split(' ')                     # 以空格隔开，word-level
        else:
            self.tokenizer = lambda x: [y for y in x]                   # char-level


'''Recurrent Neural Network for Text Classification with Multi-Task Learning'''


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
        self.fc = nn.Linear(config.hidden_size * 2, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, labels=None):
        out = self.embedding(input_ids)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        out, _ = self.lstm(out)
        outputs = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(outputs.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + (outputs,)

        return outputs

    '''变长RNN，效果差不多，甚至还低了点...'''
    # def forward(self, x):
    #     x, seq_len = x
    #     out = self.embedding(x)
    #     _, idx_sort = torch.sort(seq_len, dim=0, descending=True)  # 长度从长到短排序（index）
    #     _, idx_unsort = torch.sort(idx_sort)  # 排序后，原序列的 index
    #     out = torch.index_select(out, 0, idx_sort)
    #     seq_len = list(seq_len[idx_sort])
    #     out = nn.utils.rnn.pack_padded_sequence(out, seq_len, batch_first=True)
    #     # [batche_size, seq_len, num_directions * hidden_size]
    #     out, (hn, _) = self.lstm(out)
    #     out = torch.cat((hn[2], hn[3]), -1)
    #     # out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
    #     out = out.index_select(0, idx_unsort)
    #     out = self.fc(out)
    #     return out
