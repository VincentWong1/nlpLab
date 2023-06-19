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
        self.model_name = 'TextCNN'
        self.finetuning_task = finetuning_task
        self.use_word = args.use_word

        self.embedding_type = args.embedding_type
        self.embedding_path = os.path.join(args.model_name_or_path, args.embedding_type)
        self.class_list = [x.strip() for x in
                           open(os.path.join('dataset', self.finetuning_task, 'class.txt')).readlines()]     # 类别名单

        self.dropout = 0.5                                              # 随机失活
        self.num_labels = len(self.class_list)                          # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.filter_sizes = (2, 3, 4)                                    # 卷积核尺寸
        self.num_filters = 256                                           # 卷积核数量(channels数)

        self.n_num_feat = 0                                             # 数值型特征数量，在运行时赋值
        self.num_feat_size = 256                                        # 数值型特征fc层
        if self.use_word:
            self.tokenizer = lambda x: x.split(' ')                     # 以空格隔开，word-level
        else:
            self.tokenizer = lambda x: [y for y in x]                   # char-level


'''Convolutional Neural Networks for Sentence Classification'''


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
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)

        if config.n_num_feat != 0:
            self.fc_num = nn.Linear(config.n_num_feat, config.num_feat_size)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes) + config.num_feat_size
                            if config.n_num_feat != 0 else 0, config.num_labels)

        self.init_weights()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, input_ids, num=None, labels=None):
        out = self.embedding(input_ids)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)

        if num is not None:
            num_feat = self.fc_num(num)
            out = torch.cat([out, num_feat], 1)

        out = self.dropout(out)
        outputs = self.fc(out)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(outputs.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + (outputs,)

        return outputs
