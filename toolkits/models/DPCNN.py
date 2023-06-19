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
        self.model_name = 'DPCNN'
        self.finetuning_task = finetuning_task
        self.use_word = args.use_word

        self.embedding_type = args.embedding_type
        self.embedding_path = os.path.join(args.model_name_or_path, args.embedding_type)
        self.class_list = [x.strip() for x in
                           open(os.path.join('dataset', self.finetuning_task, 'class.txt')).readlines()]  # 类别名单

        self.num_labels = len(self.class_list)                          # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_filters = 250                                           # 卷积核数量(channels数)
        if self.use_word:
            self.tokenizer = lambda x: x.split(' ')                     # 以空格隔开，word-level
        else:
            self.tokenizer = lambda x: [y for y in x]                   # char-level


'''Deep Pyramid Convolutional Neural Networks for Text Categorization'''


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
        self.conv_region = nn.Conv2d(1, config.num_filters, (3, embed), stride=1)
        self.conv = nn.Conv2d(config.num_filters, config.num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(config.num_filters, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids)
        x = x.unsqueeze(1)  # [batch_size, 250, seq_len, 1]
        x = self.conv_region(x)  # [batch_size, 250, seq_len-3+1, 1]

        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        while x.size()[2] > 2:
            x = self._block(x)
        x = x.squeeze()  # [batch_size, num_filters(250)]
        outputs = self.fc(x)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(outputs.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + (outputs,)

        return outputs

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x
