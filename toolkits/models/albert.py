# coding: UTF-8
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

import os
import json
from io import open

from pytorch_pretrained.file_utils import CONFIG_NAME
from pytorch_pretrained.modeling_albert import PretrainedConfig, FullTokenizer, AlbertPreTrainedModel, AlbertModel


class Config(PretrainedConfig):
    def __init__(self,
                 args,
                 do_lower_case=True,
                 vocab_size_or_config_json_file=30000,
                 embedding_size=128,
                 hidden_size=4096,
                 num_hidden_layers=12,
                 num_hidden_groups=1,
                 num_attention_heads=64,
                 intermediate_size=16384,
                 inner_group_num=1,
                 hidden_act="gelu_new",
                 hidden_dropout_prob=0,
                 attention_probs_dropout_prob=0,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 **kwargs):
        super(Config, self).__init__(**kwargs)

        # initialization
        self.vocab_size = vocab_size_or_config_json_file
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.embedding_size = embedding_size
        self.inner_group_num = inner_group_num
        self.num_hidden_groups = num_hidden_groups

        # read config file
        with open(os.path.join(args.model_name_or_path, CONFIG_NAME), "r", encoding='utf-8') as reader:
            json_config = json.loads(reader.read())
        for key, value in json_config.items():
            self.__dict__[key] = value

        # set custom config
        self.model_name = 'albert'
        self.class_list = [x.strip() for x in
                           open(os.path.join('dataset', self.finetuning_task, 'class.txt')).readlines()]  # 类别名单

        self.num_labels = len(self.class_list)                          # 类别数
        self.n_num_feat = 0                                             # 数值型特征数量，在运行时赋值
        self.num_feat_size1 = 64                                        # 数值型特征fc层
        self.num_feat_size2 = 32                                        # 数值型特征fc层
        self.tokenizer = FullTokenizer(args.model_name_or_path, do_lower_case)


class Model(AlbertPreTrainedModel):

    def __init__(self, config):
        super(Model, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = AlbertModel(config)
        # for param in self.bert.parameters():
        #     param.requires_grad = True
        self.dropout = nn.Dropout(0.1 if config.hidden_dropout_prob == 0 else config.hidden_dropout_prob)
        if config.n_num_feat != 0:
            self.fc_num = nn.Sequential(
                nn.Linear(config.n_num_feat, config.num_feat_size1),
                nn.BatchNorm1d(config.num_feat_size1),
                nn.Tanh(),
                # nn.ReLU(inplace=True),
                nn.Linear(config.num_feat_size1, config.num_feat_size2),
                # nn.ReLU(inplace=True),
                nn.Tanh(),
            )
        self.classifier = nn.Linear(config.hidden_size + config.num_feat_size2 if config.n_num_feat != 0 else 0,
                                    self.config.num_labels)

        self.init_weights()

    # def forward(self, input_ids, attention_mask=None, token_type_ids=None,
    #             position_ids=None, head_mask=None, num=None, labels=None):
    #
    #     outputs = self.bert(input_ids,
    #                         attention_mask=attention_mask,
    #                         token_type_ids=token_type_ids,
    #                         position_ids=position_ids,
    #                         head_mask=head_mask)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, num=None):

        labels = None
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids)

        pooled_output = outputs[1]

        if num is not None:
            num_feat = self.fc_num(num)
            pooled_output = torch.cat([pooled_output, num_feat], 1)

        pooled_output = self.dropout(pooled_output+0.1)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
