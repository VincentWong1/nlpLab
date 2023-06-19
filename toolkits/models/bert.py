# coding: UTF-8
import os
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from pytorch_pretrained import BertModel, BertTokenizer, BertConfig
from pytorch_pretrained.file_utils import CONFIG_NAME


class Config(BertConfig):

    """配置参数"""
    def __init__(self, args, finetuning_task, **kwargs):
        super(Config, self).__init__(os.path.join(args.model_name_or_path, CONFIG_NAME))
        self.model_name = 'bert'
        self.finetuning_task = finetuning_task
        self.bert_path = args.model_name_or_path

        self.embedding_type = args.embedding_type
        self.embedding_path = os.path.join(args.model_name_or_path, args.embedding_type)
        self.class_list = [x.strip() for x in
                           open(os.path.join('dataset', self.finetuning_task, 'class.txt')).readlines()]  # 类别名单

        self.num_labels = len(self.class_list)                          # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.dropout = 0.5                                              # 随机失活
        self.hidden_size = 768

        self.tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)


class Model(BertModel):

    def __init__(self, config):
        super(Model, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, labels=None):
        _, pooled = self.bert(input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              output_all_encoded_layers=False)
        outputs = self.fc(pooled)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(outputs.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(outputs.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + (outputs, )

        return outputs
