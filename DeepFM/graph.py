import torch
import torch.nn as nn
from collections import OrderedDict


#fm->wide&deep->deepfm->dcn->xdeepfm
#fm->fnn->nfm->afm->autoInt
#fm->pnn()

class deepfm(nn.Module):

    def __init__(self, feat_size, sparse_feat_col, dense_feat_col, embedding_dim=4,
                 dnn_hidden_unit=[400,400,400], dnn_dropout_rate=0.2):
        """
        :param feat_size:           dict   数字特征中每个特征的类别数，稀疏特征为1{feat_name: value_len}
        :param sparse_feat_col:     list   稀疏特征名称列表
        :param dense_feat_col:      list   稠密特征列表
        :param embedding_dim:       int    输入embedding纬度
        :param dnn_hidden_unit      list   深层网络中每层神经元个数
        :param dnn_dropout_rate     float  神经元丢弃概率
        """
        super(deepfm, self).__init__()
        self.feat_size = feat_size
        self.sparse_feat_col = sparse_feat_col
        self.dense_feat_col = dense_feat_col
        self.embedding_dim = embedding_dim
        self.dnn_hidden_unit = dnn_hidden_unit
        self.dropout_rate = dnn_dropout_rate

        """
        输入层
        """
        self.feat_index = self.build_input_feature(self.feat_size)
        self.sparse_embedding_dict = self.create_embedding_matrix(self.sparse_feat_col, self.feat_size,
                                                                     embedding_dim=embedding_dim)
        """
        wide模块参数初始化
        logit_wide = w0+wixi, i=1,2,..n
        """
        #线性部分  logit1=W0+w1x1+w2x2

        #稀疏特征：wi        if i in sparse_feat_col   ->w的求和
        #稠密特征：w0+wixi   if i in dense_feat_col
        self.sparse_col_w = self.create_embedding_matrix(self.sparse_feat_col, self.feat_size,embedding_dim=1)
        self.linear_dense = nn.Linear(len(self.dense_feat_col), 1, bias=True)

        """
        dnn模块参数初始化
        """
        self.dropout_dnn = nn.Dropout(self.dropout_rate)
        self.dnn_input_size = len(self.sparse_feat_col)*self.embedding_dim + len(dense_feat_col)

        hidden_units = [self.dnn_input_size] + dnn_hidden_unit

        self.linears_dnn = nn.ModuleList(
            [nn.Linear(hidden_units[i],hidden_units[i+1]) for i in range(len(hidden_units)-1)]
        )

        self.batchNorms_dnn = nn.ModuleList(
            [nn.BatchNorm1d(num) for num in self.dnn_hidden_unit]
        )
        self.relu = nn.ReLU()
        self.dnn_linear = nn.Linear(self.dnn_hidden_unit[-1], 1, bias=False)

    def forward(self,x):
        """
        :param x:         x:tensor([batch, feat_value])
        :return:          y_pred
        """

        # wide部分 广义线性回归

        # 线性部分
        # logit1=W0+wixi  i=1,2...n
        # x[:,self.feat_index[feat_name][0]:self.feat_index[feat_name][1]] 取出特征对应的value [batch, 1]
        # 稀疏特征通过value去对应的embeeding取出向量放入列表 [batch, feat_sparse_num, embedding_size],自动具备01功能
        # 稠密特征取出对应的value放入列表即可               [batch, feat_dense_num]

        sparse_w = [self.sparse_col_w[feat_name](
            x[:,self.feat_index[feat_name][0]:self.feat_index[feat_name][1]].long() #[batch, 1, embedding_dim]
        ) for feat_name in self.sparse_feat_col
        ]
        sparse_w = torch.cat(sparse_w, dim=1)   #[batch, feat_sparse_num, embedding_dim]

        dense_intput = [x[:,self.feat_index[feat_name][0]:self.feat_index[feat_name][1]] ##[batch, 1]
                            for feat_name in self.dense_feat_col]
        dense_intput = torch.cat(dense_intput,dim=-1)                     #[batch, feat_dense_num]


        #稀疏特征，某一特征不同的value取出不同的向量，自带01功能
        #logit1_sparse = wixi , i=1,2,... if i in sparse_feat_col
        #logit1_sparse = wi   , i=1,2,...if i in sparse_feat_col
        sparse_linaer_logit = torch.sum(sparse_w,dim=1, keepdim=False) #[batch, 1]

        #稠密特征
        #logit1_dense = w0+ wixi, i=1,2,...   if i in dense_feat_col
        dense_linear_logit = self.linear_dense(dense_intput)

        logits = sparse_linaer_logit + dense_linear_logit


        #二阶特征交叉部分
        sparse_embedding_input = [self.sparse_embedding_dict[feat_name](
            x[:,self.feat_index[feat_name][0]:self.feat_index[feat_name][1]].long() #[batch, 1, embedding_dim]
        ) for feat_name in self.sparse_feat_col
        ]
        sparse_embedding_input = torch.cat(sparse_embedding_input, dim=1)   #[batch, feat_sparse_num, embedding_dim]

        square_of_sum = torch.pow(torch.sum(sparse_embedding_input, dim=1, keepdim=True), 2)  # [batch,1,embedding_dim]
        sum_of_square = torch.sum(torch.pow(sparse_embedding_input, 2), dim=1, keepdim=True)  # [batch,1,embedding_dim]
        cross_term = square_of_sum - sum_of_square
        fm_secon_logit = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)  # [batch,1]
        logits += fm_secon_logit


        #deep模块
        # 将输入打平
        batch,_,_ = sparse_embedding_input.shape
        sparse_embedding_input = sparse_embedding_input.view(batch, -1)    #[batch, feat_sparse_num*embedding_dim]
        dnn_input = torch.cat([sparse_embedding_input, dense_intput], dim=-1)#[batch, feat_sparse_num*embedding_dim+dense_feat_num]

        for i in range(len(self.dnn_hidden_unit)):
            fc = self.linears_dnn[i](dnn_input)
            fc = self.batchNorms_dnn[i](fc)
            fc = self.relu(fc)
            fc = self.dropout_dnn(fc)
            dnn_input = fc
        dnn_output = self.dnn_linear(dnn_input)

        logits += dnn_output


        return torch.sigmoid(logits.squeeze())   ##[batch]


    def build_input_feature(self, feat_size):
        """
        默认feat_size字典排序列表为输入特征的顺序
        :param feat_size:  dict   数字特征中每个特征的类别数，稀疏特征为1 {feat_name: value_len}
        :return:           dict   特征索引 {feat_name:(start, end)} {"I1":(0,1)}
        """
        feat_index = OrderedDict()
        start = 0

        for feat_name in feat_size.keys():
            feat_index[feat_name] = (start, start+1)
            start += 1

        return feat_index


    def create_embedding_matrix(self, sparse_feat_col, feat_size, embedding_dim):
        """
        :param sparse_feat_col:         list   稀疏特征名称列表
        :param feat_size:               dict   数字特征中每个特征的类别数，稀疏特征为1{feat_name: value_len}
        :param embedding_dim:           int    输入embedding纬度
        :param init_std:                float  初始化方差
        :return:                        dict   {feat_name:nn.Embedding[feat_name_len, embedding_dim]}
        """
        #创建embedding
        embedding_dict = nn.ModuleDict(
            {feat_name:nn.Embedding(feat_size[feat_name], embedding_dim) for feat_name in sparse_feat_col}
        )

        #embedding初始化
        for tensor in embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=0.0001)

        return embedding_dict