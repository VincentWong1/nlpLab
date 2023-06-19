import pandas as pd
import torch

import torch.utils.data as Data
import torch.nn as nn
import time

from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#导入模型
from model.FM import fm
from model.wide_deep import wide_deep
from model.deepfm import deepfm

if __name__ == '__main__':
    #导入数据
    data_path = "../data/samples.csv"
    data = pd.read_csv(data_path)
    print(data.columns)

    #特征列
    sparse_feat_col = ["C" + str(i) for i in range(1,27)]  #稀疏特征
    dense_feat_col =["I" + str(i) for i in range(1,14)]    #稠密特征

    #统计每个特征的数目（稀疏特征为类别数，稠密特征为1）
    dense_feat_dict = {feat : 1 for feat in dense_feat_col}
    sparse_feat_dict = {feat:len(data[feat].unique()) for feat in sparse_feat_col}
    feat_size = {}
    feat_size.update(dense_feat_dict)
    feat_size.update(sparse_feat_dict)
    print(feat_size)

    #划分数据集
    test_size = 0.1
    train, test = train_test_split(data, test_size=test_size, random_state=0)
    del data

    #构造数据加载器
    train_x = torch.from_numpy(train.iloc[:,1:-2].to_numpy())
    train_y = torch.from_numpy(train.iloc[:,0].to_numpy())
    test_x = torch.from_numpy(test.iloc[:,1:-2].to_numpy())
    test_y = torch.from_numpy(test.iloc[:,0].to_numpy())

    train_tensor_data = Data.TensorDataset(train_x, train_y)
    test_tensor_data = Data.TensorDataset(test_x, test_y)
    del train_x, train_y, test_x, test_y

    batch_size_train = 50
    batch_size_test = 1000
    trainLoader = DataLoader(dataset=train_tensor_data, batch_size=batch_size_train, shuffle=True)
    testLoader = DataLoader(dataset=test_tensor_data, batch_size=batch_size_test, shuffle=True)
    del train_tensor_data, test_tensor_data

    #模型训练前准备
    epochs = 10
    print_batch_num = 10

    learing_rate = 0.001
    print("cuda is available:", torch.cuda.is_available())
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 损失函数和模型放在gpu上
    # model = fm(feat_size, sparse_feat_col, dense_feat_col, embedding_dim=4).to(device)
    #model = wide_deep(feat_size,sparse_feat_col,dense_feat_col,embedding_dim=4).to(device)
    model = deepfm(feat_size, sparse_feat_col, dense_feat_col, embedding_dim=4).to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = Adam(model.parameters(), lr=learing_rate, weight_decay=0.1)

    start_time = time.time()
    beatAcc = 0.0

    #开始训练
    for epoch in range(epochs):
        print("*******epochs:{}/{} beging*****".format(epoch+1, epochs))
        loss_train_list = []
        for num, (batch_x, batch_y) in enumerate(trainLoader):

            model.train()
            #数据放在gpu上
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            y_pred = model(batch_x)
            loss = criterion(y_pred, batch_y)

            #反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train_list.append(loss.cpu().item())

            #测试部分
            if((num+1)%print_batch_num == 0):
                model.eval()

                loss_train_avg = sum(loss_train_list)/len(loss_train_list) #print_num_bacth个batch训练样本的平均损失
                loss_train_list = []                                 #置空训练集损失集合

                loss_test_list = []      #测试集损失列表
                pred_list = []           #测试集预测集合
                label_list = []          #测试集标签集合

                for test_x, test_y in testLoader:
                    #放gpu上
                    test_x = test_x.float().to(device)
                    test_y = test_y.float().to(device)
                    pred_test = model(test_x)
                    loss_test = criterion(pred_test, test_y)

                    loss_test_list.append(loss_test.cpu().item())
                    label_list.extend(list(test_y.cpu()))

                    # 将预测概率转0 1
                    zeros = torch.zeros_like(pred_test.cpu())
                    ones = torch.ones_like(pred_test.cpu())
                    pred_label = torch.where(pred_test.cpu()>=0.5, ones, zeros)
                    pred_list.extend(pred_label)

                loss_test_avg = sum(loss_test_list)/len(loss_test_list)
                acc_test = accuracy_score(label_list, pred_list)
                print("epoch:{}/{}, bacth:{}/{}:train loss:{:.5f}, test loss:{:.5f}, test acc:{:.5f},耗时:{:.2f}s".format(
                    epoch+1, epochs, num+1, len(trainLoader), loss_train_avg, loss_test_avg, acc_test, time.time()-start_time
                ))

                #验证集效果最好的保存
                if(acc_test>beatAcc):
                    beatAcc = acc_test
                    print("当前训练周期取得最佳效果：{}".format(beatAcc))
                    #保存模型
                    #torch.save(model.cpu(),path = ....)
                start_time = time.time()   #重新开始计时

    print("训练结束，当前模型测试集拟合最佳效果为:{:.5f}".format(beatAcc))