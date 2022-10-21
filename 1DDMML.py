import numpy as np
np.set_printoptions(linewidth=400)
import functions as fs
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import time
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import random

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score


# 设置随机数种子
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

#多层全连接神经网络的嵌入表示（激活函数可选取）deepNeuralNetworkEmbedding
# input: input_size(数据维度)，hidden_struct(隐层到最后一层的神经元数 eg.[n_h1, n_o])
# output:维度为n_o的数据
class singeDNN_Embedding(nn.Module):
    def __init__(self, input_size, hidden_struct):
        super(singeDNN_Embedding, self).__init__()
        activate = nn.Tanh() #选取的激活函数
        self.embedNN = nn.Sequential()
        for i in range(len(hidden_struct)):
            if i == 0:
                self.embedNN.add_module('layer_{}'.format(i), nn.Linear(input_size, hidden_struct[i]))
            else:
                self.embedNN.add_module('layer_{}'.format(i), nn.Linear(hidden_struct[i-1], hidden_struct[i]))
            self.embedNN.add_module('activate_of_layer_{}'.format(i), activate)

    def forward(self, x):
        return self.embedNN(x)

#多个深度神经网络的模型（三个单深度神经网络，即由属性内、属性间、属性对类三类数据来训练）
class DDDML_Embedding(nn.Module):
    def __init__(self, input_size_Ia, input_size_Ie, input_size_AC, hidden_struct):
        super(DDDML_Embedding, self).__init__()
        self.inputIa = input_size_Ia
        self.inputIe = input_size_Ie
        self.model_Ia = singeDNN_Embedding(self.inputIa, hidden_struct)
        self.model_Ie = singeDNN_Embedding(self.inputIe, hidden_struct)
        self.model_AC = singeDNN_Embedding(input_size_AC, hidden_struct)

    def forward(self, data):
        x_Ia = data[:, :self.inputIa]
        x_Ie = data[:, self.inputIa: (self.inputIe+self.inputIa)]
        x_AC = data[:, (self.inputIe+self.inputIa):]
        return self.model_Ia(x_Ia), self.model_Ie(x_Ie), self.model_AC(x_AC)

#基于融合多个单个深度神经网络嵌入的多元组度量损失 multiView Deep Neural Network For Multi Tuple Loss
# multiTupleloss: Takes embeddings of an anchor sample, a positive sample and (n-1) negative samples
def subnet_loss(x_data, y_lable, paired_samples, tau = 3):
    Xi, Xj = paired_samples.T[0:2]
    lij = (y_lable[Xi] == y_lable[Xj]) * 2 - 1
    # print((x_data[Xi] - x_data[Xj])**2)
    df2 = torch.sum((x_data[Xi] - x_data[Xj])**2, -1)
    loss = torch.sum(F.softplus(1 - lij * (tau - df2)))
    return loss

def distance_difference(X1, X2):
    dis1 = F.pdist(X1, p=2)
    dis2 = F.pdist(X2, p=2)
    return torch.norm(dis1 - dis2)**2

def HSIC(X1, X2, device):
    N = X1.shape[0]
    K1 = torch.mm(X1, X1.T)
    if len(X2.shape)==1:
        K2 = (X2.unsqueeze(0) == X2.unsqueeze(1)).float()
    else:
        K2 = torch.mm(X2, X2.T)
    H = torch.eye(N) - torch.ones(N, N)/N
    if device:
        H = H.cuda()
    return torch.mm(torch.mm(torch.mm(K1, H), K2), H).trace() / ((N-1)*(N-1))

def DDDML_Loss(embeddings, y_batch, paired_samples, lambda2, tau):
    (embedding_Ia, embedding_Ie, embedding_AC) = embeddings
    lossJIa = subnet_loss(embedding_Ia, y_batch, paired_samples,tau)#属性内网络的多元组度量损失
    lossJIe = subnet_loss(embedding_Ie, y_batch, paired_samples,tau)#属性间网络的多元组度量损失
    lossJAC = subnet_loss(embedding_AC, y_batch, paired_samples,tau)#属性对类网络的多元组度量损失

    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    HSIC_IaIe = HSIC(embedding_Ia, embedding_Ie, device)
    HSIC_IaAC = HSIC(embedding_Ia, embedding_AC, device)
    HSIC_IeAC = HSIC(embedding_Ie, embedding_AC, device)

    loss_Ia = lossJIa + lambda2 * (HSIC_IaIe + HSIC_IaAC - HSIC(embedding_Ia, y_batch, device))
    loss_Ie = lossJIe + lambda2 * (HSIC_IaIe + HSIC_IeAC - HSIC(embedding_Ie, y_batch, device))
    loss_AC = lossJAC + lambda2 * (HSIC_IaAC + HSIC_IeAC - HSIC(embedding_AC, y_batch, device))

    weight = np.array([loss_Ia.item(), loss_Ie.item(), loss_AC.item()])
    # weight = np.array([lossJIa.item(), lossJIe.item(), lossJAC.item()])
    # print(weight)
    weight = (1.0/weight) / np.sum(1.0/weight)
    loss = weight[0]*lossJIa + weight[1]*lossJIe + weight[2]*lossJAC
    return loss, weight
    '''

    DD_IaIe = distance_difference(embedding_Ia, embedding_Ie)
    DD_IaAC = distance_difference(embedding_Ia, embedding_AC)
    DD_IeAC = distance_difference(embedding_Ie, embedding_AC)

    weight = np.array([lossJIa.item(), lossJIe.item(), lossJAC.item()])
    weight = (1.0/weight) / np.sum(1.0/weight)
    loss = weight[0]*lossJIa + weight[1]*lossJIe + weight[2]*lossJAC + lambda2 * (DD_IaIe + DD_IaAC + DD_IeAC)
    return loss, weight
    # '''


def build_classifier_forTripleView(x_train_tensor, y_train, x_test_tensor, y_test, model_TripleView, subnet_weight):
    with torch.no_grad():
        model.eval()
        model.cpu()
        x_train_embeddings = model_TripleView(x_train_tensor)
        x_test_embeddings = model_TripleView(x_test_tensor)
        x_train_embeddings = tuple(x_train_embeddings[i].numpy()*subnet_weight[i] for i in range(3) if subnet_weight[i]!=0)
        x_test_embeddings = tuple(x_test_embeddings[i].numpy()*subnet_weight[i] for i in range(3) if subnet_weight[i]!=0)
        x_train_embedding = np.hstack(x_train_embeddings)
        x_test_embedding = np.hstack(x_test_embeddings)

        knn_classifier = KNeighborsClassifier(n_neighbors=1)
        knn_classifier.fit(x_train_embedding, y_train)
        y_predict = knn_classifier.predict(x_test_embedding)
        score = f1_score(y_test, y_predict, average='micro')
        return score


#---------------------主程序-----------------------#
#---------------------------主程序入口---------------------------------#
result_file = "E:/实验与数据/multiView_Deep_Metric_Learning/result/DDMML.txt"
open(result_file,'w').close()
dataset_name = ["abalone","adult","anneal","audiology","automobile","breast-cancer","car","census","colic","credit-a","credit-g","crx","german","hayes-roth","heart-c","heart-h","heart-statlog","hepatitis","hypothyroid","kr-vs-kp","labor","lymphography","mushroom","nursery","nsl-kdd","post-operative","primary-tumor","promoter","sick","soybean","splice","vote","vowel","zoo"]

for i in range(34):#34  7, 8  7, 8,1, 2
    txtfile=open(result_file,'a+')
    print(dataset_name[i], end=',shape=')
    print(dataset_name[i], end=',shape=', file=txtfile)
    start = time.time()
    F1scores = []
    #获取属性内、属性间、属性对类三个视图的耦合数据及其维度，类标签
    Ia_data, Ie_data, AC_data, dimEmbed, Y_label = fs.coupleData(dataset_name[i])
    coupledData = np.c_[Ia_data, Ie_data, AC_data]
    print(dimEmbed)
    print(dimEmbed, file=txtfile)
    # print(Ia_data, Ie_data, AC_data, Y_label)

   #模型超参设置
    hidden_struct = [500, 90]                  # 隐层到输出层的神经元数
    num_classes = np.unique(Y_label)           #类标签个数
    num_epochs = 100                           # 训练轮数
    batch_size = 200                          # 一批数据个数
    learning_rate = 1.0e-3                     # 神经网络的学习率
    tau = 3
    lambda1 = 1.0e-3                           # 损失函数权衡参数参设置
    lambda2 = 1.0e-3                           # 不同子网络的关联性权衡参数
    scheduler_para = {"step":50,"gamma":0.95} #调整学习的参数：训练step轮后learning_rate*gamma
    setup_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run = 0
    for train_index, test_index in StratifiedShuffleSplit(n_splits=5,test_size=0.2,random_state=0).split(coupledData, Y_label):
        run += 1
        # 数据划分print(train_index.shape, train_index)
        x_train, y_train, x_test, y_test = coupledData[train_index], Y_label[train_index], coupledData[test_index], Y_label[test_index]
        input_size = x_train.shape[1]         # 输入数据的维度
        x_train_tensor, y_train_tensor, x_test_tensor = torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long(), torch.from_numpy(x_test).float()
        dataLoader = DataLoader(dataset=TensorDataset(x_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=False)

        model = DDDML_Embedding(dimEmbed["IaEmbed"],dimEmbed["IeEmbed"],dimEmbed["ACEmbed"], hidden_struct)
        # print(model)
        if device:
            model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lambda1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_para["step"], gamma=scheduler_para["gamma"] )
        model.train()

        loss_epoch = []
        for epoch in range(num_epochs):
            if device:
                model.cuda()
            loss_batch = []
            subnet_loss_batch = np.zeros(3)
            for i_batch, (x_batch, y_batch) in enumerate(dataLoader):
                paired_samples = torch.cat((torch.arange(y_batch.shape[0]).unsqueeze(1),
                                            torch.randint(y_batch.shape[0], size=(y_batch.shape[0],1))), dim=1)
                if device:
                    x_batch, y_batch, paired_samples = x_batch.cuda(), y_batch.cuda(), paired_samples.cuda()
                optimizer.zero_grad()
                embeddings = model(x_batch)
                # print(embeddings)
                # print(paired_samples)
                loss, weight = DDDML_Loss(embeddings, y_batch, paired_samples, lambda2, tau)
                loss_batch.append(loss.item())
                loss.backward()
                optimizer.step()
            scheduler.step()
            loss_epoch.append(np.mean(loss_batch))
            if epoch+1 in [60, 80, 100]:
        #         # print('n=%3d,     loss=%9.4f,     view_weight=%s,     Current_score=%7.4f%%'%(epoch+1, train_loss,loss_fn.weight,score*100), file=kwargs['txtfile'])
                print('epoch=%3d,  loss=%8.4f'%(epoch+1, loss_epoch[-1]),end=',  ')
        #         # for param in model.parameters(): print(param)
        #
                score = build_classifier_forTripleView(x_train_tensor   = x_train_tensor,
                                                       y_train          = y_train,
                                                       x_test_tensor    = x_test_tensor,
                                                       y_test           = y_test,
                                                       model_TripleView = model,
                                                       subnet_weight    = weight)
                if run == 1:
                    print("run_time:%11.4f,  F1_score: %7.4f"%(time.time()-start, score*100))
                    print("run_time:%11.4f,  F1_score: %7.4f"%(time.time()-start, score*100), end=",   ", file=txtfile)
                else:
                    print('%7.4f'%(score*100))
                    print('%7.4f'%(score*100), end=",   ", file=txtfile)
    print("-"*150,"\n")