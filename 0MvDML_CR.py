import numpy as np
np.set_printoptions(linewidth=400,suppress = True)
import functions as fs
from sklearn.model_selection import StratifiedShuffleSplit
import time
import torch
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
    def __init__(self, input_size, hidden_struct, dropout=0.05):
        super(singeDNN_Embedding, self).__init__()
        activate = nn.Tanh() #选取的激活函数
        self.embedNN = nn.Sequential()
        for i in range(len(hidden_struct)):
            if i == 0:
                self.embedNN.add_module('layer_{}'.format(i), nn.Linear(input_size, hidden_struct[i]))
            else:
                self.embedNN.add_module('layer_{}'.format(i), nn.Linear(hidden_struct[i-1], hidden_struct[i]))
            self.embedNN.add_module('activate_of_layer_{}'.format(i), activate)
            self.embedNN.add_module('layer_{}dropout'.format(i), nn.Dropout(dropout))

    def forward(self, x):
        return self.embedNN(x)

#多个深度神经网络的模型（三个单深度神经网络，即由属性内、属性间、属性对类三类数据来训练）
class DDDML_Embedding(nn.Module):
    def __init__(self, input_size_Ia, input_size_Ie, input_size_AC, hidden_struct, dropout):
        super(DDDML_Embedding, self).__init__()
        self.inputIa = input_size_Ia
        self.inputIe = input_size_Ie
        self.model_Ia = singeDNN_Embedding(self.inputIa, hidden_struct, dropout)
        self.model_Ie = singeDNN_Embedding(self.inputIe, hidden_struct, dropout)
        self.model_AC = singeDNN_Embedding(input_size_AC, hidden_struct, dropout)

    def forward(self, data):
        x_Ia = data[:, :self.inputIa]
        x_Ie = data[:, self.inputIa: (self.inputIe+self.inputIa)]
        x_AC = data[:, (self.inputIe+self.inputIa):]
        return self.model_Ia(x_Ia), self.model_Ie(x_Ie), self.model_AC(x_AC)


#基于融合多个单个深度神经网络嵌入的多元组度量损失 multiView Deep Neural Network For Multi Tuple Loss
# multiTupleloss: Takes embeddings of an anchor sample, a positive sample and (n-1) negative samples
class Proxy_DDDMLloss(torch.nn.Module):
    def __init__(self, num_classes, num_embed, lambda2=1.0e-3, mu=1, tau=3):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(3, num_classes, num_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.num_classes = num_classes
        self.num_embed = num_embed
        self.lambda2 = lambda2
        self.mu = mu
        self.tau = tau

    def subnet_loss(self, embed_X, y, proxies):
        dis = torch.cdist(embed_X, proxies, p=2)**2  #计算平方欧氏距离
        P_one_hot = torch.eye(self.num_classes)[y].cuda()
        N_one_hot = 1 - P_one_hot

        margin = P_one_hot * (1 - self.tau + dis) + N_one_hot * (1 + self.tau - dis)
        margin_weight = F.softplus(margin) * (P_one_hot + N_one_hot / (self.num_classes-1))
        loss = torch.sum(margin_weight)
        return loss

    def HSIC(self, X1, X2):
        N = X1.shape[0]
        K1 = torch.mm(X1, X1.T)
        if len(X2.shape)==1:
            K2 = (X2.unsqueeze(0) == X2.unsqueeze(1)).float()
        else:
            K2 = torch.mm(X2, X2.T)
        H = (torch.eye(N) - torch.ones(N, N)/N).cuda()
        return torch.mm(torch.mm(torch.mm(K1, H), K2), H).trace() / ((N-1)*(N-1))

    def forward(self, embeddings, y_batch, weight):
        self.viewWeight = weight
        (embedding_Ia, embedding_Ie, embedding_AC) = embeddings
        lossJIa = self.subnet_loss(embedding_Ia, y_batch, self.proxies[0])#属性内网络的多元组度量损失
        lossJIe = self.subnet_loss(embedding_Ie, y_batch, self.proxies[1])#属性间网络的多元组度量损失
        lossJAC = self.subnet_loss(embedding_AC, y_batch, self.proxies[2])#属性对类网络的多元组度量损失

        HSIC_IaIe = self.HSIC(embedding_Ia, embedding_Ie)
        HSIC_IaAC = self.HSIC(embedding_Ia, embedding_AC)
        HSIC_IeAC = self.HSIC(embedding_Ie, embedding_AC)

        # weight = np.array([lossJIa.item(), lossJIe.item(), lossJAC.item()])
        # weight = (1.0/weight) / np.sum(1.0/weight)

        loss_Ia = weight[0]*lossJIa + self.lambda2 * (HSIC_IaIe + HSIC_IaAC - self.HSIC(embedding_Ia, y_batch))
        loss_Ie = weight[1]*lossJIe + self.lambda2 * (HSIC_IaIe + HSIC_IeAC - self.HSIC(embedding_Ie, y_batch))
        loss_AC = weight[2]*lossJAC + self.lambda2 * (HSIC_IaAC + HSIC_IeAC - self.HSIC(embedding_AC, y_batch))

        J_loss = np.array([lossJIa.item(), lossJIe.item(), lossJAC.item()])
        loss = loss_Ia + loss_Ie + loss_AC + mu * np.sum(weight)
        # print(weight,J_loss, loss_Ia.item(), loss_Ie.item(), loss_AC.item())
        return loss, J_loss


def build_classifier_forTripleView(x_train_tensor, y_train, x_test_tensor, y_test, model_TripleView, subnet_weight):
    subnet_weight = subnet_weight**0.5
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


#---------------------------主程序入口---------------------------------#
# result_file = "E:/实验与数据/multiView_Deep_Metric_Learning/result/MvDML_CR.txt"
# open(result_file,'w').close()
dataset_name = 	["abalone","adult","anneal","audiology","automobile",
				"breast-cancer","car","census","colic","credit-a",
				"credit-g","crx","german","hayes-roth","heart-c",
				"heart-h","heart-statlog","hepatitis","hypothyroid","kr-vs-kp",
				"labor","lymphography","mushroom","nursery","nsl-kdd",
				"post-operative","primary-tumor","promoter","sick","soybean",
				"splice","vote","vowel","zoo"]
				#"census","hypothyroid"

for i in range(2,3):
    # txtfile=open(result_file,'a+')
    print(dataset_name[i], end=',shape=')
    F1scores = []
    #获取属性内、属性间、属性对类三个视图的耦合数据及其维度，类标签
    Ia_data, Ie_data, AC_data, dimEmbed, Y_label = fs.coupleData(dataset_name[i])
    coupledData = np.c_[Ia_data, Ie_data, AC_data]
    print(dimEmbed)
    # print(Ia_data, Ie_data, AC_data, Y_label)

   #模型超参设置
    hidden_struct = [500, 90]                  # 隐层到输出层的神经元数
    dropout = 0.05                             #神经网络dropout值
    num_classes = len(np.unique(Y_label))      #类标签个数
    num_epochs = 500                           # 训练轮数
    batch_size = 500                           # 一批数据个数
    learning_rate = 1.0e-3                     # 神经网络的学习率
    tau = 8
    mu = 1.5
    lambda1 = 1.0e-3                           # 损失函数权衡参数参设置
    lambda2 = 2                               # 30 不同子网络的关联性权衡参数
    scheduler_para = {"step":50, "gamma":0.95} #调整学习的参数：训练step轮后learning_rate*gamma
    setup_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run=0
    results = []
    for train_index, test_index in StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=0).split(coupledData, Y_label):
        # 数据划分print(train_index.shape, train_index)
        F1scores = []
        run += 1
        x_train, y_train, x_test, y_test = coupledData[train_index], Y_label[train_index], coupledData[test_index], Y_label[test_index]
        input_size = x_train.shape[1]         # 输入数据的维度
        x_train_tensor, y_train_tensor, x_test_tensor = torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long(), torch.from_numpy(x_test).float()
        dataLoader = DataLoader(dataset=TensorDataset(x_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=False)
        num_batch = dataLoader.__len__()

        start = time.time()
        model = DDDML_Embedding(dimEmbed["IaEmbed"],dimEmbed["IeEmbed"],dimEmbed["ACEmbed"], hidden_struct, dropout)
        DDDML_Loss = Proxy_DDDMLloss(num_classes=num_classes, num_embed=hidden_struct[-1], lambda2=lambda2, mu=1, tau=tau)
        # print(model)
        if device:
            model.cuda()
            DDDML_Loss.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lambda1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_para["step"], gamma=scheduler_para["gamma"] )
        model.train()

        loss_epoch = []
        weight = np.array([1, 1, 1])/3  #三个子网络的初始权重
        for epoch in range(num_epochs):
            if device:
                model.cuda()
            loss_batch = []
            subnet_loss_batch = np.zeros(3)
            J_loss = 0
            for i_batch, (x_batch, y_batch) in enumerate(dataLoader):
                if device:
                    x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
                optimizer.zero_grad()
                embeddings = model(x_batch)
                # print(embeddings)
                loss, Jloss = DDDML_Loss(embeddings, y_batch, weight)
                J_loss += Jloss
                if i_batch+1 == num_batch:
                    unif = J_loss / np.sum(J_loss)
                    mu = max(np.max([3*unif-1, 0.5-1.5*unif]), mu)#保证权重非负
                    weight = (mu + 1 - 3 * unif) / (3 * mu)
                    # print(J_loss,weight)
                    # weight = (1.0/J_loss) / np.sum(1.0/J_loss)
                    J_loss = 0
                loss_batch.append(loss.item())
                loss.backward()
                optimizer.step()
            scheduler.step()
            loss_epoch.append(np.mean(loss_batch))
            if (epoch+1)%10==0:
                F1scores.append([run, epoch+1])
                F1scores[-1].extend(list(weight))
        #         # print('n=%3d,     loss=%9.4f,     view_weight=%s,     Current_score=%7.4f%%'%(epoch+1, train_loss,loss_fn.weight,score*100), file=kwargs['txtfile'])
                print('epoch=%3d,  loss=%8.4f,  view_weight=[%7.4f,%7.4f,%7.4f],'%(epoch+1, loss_epoch[-1],weight[0],weight[1],weight[2]),end='     Current_score=')

                for w in np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                                    [1, 1, 0], [1, 0, 1], [0, 1, 1],
                                    [1, 1, 1]]) * weight:
                    score = build_classifier_forTripleView(x_train_tensor   = x_train_tensor,
                                                           y_train          = y_train,
                                                           x_test_tensor    = x_test_tensor,
                                                           y_test           = y_test,
                                                           model_TripleView = model,
                                                           subnet_weight    = w)
                    print('%7.4f'%(score*100), end=",   ")
                    F1scores[-1].append(score*100)
                print()
        F1scores = np.array(F1scores)
        F1scores_sort = F1scores[np.argsort(-F1scores[:,-1])]
        if run==1:
            results = F1scores[np.argsort(-F1scores[:,-1])][:5]
        else:
            results = np.r_[results, F1scores[np.argsort(-F1scores[:,-1])][:5]]
        print("  run_time:%11.4f"%(time.time()-start))
        # print("%14s, run%d, run_time:%11.4f"%(dataset_name[i], run, time.time()-start), file=txtfile)
        # print("  run_time:%11.4f"%(time.time()-start), file=txtfile)

    # print(results, file=txtfile)
    # print("-"*150,"\n\n")
    # print("-"*150, file=txtfile)
    # txtfile.close()
