import numpy as np
np.set_printoptions(linewidth=400)
import functions as fs
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import StratifiedShuffleSplit
import datetime
import time
import torch
from sklearn.metrics import f1_score


def init_M_Alpha(X, EmbDim):
    Mp = []
    for v in range(len(X)):
        N = X[v].shape[0]
        S = torch.mm(X[v], X[v].T)
        DN = torch.rsqrt(torch.sum(S, axis=0))
        DN = torch.diag(torch.where(torch.isinf(DN), torch.full_like(DN, 0), DN))
        L = torch.eye(N).cuda() - torch.mm(torch.mm(DN, S), DN)
        matrix = torch.mm(torch.mm(X[v].T, L), X[v])
        _, evecs = torch.linalg.eigh(matrix)
        Mp.append(evecs[:, -EmbDim[v]:])
    Alpha = torch.ones(len(X)).cuda() / len(X)
    return Mp, Alpha


def min_dist_sample_pair(distX,Y,label):
    maxD = torch.max(distX)
    distX.fill_diagonal_(maxD+1)
    N = distX.shape[0]
    sample_pair, sample_pair_dist = [range(N)], []
    for l in label:
        min_dist = torch.min(distX[:, Y==l],1)[0]
        sample_pair_dist.append(min_dist)
        xi, xj = torch.where(torch.eq(distX, min_dist[:,None]))
        n_xi = xi.shape[0]
        if n_xi==N: #没有2个及以上与xi的最小距离相等的xj
            sample_pair.append(xj)
        else:
            select_xj = [xj[n] for n in range(n_xi-1) if xi[n]!=xi[n+1]]
            if len(select_xj) < N: select_xj.append(xj[n_xi-1])
            sample_pair.append(select_xj)
    sample_pair = torch.tensor(sample_pair).t().cuda()
    pair_dist = torch.stack(sample_pair_dist).t().cuda()
    sample_pair_dist = torch.where(pair_dist!=maxD+1, pair_dist, torch.zeros_like(pair_dist))

    exp_dist = torch.exp(-sample_pair_dist)
    sample_pair_prob = torch.div(exp_dist.t(), torch.sum(exp_dist, dim=1)).t()
    matches = (Y.unsqueeze(1) == label.unsqueeze(0)).byte()
    coef= (matches + sample_pair_prob)
    return sample_pair, sample_pair_dist, coef


def gradient_Mp(X, Y, Mp, Alpha, Lambda):
    embedX = [torch.mm(x, mp) for x, mp in zip(X, Mp)] #多视图数据在MP下的嵌入
    distMatrix = [torch.norm(ex[:, None] - ex, dim=2, p=2) for ex in embedX] #不同视图嵌入下样本间的距离
    distX = 0  #在几个视图下的加权距离
    for a, d in zip(Alpha, distMatrix):
        distX += a * d
    label = torch.unique(Y)
    sample_pair, sample_pair_dist, coef = min_dist_sample_pair(distX, Y, label)
    coef_sqrt = coef.sqrt()

    delt_Mp = []
    for v in range(len(Mp)):
        delt_Mp_1_3 = torch.zeros_like(Mp[v])
        for l in range(len(label)):
            X_with_coef = (X[v][sample_pair[:, l+1]] - X[v][sample_pair[:,0]]) * coef_sqrt[:, l].reshape((-1,1))
            delt_Mp_1_3 += torch.mm(X_with_coef.t(), torch.mm(X_with_coef, Mp[v]))
        delt_Mp.append(0.5 * delt_Mp_1_3 / X[v].shape[0] + Lambda * Mp[v])
    return delt_Mp


def gradient_Alpha(X, Y, Mp, Alpha, Mu):
    embedX = [torch.mm(x, mp) for x, mp in zip(X, Mp)] #多视图数据在MP下的嵌入
    distMatrix = [torch.norm(ex[:, None] - ex, dim=2, p=2) for ex in embedX] #不同视图嵌入下样本间的距离
    distX = 0  #在几个视图下的加权距离
    for a, d in zip(Alpha, distMatrix):
        distX += a * d
    label = torch.unique(Y)
    sample_pair, sample_pair_dist, coef = min_dist_sample_pair(distX, Y, label)

    delt_Alpha = torch.zeros(len(X))
    for v in range(len(X)):
        each_view_pair_dist = torch.stack([distMatrix[v][n, sample_pair[n,1:]] for n in range(sample_pair.shape[0])]).cuda()
        delt_Alpha[v] = torch.sum(coef * each_view_pair_dist) / X[v].shape[0] + Mu * Alpha[v]
    return delt_Alpha.cuda()


def testModel(Xtrain, Ytrain, Xtest, Mp, Alpha):
    max_batch_size = 1000
    nTrain, nTest = Xtrain[0].shape[0], Xtest[0].shape[0]
    embedXtrain = [torch.mm(x, mp) for x, mp in zip(Xtrain, Mp)]
    embedXtest = [torch.mm(x, mp) for x, mp in zip(Xtest, Mp)]
    # print(embedXtrain,embedXtest)
    id_batch = np.array_split(np.arange(nTest), np.ceil(nTest / max_batch_size))
    predTestY = torch.zeros(nTest, dtype=torch.long)
    for id in id_batch:
        embedXtest_batch = [x[id] for x in embedXtest]
        distMatrix = torch.zeros((len(id), nTrain))
        for v in range(len(embedXtest_batch)):
            distMatrix += Alpha[v] * torch.cdist(embedXtest_batch[v], embedXtrain[v])
        index = torch.min(distMatrix, dim=1)[1]
        predTestY[id] = Ytrain[index]
    return predTestY


def MVMIML(MulViewX,y,lambda_=1,mu=1, eta1=0.1, eta2=0.1, maxEmbDim=90,maxIterR=5,maxIterQ=10,max_batch_size=100):
    id_batch = np.array_split(np.arange(len(y)), np.ceil(len(y) / max_batch_size))
    EmbDim = [min(d.shape[1], maxEmbDim) for d in MulViewX]
    # initialize MP and Alpha  当x的维度太大时，M矩阵的维度也很大，超出内存，故转化为M=MP*MP.T，生成MP
    MulViewX_firstBatch = [X[id_batch[0]].cuda() for X in MulViewX]
    Mp, Alpha = init_M_Alpha(MulViewX_firstBatch, EmbDim)

    for iter_updata_Alpha in range(maxIterQ):
        for iter_updata_M in range(maxIterR):#更新度量矩阵 M
            for id in id_batch:
                MulViewX_batch, y_bach = [X[id].cuda() for X in MulViewX], y[id].cuda()
                delt_Mp = gradient_Mp(MulViewX_batch, y_bach, Mp, Alpha, lambda_)
                Mp = [mp - eta1 * delt_mp for mp, delt_mp in zip(Mp, delt_Mp)]

        for id in id_batch: #更新权重向量 Alpha
            MulViewX_batch, y_bach = [X[id].cuda() for X in MulViewX], y[id].cuda()
            delt_Alpha = gradient_Alpha(MulViewX_batch, y_bach, Mp, Alpha, mu)
            Alpha -= Alpha - eta2 * delt_Alpha
    return Mp, Alpha/torch.sum(Alpha)


#---------------------------主程序入口---------------------------------#
result_file = "E:/实验与数据/multiView_Deep_Metric_Learning/result/MVMIML.txt"
# open(result_file,'w').close()
dataset_name = ["abalone","adult","anneal","audiology","automobile","breast-cancer","car","census","colic","credit-a","credit-g","crx","german","hayes-roth","heart-c","heart-h","heart-statlog","hepatitis","hypothyroid","kr-vs-kp","labor","lymphography","mushroom","nsl-kdd","nursery","post-operative","primary-tumor","promoter","sick","soybean","splice","vote","vowel","zoo"]
for i in range(34):#4, 6  7, 8
    txtfile=open(result_file,'a+')
    print(i+1,dataset_name[i], end=',shape=')
    print(i+1,dataset_name[i], end=',', file=txtfile)
    start = time.time()
    #获取属性内、属性间、属性对类三个视图的耦合数据及其维度，类标签
    Ia_data, Ie_data, AC_data, dimEmbed, Y_label = fs.coupleData(dataset_name[i])
    print('current_time',datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'      embedDim=', dimEmbed)
    print('embedDim=', dimEmbed, file=txtfile)
    max_dim = 10000
    if dimEmbed['IeEmbed'] > max_dim:
        dimEmbed['IeEmbed'] = max_dim
        Ie_data = SelectKBest(chi2, k=max_dim).fit_transform(Ie_data, Y_label)
        print("IeEmbed的数据维度修改为", max_dim)
        print("IeEmbed的数据维度修改为", max_dim, file=txtfile)

   #模型超参设置
    EmbDim = 90                   #每个视图的嵌入维度数
    maxIterR, maxIterQ = 1, 3     #训练轮数
    batch_size = 1500              #一批数据个数
    lambda_, mu = 0.1, 0.1          #度量矩阵Mp和权重Alpha的权重因子
    eta1, eta2 = 0.1, 0.1         #度量矩阵Mp和权重Alpha的学习率
    run = 0

    for train_index, test_index in StratifiedShuffleSplit(n_splits=10,test_size=0.2,random_state=0).split(Ia_data, Y_label):
        run += 1
        x_train, y_train = [Ia_data[train_index], Ie_data[train_index], AC_data[train_index]], Y_label[train_index]
        x_test, y_test = [Ia_data[test_index], Ie_data[test_index], AC_data[test_index]], Y_label[test_index]

        x_train, y_train = [torch.from_numpy(X).float() for X in x_train], torch.from_numpy(y_train).long()
        Mp, Alpha = MVMIML(MulViewX=x_train,
                           y=y_train,
                           lambda_=lambda_,
                           mu=mu,
                           eta1=eta1,
                           eta2=eta2,
                           maxEmbDim=EmbDim,
                           maxIterR=maxIterR,
                           maxIterQ=maxIterQ,
                           max_batch_size=batch_size)
        #由于显存不足，在CPU中测试
        x_train, y_train = [X.cpu() for X in x_train], y_train.cpu()
        x_test = [torch.from_numpy(X).float() for X in x_test]
        Mp, Alpha = [mp.cpu() for mp in Mp], Alpha.cpu()
        y_predict = testModel(Xtrain=x_train,
                              Ytrain=y_train,
                              Xtest=x_test,
                              Mp=Mp,
                              Alpha=Alpha)
        score = f1_score(y_test, y_predict, average='micro')
        if run == 1:
            print("run_time:%11.4f,  F1_score: %7.4f"%(time.time()-start, score*100),end=", ", flush=True)
            print("run_time:%11.4f,  F1_score: %7.4f"%(time.time()-start, score*100), end=",   ", file=txtfile)
        else:
            print('%7.4f'%(score*100),end=", ", flush=True)
            print('%7.4f'%(score*100), end=",   ", file=txtfile)
    print()
    print(file=txtfile)#"\n", "-"*100,
    txtfile.close()