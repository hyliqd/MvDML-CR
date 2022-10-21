import numpy as np
np.set_printoptions(linewidth=400)
import functions as fs
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import StratifiedShuffleSplit
import time
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

def init_P(X, EmbDim):
    P = []
    for v in range(len(X)):
        N = X[v].shape[0]
        S = torch.mm(X[v], X[v].T)
        DN = torch.rsqrt(torch.sum(S, axis=0))
        DN = torch.diag(torch.where(torch.isinf(DN), torch.full_like(DN, 0), DN))
        L = torch.eye(N).cuda() - torch.mm(torch.mm(DN, S), DN)
        matrix = torch.mm(torch.mm(X[v].T, L), X[v])
        _, evecs = torch.linalg.eigh(matrix)
        P.append(evecs[:, -EmbDim[v]:])
    return P

def scatterMatrix(X, y):
    (N, M) = X.shape
    #计算全散度矩阵St
    miu = torch.mean(X, axis=0)
    St = torch.mm((X - miu).T, X - miu)
    # 计算类间散度矩阵Sb
    Sb = 0
    for label in torch.unique(y):
        x_clu = X[y == label, :]
        miuj_miu = (torch.mean(x_clu, axis=0) - miu).reshape((M, 1))
        x_clu, miuj_miu = x_clu.cpu(), miuj_miu.cpu()
        Sb += x_clu.shape[0] * torch.mm(miuj_miu, miuj_miu.T)
    return St / N, (Sb / N).cuda()

def HSIC(v, X, P):
    N = X[v].shape[0]
    H = (torch.eye(N) - torch.ones((N, N))/N).cuda()
    K_complement = 0
    for vi in range(len(X)):
        if vi != v:
            K_complement += torch.mm(torch.mm(X[vi], torch.mm(P[vi], P[vi].T)), X[vi].T)
    K = torch.mm(torch.mm(H, K_complement), H)
    return torch.mm(torch.mm(X[v].T, K), X[v])

def FISH(MulViewX,y,lambda1=1,lambda2=1,maxEmbDim=90,maxIter=10,max_batch_size=100):
    id_batch = np.array_split(np.arange(len(y)), np.ceil(len(y) / max_batch_size))
    EmbDim = [min(d.shape[1], maxEmbDim) for d in MulViewX]
    #initialize P
    MulViewX_firstBatch = [X[id_batch[0]].cuda() for X in MulViewX]
    P = init_P(MulViewX_firstBatch, EmbDim)
    # print("update P")
    # update P
    for iter in range(maxIter):
        for id in id_batch:
            MulViewX_batch, y_bach = [X[id].cuda() for X in MulViewX], y[id].cuda()
            for v in range(len(MulViewX_batch)):
                A, B = scatterMatrix(MulViewX_batch[v], y_bach)
                C = HSIC(v, MulViewX_batch, P)
                A, B, C = A.cpu(), B.cpu(), C.cpu()
                _, evecs = torch.linalg.eigh(A - lambda1 * B + lambda2 * C)
                P[v] = evecs[:, -EmbDim[v]:].cuda()
                A, B, C = A.cuda(), B.cuda(), C.cuda()
    return P



#---------------------------主程序入口---------------------------------#
# result_file = "E:/实验与数据/multiView_Deep_Metric_Learning/result/FISH_DML.txt"
# open(result_file,'w').close()
dataset_name = ["abalone","adult","anneal","audiology","automobile","breast-cancer","car","census","colic","credit-a","credit-g","crx","german","hayes-roth","heart-c","heart-h","heart-statlog","hepatitis","hypothyroid","kr-vs-kp","labor","lymphography","mushroom","nursery","nsl-kdd","post-operative","primary-tumor","promoter","sick","soybean","splice","vote","vowel","zoo"]
for i in range(4,5):#34  7, 8
    # txtfile=open(result_file,'a+')
    print(dataset_name[i], end=',shape=')
    # print(dataset_name[i], end=',shape=', file=txtfile)
    start = time.time()
    #获取属性内、属性间、属性对类三个视图的耦合数据及其维度，类标签
    Ia_data, Ie_data, AC_data, dimEmbed, Y_label = fs.coupleData(dataset_name[i])
    print(dimEmbed)
    # print(dimEmbed, file=txtfile)
    max_dim = 10000
    if dimEmbed['IeEmbed'] > max_dim:
        dimEmbed['IeEmbed'] = max_dim
        Ie_data = SelectKBest(chi2, k=max_dim).fit_transform(Ie_data, Y_label)
        print("IeEmbed的数据维度修改为", max_dim)
        # print("IeEmbed的数据维度修改为", max_dim, file=txtfile)

   #模型超参设置
    EmbDim = 90           #每个视图的嵌入维度数
    num_epochs = 10       #训练轮数
    batch_size = 2000     #一批数据个数
    lambda1 = 1.0         #目标函数的子视图内部的参数
    lambda2 = 1.0         #目标函数的视图之间的参数

    run = 0
    for train_index, test_index in StratifiedShuffleSplit(n_splits=15,test_size=0.2,random_state=0).split(Ia_data, Y_label):
        run += 1
        x_train, y_train = [Ia_data[train_index], Ie_data[train_index], AC_data[train_index]], Y_label[train_index]
        x_test, y_test = [Ia_data[test_index], Ie_data[test_index], AC_data[test_index]], Y_label[test_index]

        x_train, y_train = [torch.from_numpy(X).float() for X in x_train], torch.from_numpy(y_train).long()
        P = FISH(MulViewX=x_train,
                 y=y_train,
                 lambda1=lambda1,
                 lambda2=lambda2,
                 maxEmbDim=EmbDim,
                 maxIter=num_epochs,
                 max_batch_size=batch_size)
        # print(P)

        x_train, y_train = [X.cpu() for X in x_train], y_train.cpu()
        x_test = [torch.from_numpy(X).float() for X in x_test]
        P = [p.cpu() for p in P]
        x_train_emb = torch.mm(x_train[0], P[0])
        x_test_emb = torch.mm(x_test[0], P[0])
        for v in range(1, len(P)):
            x_train_emb = torch.cat((x_train_emb, torch.mm(x_train[v], P[v])), 1)
            x_test_emb = torch.cat((x_test_emb, torch.mm(x_test[v], P[v])), 1)

        knn_classifier = KNeighborsClassifier(n_neighbors=1)
        knn_classifier.fit(x_train_emb, y_train)
        y_predict = knn_classifier.predict(x_test_emb)
        score = f1_score(y_test, y_predict, average='micro')
        if run == 1:
            print("run_time:%11.4f,  F1_score: %7.4f"%(time.time()-start, score*100))
            # print("run_time:%11.4f,  F1_score: %7.4f"%(time.time()-start, score*100), end=",   ", file=txtfile)
        else:
            print('%7.4f'%(score*100))
            # print('%7.4f'%(score*100), end=",   ", file=txtfile)
    print("-"*100)
    # print("\n", "-"*100, file=txtfile)
    # txtfile.close()








'''
def init_P(X, EmbDim):
    P = []
    for v in range(len(X)):
        N = X[v].shape[0]
        S = np.dot(X[v], X[v].T)
        DN = np.sqrt(np.sum(S, axis=0))
        DN = np.diag(np.divide(1.0, DN, out=np.zeros_like(DN, dtype=np.float64),where=DN!=0))
        L = np.eye(N) - np.dot(np.dot(DN, S), DN)
        matrix = np.dot(np.dot(X[v].T, L), X[v])
        evals, evecs = linalg.eigh(matrix)
        evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors
        evecs /= np.linalg.norm(evecs, axis=0)
        P.append(evecs[:, :EmbDim[v]])
    return P

def scatterMatrix(X, y):
    (N, M) = X.shape
    #计算全散度矩阵St
    miu = np.mean(X, axis=0)
    St = np.dot((X - miu).T, X - miu)
    # 计算类间散度矩阵Sb
    Sb = 0
    for label in np.unique(y):
        x_clu = X[y == label, :]
        miuj_miu = (np.mean(x_clu, axis=0) - miu).reshape((M, 1))
        Sb += x_clu.shape[0] * np.dot(miuj_miu, miuj_miu.T)
    return St / N, Sb / N

def HSIC(v, X, P):
    N = X[v].shape[0]
    H = np.eye(N) - np.ones((N, N))/N
    K_complement = 0
    for vi in range(len(X)):
        if vi != v:
            K_complement += np.dot(np.dot(X[vi], np.dot(P[vi], P[vi].T)), X[vi].T)
    K = np.dot(np.dot(H, K_complement), H)
    return np.dot(np.dot(X[v].T, K), X[v])

def FISH(MulViewX,y,lambda1=1,lambda2=1,maxEmbDim=90,maxIter=10,max_batch_size=100):
    id_batch = np.array_split(np.arange(len(y)), np.ceil(len(y) / max_batch_size))
    EmbDim = [min(d.shape[1], maxEmbDim) for d in MulViewX]
    #initialize P
    P = init_P(MulViewX, EmbDim)
    # update P
    for iter in range(maxIter):
        for id in id_batch:
            MulViewX_batch, y_bach = [X[id] for X in MulViewX], y[id]
            for v in range(len(MulViewX_batch)):
                A, B = scatterMatrix(MulViewX_batch[v], y_bach)
                C = HSIC(v, MulViewX_batch, P)
                evals, evecs = linalg.eigh(A - lambda1 * B + lambda2 * C)
                evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors
                evecs /= np.linalg.norm(evecs, axis=0)
                P[v] = evecs[:, :EmbDim[v]]
    return P
'''