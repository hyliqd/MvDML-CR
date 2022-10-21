import numpy as np
np.set_printoptions(linewidth=400)
import functions as fs
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import StratifiedShuffleSplit
import time
import torch
from sklearn.metrics import f1_score

def init_W_b_u(Mv_X, Y, gamma=1.0):#使用least_squares_regression初始化W, b
    W, b, u = [], [], []
    for v in range(len(Mv_X)):
        (N, dim) = Mv_X[v].shape
        Xv = (Mv_X[v] - np.mean(Mv_X[v], axis=0))#数据X中心化
        if dim < N:
            temp = np.dot(Xv.T, Xv) + gamma * np.eye(dim)
            W.append(np.dot(np.linalg.pinv(temp), np.dot(Xv.T, Y)))
        else:
            temp = np.dot(Xv, Xv.T) + gamma * np.eye(N)
            W.append(np.dot(Xv.T, np.dot(np.linalg.pinv(temp), Y)))
        b.append(np.mean(Y - np.dot(Xv, W[v]), axis=0).reshape(Y.shape[1], 1))
        error = np.dot(Xv, W[v]) + np.repeat(b[v], N, axis=1).T - Y
        u.append(0.5 / (np.linalg.norm(error, axis=1) + 1e-12))
    return W, b, u # 3(M*C), 3(C*1), 3(N),

def MCCD(MulViewX,Y,gamma1=1,gamma2=1,maxIter=10,max_batch_size=100):
    id_batch = np.array_split(np.arange(Y.shape[0]), np.ceil(Y.shape[0] / max_batch_size))
    EmbDim = [d.shape[1] for d in MulViewX]
    viewNum = len(MulViewX)
    #initialize W, b, u
    W, b, u = init_W_b_u(MulViewX, Y, gamma=gamma1)
    # update W, b
    threshold = 1e-6
    epsilon = 1e-12
    MulViewX, Y = [torch.from_numpy(X).float() for X in MulViewX], torch.from_numpy(Y).float()
    W = [torch.from_numpy(Wv).float().cuda() for Wv in W]
    b = [torch.from_numpy(bv).float().cuda() for bv in b]
    u = [torch.from_numpy(uv).float().cuda() for uv in u]
    obj = []
    for iter in range(maxIter):
        obj.append(0)
        for id in id_batch:
            MulViewX_batch, Y_bach = [X[id].cuda() for X in MulViewX], Y[id].cuda()
            u_bach = [u_[id] for u_ in u]
            nSmp = Y_bach.shape[0]
            B = 2 * Y_bach - 1
            M = torch.zeros_like(Y_bach)
            en = torch.ones(nSmp, 1).cuda()
            Hn = (torch.eye(nSmp) - torch.ones(nSmp)/nSmp).cuda()

            Z = Y_bach + torch.mul(B, M)
            sumU = torch.zeros(nSmp, nSmp).cuda()
            sumQ = torch.zeros_like(Y_bach)
            for v in range(viewNum):
                sumKHn = torch.zeros_like(sumU)
                for v1 in range(viewNum):
                    if v1 != v:
                        emb = torch.mm(MulViewX_batch[v1], W[v1])
                        sumKHn += torch.mm(emb, emb.T)
                sumKHn = torch.mm(torch.mm(Hn, sumKHn), Hn)
                K = torch.mm(torch.mm(MulViewX_batch[v].T, sumKHn), MulViewX_batch[v])
                Uv = torch.diag(u_bach[v])
                u_bach_array = torch.unsqueeze(u_bach[v], 0).T
                H = Uv - 1/torch.sum(u_bach[v]) * torch.mm(u_bach_array, u_bach_array.T)

                G = torch.mm(torch.mm(MulViewX_batch[v].T, H), MulViewX_batch[v])\
                    + gamma1 * torch.eye(EmbDim[v]).cuda()\
                    + gamma2 * K
                W[v] = torch.mm(torch.linalg.pinv(G),
                                torch.mm(torch.mm(MulViewX_batch[v].T, H), Z))
                b[v] = torch.mm((Z - torch.mm(MulViewX_batch[v], W[v])).T,
                                u_bach_array/torch.sum(u_bach[v]))
                sumU += Uv
                error = torch.mm(MulViewX_batch[v], W[v]) + torch.mm(en, b[v].T) - Y_bach
                sumQ += torch.mm(Uv, B * error)

            # update M
            m = torch.mm(torch.diag(1./(torch.diag(sumU) + epsilon)), sumQ)
            M = torch.max(m, torch.zeros_like(m))

            #update U and calculate the objective value
            for v in range(viewNum):
                temp = torch.mm(MulViewX_batch[v], W[v]) + torch.mm(en, b[v].T) - Y_bach - B * M
                u[v][id] = 0.5/torch.sqrt(torch.sum(temp * temp, dim=1) + epsilon)
                sumKHn = torch.zeros_like(sumU)
                for v1 in range(viewNum):
                    if v1 != v:
                        emb = torch.mm(MulViewX_batch[v1], W[v1])
                        sumKHn += torch.mm(emb, emb.T)

                HXW = torch.mm(torch.mm(H, MulViewX_batch[v]), W[v])
                obj[iter] += torch.sum(torch.norm(temp, p=2, dim=1)) \
                             + gamma1 * torch.pow(torch.norm(W[v], p=2), 2) \
                             + gamma2 * torch.trace(torch.mm(HXW.T, HXW))
        if iter>1 and (obj[iter]-obj[iter-1])/obj[iter-1] < threshold:
            break
    W = [Wv.cpu().numpy() for Wv in W]
    b = [bv.cpu().numpy() for bv in b]
    return W, b, obj

def make_pred(MulViewX, W, b):
    # print(W)
    nClass = b[0].shape[0]
    Ic = np.eye(nClass)
    nteSmp = MulViewX[0].shape[0]
    en = np.ones((nteSmp,1))
    F = np.zeros((nteSmp, nClass))
    viewNum = len(MulViewX)
    for cc in range(nClass):
        Ycc = np.repeat(Ic[cc].reshape(nClass, 1), nteSmp, axis=1).T
        for v in range(viewNum):
            temp = np.dot(MulViewX[v], W[v]) + np.dot(en, b[v].T) - Ycc
            F[:, cc] += np.linalg.norm(temp,ord=2,axis=1)
    ypre = np.argmin(F, axis=1)
    return ypre


#---------------------------主程序入口---------------------------------#
result_file = "E:/实验与数据/multiView_Deep_Metric_Learning/result/MCCD.txt"
open(result_file,'w').close()
dataset_name = ["abalone","adult","anneal","audiology","automobile","breast-cancer","car","census","colic","credit-a","credit-g","crx","german","hayes-roth","heart-c","heart-h","heart-statlog","hepatitis","hypothyroid","kr-vs-kp","labor","lymphography","mushroom","nursery","nsl-kdd","post-operative","primary-tumor","promoter","sick","soybean","splice","vote","vowel","zoo"]

for i in range(34):#34  7, 8
    txtfile=open(result_file,'a+')
    print(dataset_name[i], end=',shape=')
    print(dataset_name[i], end=',shape=', file=txtfile)
    start = time.time()
    #获取属性内、属性间、属性对类三个视图的耦合数据及其维度，类标签
    Ia_data, Ie_data, AC_data, dimEmbed, Y_label = fs.coupleData(dataset_name[i])
    print(dimEmbed)
    print(dimEmbed, file=txtfile)
    max_dim = 5000
    if dimEmbed['IeEmbed'] > max_dim:
        dimEmbed['IeEmbed'] = max_dim
        Ie_data = SelectKBest(chi2, k=max_dim).fit_transform(Ie_data, Y_label)
        print("IeEmbed的数据维度修改为", max_dim)
        print("IeEmbed的数据维度修改为", max_dim, file=txtfile)

   #模型超参设置
    nClass = int(np.max(Y_label))+1 #类标签数
    maxIter = 10                    #训练轮数
    max_batch_size = 5000           #一批数据个数
    gamma1 = 0.1                    #目标函数的子视图内部的参数
    gamma2 = 0.1                    #目标函数的视图之间的参数

    run = 0
    for train_index, test_index in StratifiedShuffleSplit(n_splits=10,test_size=0.2,random_state=0).split(Ia_data, Y_label):
        run += 1
        MulViewX_train, y_train = [Ia_data[train_index], Ie_data[train_index], AC_data[train_index]], Y_label[train_index]
        MulViewX_test, y_test = [Ia_data[test_index], Ie_data[test_index], AC_data[test_index]], Y_label[test_index]

        W, b, obj = MCCD(MulViewX       = MulViewX_train,
                         Y              = np.eye(nClass)[y_train.astype(int)],
                         gamma1         = gamma1,
                         gamma2         = gamma2,
                         maxIter        = maxIter,
                         max_batch_size = max_batch_size)
        y_predict = make_pred(MulViewX=MulViewX_test, W=W, b=b)
        score = f1_score(y_test, y_predict, average='micro')
        if run == 1:
            print("run_time:%11.4f,  F1_score: %7.4f"%(time.time()-start, score*100))
            print("run_time:%11.4f,  F1_score: %7.4f"%(time.time()-start, score*100), end=",   ", file=txtfile)
        else:
            print('%7.4f'%(score*100))
            print('%7.4f'%(score*100), end=",   ", file=txtfile)
    print("-"*100)
    print("\n", "-"*100, file=txtfile)
    txtfile.close()