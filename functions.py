import numpy as np
import pandas as pd
from sklearn import preprocessing
import os
import pickle

#------------------------------数据获取与处理-----------------------------------#
def get_dataset(filename):#获取数据，输出数据和属性类型-
    read_path = "../UCI_Dataset//"+filename+".csv"
    dataset = pd.read_csv(read_path, index_col=0, low_memory=False)
    data_type = dataset.head(1).values[0][:-1]
    data = dataset[1:].astype(float).values

    #类标签只有1条记录的复制为2条
    label_count = np.array(np.unique(data[:,-1],return_index=True,return_counts=True))
    id_unique_label = label_count[1,label_count[2,:]==1].astype(int)
    if len(id_unique_label)!=0:
        data = np.vstack((data,data[id_unique_label]))
    return data, data_type


def intraAttriCouple(data,cate_attri):
    dictionary = dict()
    for col in cate_attri:
        values = np.unique(data[:,col])
        embed = np.zeros([len(values), len(values)+1], dtype = 'float32')
        for value in range(len(values)):
            embed[value, 0] = np.size(np.where(data[:,col] == values[value]))/data.shape[0] #后验概率
            embed[value, value+1] = 1 #onehot
        dictionary[col] = embed
    return dictionary


def conditional_probability(data_x,label_y):#计算x的条件概率；返回x对y的条件概率的矩阵
    df = pd.DataFrame(np.vstack((data_x,label_y)).T,columns=['xname','yname'])
    count_x_y = {kx:{ky:len(groupy) for ky,groupy in groupx.groupby('yname')} for kx,groupx in df.groupby('xname')}
    condProb = []
    for kx in count_x_y:
        prob,num_kx = [],np.sum(list(count_x_y[kx].values()))
        for ky in np.unique(label_y):
            if ky in count_x_y[kx].keys():
                prob.append(count_x_y[kx][ky] / num_kx)
            else:
                prob.append(0)
        condProb.append(prob)
    return np.array(condProb)

def interAttriCouple(data,cate_attri):
    M = data.shape[1]
    data_disc = data.copy()
    if M != len(cate_attri): #离散化数值属性
        nume_attri = np.delete(np.r_[0:M], cate_attri)
        data_disc[:,nume_attri] = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform').fit_transform(data[:,nume_attri])

    dictionary = dict()
    for col in cate_attri:
        embed = []
        for i in np.delete(np.r_[0:M], col):
            if embed == []:
                embed = conditional_probability(data_disc[:,col],data_disc[:,i])
            else:
                embed = np.c_[embed,conditional_probability(data_disc[:,col],data_disc[:,i])]
        dictionary[col] = embed
    return dictionary

def attriClassCouple(data,label,cate_attri):
    dictionary = dict()
    for col in cate_attri:
        dictionary[col] = conditional_probability(data[:,col],label)
    return dictionary

def categoricalEncoder(data,cate_attri,cate_encoding):
    numeData = data[:, np.delete(np.r_[0:data.shape[1]],cate_attri)]
    embedSize = {key:cate_encoding[key].shape[1] for key in cate_encoding}
    embedData = np.c_[np.zeros([data.shape[0], np.sum(list(embedSize.values()))]), numeData]

    bIndex = 0
    for col in cate_attri:
        eIndex = bIndex + embedSize[col]
        for value in range(cate_encoding[col].shape[0]):
            embedData[np.where(data[:,col] == value), bIndex:eIndex] = cate_encoding[col][value]
        bIndex = eIndex
    return embedData

def coupleData(dataset_name):
    #数据处理
    dataset, attri_type = get_dataset(filename=dataset_name)#获取数据
    X_data, Y_label = dataset[:,:-1], dataset[:,-1]
    cate_attri = np.argwhere(attri_type == 'categorical')[:,0]
    print(X_data.shape)

    #耦合学习，分类值的数值表示
    filename = "../couple_learing//"+dataset_name+".pickle"
    if os.path.exists(filename):#直接读取已有耦合学习数据
        with open(filename,'rb') as f:
            couple_learning = pickle.load(f)
        IaEmbed = couple_learning["IaEmbed"]
        IeEmbed = couple_learning["IeEmbed"]
        ACEmbed = couple_learning["ACEmbed"]
    else:   #耦合学习并写入数据
        IaEmbed = intraAttriCouple(data=X_data,cate_attri=cate_attri)
        IeEmbed = interAttriCouple(data=X_data,cate_attri=cate_attri)
        ACEmbed = attriClassCouple(data=X_data,label=Y_label,cate_attri=cate_attri)
        couple_learning = {"IaEmbed":IaEmbed, "IeEmbed":IeEmbed, "ACEmbed":ACEmbed}
        f=open(filename,'wb')
        pickle.dump(couple_learning, f)
        f.close()
    data_encod_Ia = categoricalEncoder(data=X_data,cate_attri=cate_attri,cate_encoding=IaEmbed)
    data_encod_Ie = categoricalEncoder(data=X_data,cate_attri=cate_attri,cate_encoding=IeEmbed)
    data_encod_AC = categoricalEncoder(data=X_data,cate_attri=cate_attri,cate_encoding=ACEmbed)
    dimEmbed = {"IaEmbed": data_encod_Ia.shape[1], "IeEmbed": data_encod_Ie.shape[1], "ACEmbed": data_encod_AC.shape[1]}
    return data_encod_Ia, data_encod_Ie, data_encod_AC, dimEmbed, Y_label
