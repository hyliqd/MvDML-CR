## similar to github.com/Michaelvll/DeepCCA main
import torch
import torch.nn as nn
torch.set_default_tensor_type(torch.DoubleTensor)
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler
import numpy as np
np.set_printoptions(linewidth=400)
import functions as fs
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import StratifiedShuffleSplit
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

try:
    import cPickle as thepickle
except ImportError:
    import _pickle as thepickle
from torchsummary import summary


class linear_gcca():
    def __init__(self, number_views=3):
        self.U = [None for _ in range(number_views)]
        self.m = [None for _ in range(number_views)]

    def fit(self, H_list, outdim_size):
        r = 1e-4
        eps = 1e-8
        top_k = outdim_size
        AT_list =  []
        for i, H in enumerate(H_list):
            if i >=3 :
                print(i)
                assert i >=3
            assert torch.isnan(H).sum().item() == 0
            o_shape = H.size(0)  # N
            m = H.size(0)   # out_dim
            self.m[i] = H.mean(dim=0)
            Hbar = H - H.mean(dim=0).repeat(1, m).view(m, -1)
            assert torch.isnan(Hbar).sum().item() == 0
            A, S, B = Hbar.svd(some=True, compute_uv=True)
            A = A[:, :top_k]
            assert torch.isnan(A).sum().item() == 0
            S_thin = S[:top_k]
            S2_inv = 1. / (torch.mul( S_thin, S_thin ) + eps)
            assert torch.isnan(S2_inv).sum().item() == 0
            T2 = torch.mul( torch.mul( S_thin, S2_inv ), S_thin )
            assert torch.isnan(T2).sum().item() == 0
            T2 = torch.where(T2>eps, T2, (torch.ones(T2.shape)*eps).to(H.device).double())

            T = torch.diag(torch.sqrt(T2))
            assert torch.isnan(T).sum().item() == 0
            T_unnorm = torch.diag( S_thin + eps )
            assert torch.isnan(T_unnorm).sum().item() == 0
            AT = torch.mm(A, T)
            AT_list.append(AT)
        M_tilde = torch.cat(AT_list, dim=1)
        assert torch.isnan(M_tilde).sum().item() == 0
        Q, R = torch.linalg.qr(M_tilde)
        assert torch.isnan(R).sum().item() == 0
        assert torch.isnan(Q).sum().item() == 0
        U, lbda, _ = R.svd(some=False, compute_uv=True)
        assert torch.isnan(U).sum().item() == 0
        assert torch.isnan(lbda).sum().item() == 0
        G = Q.mm(U[:,:top_k])
        assert torch.isnan(G).sum().item() == 0

        U = [] # Mapping from views to latent space
        # Get mapping to shared space
        views = H_list
        F = [H.shape[0] for H in H_list] # features per view
        for idx, (f, view) in enumerate(zip(F, views)):
            _, R = torch.linalg.qr(view)
            Cjj_inv = torch.inverse( (R.T.mm(R) + eps * torch.eye( view.shape[1], device=view.device)) )
            assert torch.isnan(Cjj_inv).sum().item() == 0
            pinv = Cjj_inv.mm( view.T)
            U.append(pinv.mm( G ))
        self.U = U

    def _get_result(self, x, idx):
        m = x.size(0)   # out_dim
        result = x - x.mean(dim=0).repeat(1, m).view(m, -1)
        result = torch.mm(result,self.U[idx])
        return result

    def test(self, H_list):
        resualts = []
        for i, H in enumerate(H_list):
            resualts.append(self._get_result(H, i))
        return resualts


def GCCA_loss(H_list):
    r = 1e-4
    eps = 1e-8
    top_k = 10
    AT_list =  []
    for H in H_list:
        assert torch.isnan(H).sum().item() == 0
        o_shape = H.size(0)  # N
        m = H.size(1)   # out_dim
        Hbar = H - H.mean(dim=1).repeat(m, 1).view(-1, m)
        assert torch.isnan(Hbar).sum().item() == 0
        A, S, B = Hbar.svd(some=True, compute_uv=True)
        A = A[:, :top_k]
        assert torch.isnan(A).sum().item() == 0
        S_thin = S[:top_k]
        S2_inv = 1. / (torch.mul( S_thin, S_thin ) + eps)
        assert torch.isnan(S2_inv).sum().item() == 0
        T2 = torch.mul( torch.mul( S_thin, S2_inv ), S_thin )
        assert torch.isnan(T2).sum().item() == 0
        T2 = torch.where(T2>eps, T2, (torch.ones(T2.shape)*eps).to(H.device).double())

        T = torch.diag(torch.sqrt(T2))
        assert torch.isnan(T).sum().item() == 0
        T_unnorm = torch.diag( S_thin + eps )
        assert torch.isnan(T_unnorm).sum().item() == 0
        AT = torch.mm(A, T)
        AT_list.append(AT)
    M_tilde = torch.cat(AT_list, dim=1)
    assert torch.isnan(M_tilde).sum().item() == 0
    _, S, _ = M_tilde.svd(some=True)
    assert torch.isnan(S).sum().item() == 0
    use_all_singular_values = False
    if not use_all_singular_values:
        S = S[:top_k]
    corr = torch.sum(S )
    assert torch.isnan(corr).item() == 0
    loss = - corr
    return loss

class MlpNet(nn.Module):
    def __init__(self, layer_sizes, input_size):
        super(MlpNet, self).__init__()
        layers = []
        layer_sizes = [input_size] + layer_sizes
        print(layer_sizes)
        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layers.append(nn.Sequential(
                    nn.BatchNorm1d(num_features=layer_sizes[l_id], affine=False),
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                ))
            else:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    nn.Sigmoid(),
                    nn.BatchNorm1d(num_features=layer_sizes[l_id + 1], affine=False),
                ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DeepGCCA(nn.Module):
    def __init__(self, layer_sizes_list, input_size_list, outdim_size, use_all_singular_values=False, device=torch.device('cpu')):
        super(DeepGCCA, self).__init__()
        self.model_list = []
        for i in range(len(layer_sizes_list)):
            self.model_list.append(MlpNet(layer_sizes_list[i], input_size_list[i]).double())
        self.model_list = nn.ModuleList(self.model_list)
        self.loss = GCCA_loss


    def forward(self, x_list):
        """

        x_%  are the vectors needs to be make correlated
        dim = [batch_size, features]

        """
        # feature * batch_size
        output_list = []
        for x, model in zip(x_list, self.model_list):
            output_list.append(model(x))
        return output_list


#-----------------------------------------------------------------------------------------------------




class Solver():
    def __init__(self, model, linear_gcca, outdim_size, epoch_num, batch_size, learning_rate, reg_par, device=torch.device('cpu')):
        self.model = model # nn.DataParallel(model)
        self.model.to(device)
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.loss = model.loss
        self.optimizer = torch.optim.Adam(
            self.model.model_list.parameters(), lr=learning_rate, weight_decay=reg_par)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=200, gamma=0.7)
        self.device = device
        self.linear_gcca = linear_gcca()
        self.outdim_size = outdim_size

    def fit(self, x_list, vx_list=None, tx_list=None, checkpoint='checkpoint.model'):
        """
        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]
        """
        x_list = [x.to(device) for x in x_list]

        data_size = x_list[0].size(0)

        if vx_list is not None :
            best_val_loss = 0
            vx_list = [vx.to(self.device) for vx in vx_list]

        if tx_list is not None :
            tx_list = [tx.t0(self.device) for tx in tx_list]


        train_losses = []
        for epoch in range(self.epoch_num):
            epoch_start_time = time.time()
            self.model.train()

            # for name, param in model.named_parameters():
            #     print(epoch,name,param.shape, param)
            # print("*"*100)

            batch_idxs = list(BatchSampler(RandomSampler(
                range(data_size)), batch_size=self.batch_size, drop_last=False))
            for batch_idx in batch_idxs:
                self.optimizer.zero_grad()
                batch_x = [x[batch_idx, :] for x in x_list]
                output = self.model(batch_x)
                loss = self.loss(output)
                train_losses.append(loss.item())
                loss.backward()
                # print([x.grad for x in self.optimizer.param_groups[0]['params']])
                self.optimizer.step()
                self.scheduler.step()
            train_loss = np.mean(train_losses)

            if vx_list is not None:
                with torch.no_grad():
                    self.model.eval()
                    val_loss = self.test(vx_list)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(self.model.state_dict(), checkpoint)
            else:
                torch.save(self.model.state_dict(), checkpoint)
            epoch_time = time.time() - epoch_start_time
            if (epoch + 1) % 50 == 0:
                print("epoch:%3d/%d,  epoch_time:%8.4f,  train_loss: %7.4f"%(epoch + 1, self.epoch_num, epoch_time, train_loss))
        if self.linear_gcca is not None:
            _, outputs_list = self._get_outputs(x_list)
            self.train_linear_gcca(outputs_list)

        checkpoint_ = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint_)

    def test(self, x_list, use_linear_gcca=False):
        with torch.no_grad():
            losses, outputs_list = self._get_outputs(x_list)

            if use_linear_gcca:
                print("Linear CCA started!")
                outputs_list = self.linear_gcca.test(outputs_list)
                return np.mean(losses), outputs_list
            else:
                return np.mean(losses)

    def train_linear_gcca(self, x_list):
        self.linear_gcca.fit(x_list, self.outdim_size)

    def _get_outputs(self, x_list):
        with torch.no_grad():
            self.model.eval()
            data_size = x_list[0].size(0)
            batch_idxs = list(BatchSampler(SequentialSampler(
                range(data_size)), batch_size=self.batch_size, drop_last=False))
            losses = []
            outputs_list = []
            for batch_idx in batch_idxs:
                batch_x = [x[batch_idx, :].to(self.device) for x in x_list]
                outputs = self.model(batch_x)
                outputs_list.append([o_j.clone().detach() for o_j in outputs])
                loss = self.loss(outputs)
                losses.append(loss.item())
        outputs_final = []
        for i in range(len(x_list)):
            view = []
            for j in range(len(outputs_list)):
                view.append(outputs_list[j][i].clone().detach())
            view = torch.cat(view, dim=0)
            outputs_final.append(view)
        return losses, outputs_final

    def save(self, name):
        torch.save(self.model, name)




#===============================================================================================
result_file = "E:/实验与数据/multiView_Deep_Metric_Learning/result/DGCCA.txt"
# open(result_file,'w').close()
# dataset_name = ["abalone","adult","anneal","audiology","automobile","breast-cancer","car","census","colic","credit-a","credit-g","crx","german","hayes-roth","heart-c","heart-h","heart-statlog","hepatitis","hypothyroid","kr-vs-kp","labor","lymphography","mushroom","nursery","nsl-kdd","post-operative","primary-tumor","promoter","sick","soybean","splice","vote","vowel","zoo"]
dataset_name = ["zoo","car","nursery"]
for i in range(1):#34  7, 8
    txtfile=open(result_file,'a+')
    print(dataset_name[i], end=',shape=')
    print(dataset_name[i], end=',shape=', file=txtfile)
    start = time.time()
    #获取属性内、属性间、属性对类三个视图的耦合数据及其维度，类标签
    Ia_data, Ie_data, AC_data, dimEmbed, Y_label = fs.coupleData(dataset_name[i])
    print(dimEmbed)
    print(dimEmbed, file=txtfile)
    max_dim = 660
    if dimEmbed['IeEmbed'] > max_dim:
        dimEmbed['IeEmbed'] = max_dim
        Ie_data = SelectKBest(chi2, k=max_dim).fit_transform(Ie_data, Y_label)
        print("IeEmbed的数据维度修改为", max_dim)
        print("IeEmbed的数据维度修改为", max_dim, file=txtfile)

    # Parameters Section
    learning_rate = 1e-2
    epoch_num = 500
    batch_size = 2000
    reg_par = 1e-5
    use_all_singular_values = False
    apply_linear_gcca = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_name = './DGCCA.model'

    # n_samples = 1000#10000
    outdim_size = 90
    layer_sizes1 = [500, outdim_size]
    layer_sizes2 = [500, outdim_size]
    layer_sizes3 = [500, outdim_size]
    layer_sizes_list = [layer_sizes1, layer_sizes2, layer_sizes3]

    views = [torch.from_numpy(Ia_data[:100000]).double(),
             torch.from_numpy(Ie_data[:100000]).double(),
             torch.from_numpy(AC_data[:100000]).double()]

    print(f'input views shape :')
    for i, view in enumerate(views):
        print(f'view_{i} :  {view.shape}')
        view = view.to(device)

    # size of the input for view 1 and view 2
    input_shape_list = [view.shape[-1] for view in views]
    # Building, training, and producing the new features by DCCA
    model = DeepGCCA(layer_sizes_list, input_shape_list, outdim_size,
                     use_all_singular_values, device=device).double()
    # for name, param in model.named_parameters():
    #     print(name,param.shape, param)

    l_gcca = None
    if apply_linear_gcca:
        l_gcca = linear_gcca
    solver = Solver(model, l_gcca, outdim_size, epoch_num, batch_size,
                    learning_rate, reg_par, device=device)
    solver.fit(views, checkpoint=save_name)
    loss, outputs = solver.test(views, apply_linear_gcca)

    embed_Xs = torch.cat((outputs), 1).cpu().numpy()
    run = 0
    for train_index, test_index in StratifiedShuffleSplit(n_splits=10,test_size=0.2,random_state=0).split(embed_Xs, Y_label[:100000]):
        run += 1
        x_train, y_train = embed_Xs[train_index], Y_label[train_index]
        x_test,  y_test  = embed_Xs[test_index],  Y_label[test_index]

        knn_classifier = KNeighborsClassifier(n_neighbors=1)
        knn_classifier.fit(x_train, y_train)
        y_predict = knn_classifier.predict(x_test)
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