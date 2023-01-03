import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs: input data
        :param ys: prediction data
        :param batch_size:
        :param pad_with_last_sample: 
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0) 
            y_padding = np.repeat(ys[-1:], num_padding, axis=0) 
            xs = np.concatenate([xs, x_padding], axis=0) 
            ys = np.concatenate([ys, y_padding], axis=0) 
        self.size = len(xs) 
        self.num_batch = int(self.size // self.batch_size) 
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation] 
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0
        
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind 
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()
    
class StandardScaler():

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std 

    def inverse_transform(self, data): 
        return (data * self.std) + self.mean 



def sym_adj(adj):
    adj = sp.coo_matrix(adj) 
    rowsum = np.array(adj.sum(1)) 
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() 
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0. 
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()
            # (A D^-1/2)^T D^-1/2

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0. 
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj) 
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten() 
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) 
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    # I - (A D^-1/2)^T D^-1/2 = I - D^-1/2 A D^-1/2
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True): 
    if undirected: 
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T]) 
    L = calculate_normalized_laplacian(adj_mx) 
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM') 
        lambda_max = lambda_max[0] 
    L = sp.csr_matrix(L) 
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype) 
    L = (2 / lambda_max * L) - I 
    return L.astype(np.float32).todense() 

def load_pickle(pickle_file): 
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e: 
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def AtoAe(adj_mx):

    adj_mx=adj_mx- np.identity(adj_mx.shape[0])
    adj_one = np.where(adj_mx > 0, 1.0, 0.0) 
    e = int((len(np.nonzero(adj_one)[0]))) 
    Ae = np.zeros((e, e))
    M = np.zeros((adj_mx.shape[0], e))

    line=[]
    column=[]
    a=0
    for i in range(adj_one.shape[0]): 
        for j in range(adj_one.shape[1]):
            if(adj_one[i][j]!=0.0):
                line.append(i)
                column.append(j)
                M[i, a] = 1.0
                M[j, a] = 1.0
                a = a + 1  


    node = [] 
    for n in range(adj_one.shape[0]):
        node.append(sum(adj_one[n,:]))
    for n in range(adj_one.shape[1]):
        node.append(sum(adj_one[:,n]))
    std= np.std(node, ddof=1)

    for k1 in range(e):
        for k2 in range(e):
            if line[k1]==column[k2]:
                Ae_synergy=sum(adj_one[:,line[k1]])+sum(adj_one[line[k1],:])
                Ae_synergy=np.exp(-(((Ae_synergy - 2) ** 2)/(std**2)))
                Ae[k1,k2]=Ae_synergy
            if column[k1]==column[k2]:
                Ae_competition=sum(adj_one[line[k1],:])+sum(adj_one[line[k2],:])
                Ae_competition = np.exp(-(((Ae_competition - 2) ** 2) / (std ** 2)))
                Ae[k1, k2]=Ae_competition

    for k in range(e):
        Ae[k, k]=1.0 

    return Ae,M


def load_adj(pkl_filename):
    _,_,adj_mx = load_pickle(pkl_filename) 

    Ae, M = AtoAe(adj_mx) 
    
    adj = asym_adj(adj_mx)
    Ae = asym_adj(Ae)

    adj = torch.from_numpy(adj.astype(np.float32))
    Ae = torch.from_numpy(Ae.astype(np.float32))
    M = torch.from_numpy(M.astype(np.float32))

    return adj, Ae, M 


def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz')) 
        data['x_' + category] = cat_data['x'] 
        data['y_' + category] = cat_data['y'] 
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std()) 

    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size) 
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler 
    return data

def masked_mse(preds, labels, null_val=np.nan): 
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2 
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan): 
    if np.isnan(null_val):
        mask = ~torch.isnan(labels) 
    else: 
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels) 
    loss = loss * mask 
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss) 


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real): 
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse


