import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs: 输入数据
        :param ys: 预测数据
        :param batch_size: 批大小
        :param pad_with_last_sample: 填充最后一个样本，使样本数可被batch_size分割。
        """
        self.batch_size = batch_size
        self.current_ind = 0 # 当前batch
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size # 要填充的样本数
            x_padding = np.repeat(xs[-1:], num_padding, axis=0) # 最后一行数据，复制，填充
            y_padding = np.repeat(ys[-1:], num_padding, axis=0) # 最后一行数据，复制，填充
            xs = np.concatenate([xs, x_padding], axis=0) # 拼接到后面
            ys = np.concatenate([ys, y_padding], axis=0) # 拼接到后面
        self.size = len(xs) # 计算长度
        self.num_batch = int(self.size // self.batch_size) # //是整除，求取有多少个batch
        self.xs = xs
        self.ys = ys

    def shuffle(self): # 洗牌
        permutation = np.random.permutation(self.size) # 随机排列序列
        xs, ys = self.xs[permutation], self.ys[permutation] # 随机排序
        self.xs = xs
        self.ys = ys

    def get_iterator(self): # 迭代生成器
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch: # 小于batch数
                start_ind = self.batch_size * self.current_ind # 这个batch开始的起始索引位置
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1)) # 结束位置，min比较是避免没有填充的情况
                x_i = self.xs[start_ind: end_ind, ...] # 该batch输入数据
                y_i = self.ys[start_ind: end_ind, ...] # 该batch预测数据
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper() # 返回迭代的生成器数据

class StandardScaler():
    """
    标准输入
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std # (数据-平均值)/方差

    def inverse_transform(self, data): # 跟上面反着变换
        return (data * self.std) + self.mean # 数据*方差+均值



def sym_adj(adj):
    """对称规格化邻接矩阵"""
    adj = sp.coo_matrix(adj) # 变成稀疏矩阵表示
    rowsum = np.array(adj.sum(1)) # 每一行求和
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() # 求-0.5次方
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0. # 这句就是将数组中无穷大的元素置0处理
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # 稀疏对角矩阵
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()
            # (A D^-1/2)^T D^-1/2

def asym_adj(adj):
    adj = sp.coo_matrix(adj) # 变成稀疏矩阵表示
    rowsum = np.array(adj.sum(1)).flatten() # 每一行求和
    d_inv = np.power(rowsum, -1).flatten() # 求-1次方
    d_inv[np.isinf(d_inv)] = 0. # 这句就是将数组中无穷大的元素置0处理
    d_mat= sp.diags(d_inv) # 稀疏对角矩阵
    return d_mat.dot(adj).astype(np.float32).todense() # D^-1 A

def calculate_normalized_laplacian(adj): # 计算归一化拉普拉斯算子
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj) # 变成稀疏矩阵表示
    d = np.array(adj.sum(1)) # 每一行求和
    d_inv_sqrt = np.power(d, -0.5).flatten() # 求-0.5次方
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0. # 这句就是将数组中无穷大的元素置0处理
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # 稀疏对角矩阵
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    # I - (A D^-1/2)^T D^-1/2 = I - D^-1/2 A D^-1/2
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True): # 计算按比例缩小的拉普拉斯算子
    if undirected: # 无向
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T]) # 我觉得这是将有向图弄成无向图
    L = calculate_normalized_laplacian(adj_mx) # 计算归一化拉普拉斯算子
    if lambda_max is None: # 设置为None的话
        lambda_max, _ = linalg.eigsh(L, 1, which='LM') # 找到L中最大的特征值和特征向量
        lambda_max = lambda_max[0] # 从列表中提取出值
    L = sp.csr_matrix(L) # L转化为csr稀疏矩阵
    M, _ = L.shape # 得到矩阵维度数
    I = sp.identity(M, format='csr', dtype=L.dtype) # 创建单位矩阵
    L = (2 / lambda_max * L) - I # 对拉普拉斯矩阵进行缩放
    return L.astype(np.float32).todense() # 还原

def load_pickle(pickle_file): # 读取pkl的邻接矩阵文件
    try:
        with open(pickle_file, 'rb') as f: # 打开文件都用此语句避免没有关闭文件发生泄露
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e: # 脚本程序中编码与文件编码不一致的异常
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e) # 无法加载数据
        raise
    return pickle_data

def AtoAe(adj_mx):

    adj_mx=adj_mx- np.identity(adj_mx.shape[0]) # 去掉对角自环
    adj_one = np.where(adj_mx > 0, 1.0, 0.0) # 换成01矩阵
    e = int((len(np.nonzero(adj_one)[0]))) # 有多少个边,1515
    Ae = np.zeros((e, e))
    M = np.zeros((adj_mx.shape[0], e))

    # 我们默认行为起点列为终点
    line=[]
    column=[]
    a=0
    for i in range(adj_one.shape[0]): # 第i行
        for j in range(adj_one.shape[1]): # 第j列
            if(adj_one[i][j]!=0.0): # 找到非0的那些有向边
                line.append(i)
                column.append(j)
                M[i, a] = 1.0
                M[j, a] = 1.0
                a = a + 1  # 控制换行


    node = []  # 节点度
    for n in range(adj_one.shape[0]):
        node.append(sum(adj_one[n,:]))
    for n in range(adj_one.shape[1]):
        node.append(sum(adj_one[:,n]))
    std= np.std(node, ddof=1)

    for k1 in range(e):# 边的行
        for k2 in range(e):# 边的列
            if line[k1]==column[k2]:# 两个边的起点和终点相同,协同关系
                Ae_synergy=sum(adj_one[:,line[k1]])+sum(adj_one[line[k1],:])
                Ae_synergy=np.exp(-(((Ae_synergy - 2) ** 2)/(std**2)))
                Ae[k1,k2]=Ae_synergy
            if column[k1]==column[k2]:# 如果两个边终点相同,竞争关系
                Ae_competition=sum(adj_one[line[k1],:])+sum(adj_one[line[k2],:])
                Ae_competition = np.exp(-(((Ae_competition - 2) ** 2) / (std ** 2)))
                Ae[k1, k2]=Ae_competition

    for k in range(e):
        Ae[k, k]=1.0 #在原来矩阵基础上加上自环单位矩阵

    return Ae,M


def load_adj(pkl_filename):
    """
    读取邻接矩阵
    :param pkl_filename: 文件位置
    :param adjtype: 邻接矩阵类型，双向否？
    :return:
    """
    # sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename) # 传感器id，传感器id对应编号字典，路网邻接矩阵
    _,_,adj_mx = load_pickle(pkl_filename) # 传感器id，传感器id对应编号字典，路网邻接矩阵

    Ae, M = AtoAe(adj_mx) # 返回边图矩阵和边个数

    # 过渡矩阵 D^-1 A
    adj = asym_adj(adj_mx)
    Ae = asym_adj(Ae)

    adj = torch.from_numpy(adj.astype(np.float32))
    Ae = torch.from_numpy(Ae.astype(np.float32))
    M = torch.from_numpy(M.astype(np.float32))

    return adj, Ae, M  # 传感器id，传感器id对应编号字典，处理后的邻接矩阵


def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    '''

    :param dataset_dir: 数据地址
    :param batch_size: 批量大小
    :param valid_batch_size: 验证集批量大小
    :param test_batch_size: 测试集批量大小
    :return:
    '''
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz')) # 读取数据
        data['x_' + category] = cat_data['x'] # 输入
        data['y_' + category] = cat_data['y'] # 预测
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std()) # 训练的输入数据构成类
    # 数据格式
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0]) # 对输入数据标准化处理
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size) # 数据加载
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler # 传了一个类？
    return data

def masked_mse(preds, labels, null_val=np.nan): # MSE
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2 # 求MSE
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan): # MAE，null_val是空缺的脏数据的形式
    if np.isnan(null_val): # null_val是否是nan
        mask = ~torch.isnan(labels) # 判断label里面是否有空缺的，有的话该位置mask为0，其余为1
    else: # 其他脏数据形式，而不是nan
        mask = (labels!=null_val) # 是该形式的脏数据的位置mask为0，其余为1
    mask = mask.float()
    mask /=  torch.mean((mask)) # 每个mask标签除以他们整体的平均值
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    # where第一个是判断条件，第二个是符合条件的设置值，第三个是不满足条件的设置值。
    # 第一个是判断是否有空缺数据，有的话是1，没有是0。第二个是生成全0张量，第三个是按照原来值。
    # 这里出现是避免有其他形式脏数据和空缺形式脏数据同时出现，所以采用这个方法来如果选择其他形式脏数据mask后再筛一遍空缺脏数据。
    loss = torch.abs(preds-labels) # 预测值与真实值的绝对值
    loss = loss * mask # 乘以mask，消去脏数据影响
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    # 如果loss里面有缺失值，则该位置赋0，不然就还是原值
    return torch.mean(loss) # loss求平均


def masked_mape(preds, labels, null_val=np.nan): # MAPE
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels # 求MAPE
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real): # 度量
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse


