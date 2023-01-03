import torch
import numpy as np
import argparse
import time
import util
import os
from engine import trainer

# 8(17856, 170, 3)

parser = argparse.ArgumentParser()

parser.add_argument('--device',type=str,default='1',help='') # 训练显卡号
parser.add_argument('--data',type=str,default='data/PEMS08',help='data path') # 数据地址
parser.add_argument('--adjdata',type=str,default='data/PEMS08/adj_pems08.pkl',help='adj data path') # 路网邻接矩阵地址
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension') # 输入维度
parser.add_argument('--input_dim',type=int,default=1,help='')
parser.add_argument('--num_nodes',type=int,default=170,help='')

parser.add_argument('--batch_size',type=int,default=56,help='batch size') # 批量大小,默认64

parser.add_argument('--save',type=str,default='./garage/metr7',help='save path') # 模型保存地址

parser.add_argument('--seq_length',type=int,default=12,help='') # 序列长度
parser.add_argument('--nhid',type=int,default=64,help='') # 中间卷积设置的通道维度
parser.add_argument('--learning_rate',type=float,default=0.01,help='learning rate') # 学习率
parser.add_argument('--weight_decay',type=float,default=0.0002,help='weight decay rate') # 权重衰减率
parser.add_argument('--epochs',type=int,default=100,help='') # 原先默认值100
parser.add_argument('--cl_decay_steps',type=int,default=2000,help='')# 2000
parser.add_argument('--print_every',type=int,default=100,help='') # 隔多少个打印

parser.add_argument('--expid',type=int,default=1,help='experiment id') # 实验id，这个设置得莫名其妙
parser.add_argument('--blocks',type=int,default=2,help='experiment id')
parser.add_argument('--layers',type=int,default=3,help='experiment id')
parser.add_argument('--lr_decay_rate',type=int,default=0.6,help='experiment id')
parser.add_argument('--lr_step_size',type=int,default=10,help='experiment id')

parser.add_argument('--seq_len',type=int,default=12,help='')
parser.add_argument('--output_dim',type=int,default=1,help='')
parser.add_argument('--horizon',type=int,default=12,help='')
parser.add_argument('--rnn_units',type=int,default=64,help='')
parser.add_argument('--num_rnn_blocks',type=int,default=2,help='')


args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 按照PCI_BUS_ID顺序从0开始排列GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

def setup_seed(seed):
    # random.seed(seed)  # Python random module.
    # os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True # 上面两句是设置cuDNN
seed = 530302
print("随机种子数seed:", seed)
setup_seed(seed)  # 设置随机数种子

def main():

    #load data
    adj_mx, Ae, M = util.load_adj(args.adjdata) # 读取邻接矩阵
    # 得到 传感器id，传感器id对应编号字典，处理后的邻接矩阵
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size) # 读取数据
    '''''
    dataloader['train_loader'] # 训练数据加载
    dataloader['val_loader'] # 验证数据加载
    dataloader['test_loader'] # 测试数据加载
    dataloader['scaler'] = scaler # 传了一个类？
    '''''
    scaler = dataloader['scaler'] # 数据标准化的类
    supports = [adj_mx.cuda()] # 将矩阵转化为tensor形式并放到GPU上
    supports = supports + [Ae.cuda()]  # 将矩阵转化为tensor形式并放到GPU上
    # supports[0]是图邻接矩阵,supports[1]是边图邻接矩阵
    M = M.cuda()

    print(args) # 显示设置的parser内容

    engine = trainer(args, scaler, supports, M)

    print("start training...",flush=True) # flush=True就是将这个东西及时输出
    his_loss =[]
    val_time = []
    train_time = []
    batches_seen =  320*64
    for i in range(1,args.epochs+1): # epochs开始
        #if i % 10 == 0:
            #lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
            #for g in engine.optimizer.param_groups:
                #g['lr'] = lr
        train_loss = [] # 训练损失
        train_mape = [] # 训练MAPE
        train_rmse = [] # 训练RMSE
        t1 = time.time()
        dataloader['train_loader'].shuffle() # 将训练数据集随机排列
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()): # 返回当前batch数iter和输入x与预测y
            trainx = torch.Tensor(x).cuda()
            # trainx.Size是[64, 12, 207, 2]
            trainx= trainx.transpose(1, 3) # 1轴和3轴交换
            # trainx.Size是[64, 2, 207, 12]
            trainy = torch.Tensor(y).cuda()
            trainy = trainy.transpose(1, 3) # 1轴和3轴交换
            metrics = engine.train(trainx, trainy[:,0,:,:] ,batches_seen) # 训练，0那个应该是速度
            train_loss.append(metrics[0]) # 训练的loxx
            train_mape.append(metrics[1]) # 训练的MAPE
            train_rmse.append(metrics[2]) # 训练的RMSE
            batches_seen += 1
            if iter % args.print_every == 0 : # 达到打印周期
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}' # 当前batch，损失，MAPE，RMSE
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
        t2 = time.time()
        train_time.append(t2-t1)
    
        engine.scheduler.step()  # 学习率衰减 ##修改过
    
        #验证
        valid_loss = []
        valid_mape = []
        valid_rmse = []
    
        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).cuda()
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).cuda()
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:,0,:,:])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs' # 当前Epoch，推理时间
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss) # 在这个epoch中每个batch平均训练损失
        mtrain_mape = np.mean(train_mape) # 在这个epoch中每个batch平均训练MAPE
        mtrain_rmse = np.mean(train_rmse) # 在这个epoch中每个batch平均训练RMSE
    
        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss) # 存放在每个epoch中每个batch平均验证损失
    
        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        # 当前epoch，在这个epoch中平均训练损失、MAPE、RMSE，平均验证损失、MAPE、RMSE，这个epoch中训练花费的时间
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
        torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
        # 存放网络中的权重，保存地址为 ./garage/metr_epoch_[epoch数]_[这个epoch下验证loss值，只保留两位小数].pth
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time))) # 平均训练时间
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time))) # 平均验证推理时间



    #测试
    bestid = np.argmin(his_loss) # 找到最小loss时的下标，即那个epoch时的值
    engine.model.load_state_dict(torch.load(args.save+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))
    # 读取那个最佳epoch时的模型参数
    engine.model.eval()

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).cuda()
    realy = realy.transpose(1,3)[:,0,:,:] # 测试集的真实值

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).cuda()
        testx = testx.transpose(1,3) # 测试集的输入
        # testy = torch.Tensor(y).cuda()
        # testy = testy.transpose(1, 3)
        with torch.no_grad(): # 不会对其下面的求导，关闭反向传播
            # preds = engine.model(testx, testy[:,0,:,:]).transpose(1,3)
            preds = engine.model(testx).transpose(1,3)
        outputs.append(preds.squeeze()) # 去除size为1的维度

    yhat = torch.cat(outputs,dim=0) # 按dim=0竖着拼接，模型预测结果
    yhat = yhat[:realy.size(0),...] # 没懂这是干嘛

    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid],4)))
    
    amae = []
    amape = []
    armse = []
    for i in range(12): # 预测12帧，1个小时
        pred = scaler.inverse_transform(yhat[:,:,i]) # 输出预测值
        real = realy[:,:,i] # 真实值
        metrics = util.metric(pred,real) # 计算指标
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        # 验证集期间最佳模型的测试集第i帧，测试MAE，MAPE，RMSE
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    # 12帧的平均测试值，MAE，MAPE，RMSE
    print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
    # torch.save(engine.model.state_dict(), args.save+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")
    # 保存这个模型，位置为     ./garage/metr_exp1_best_[这个模型下验证loss值，只保留两位小数].pth


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    torch.cuda.empty_cache() # 释放显存
    print("Total time spent: {:.4f}".format(t2-t1)) # 总时间消耗
