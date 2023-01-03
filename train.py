import torch
import numpy as np
import argparse
import time
import util
import os
from engine import trainer

# 8(17856, 170, 3)

parser = argparse.ArgumentParser()

parser.add_argument('--device',type=str,default='0',help='') 
parser.add_argument('--data',type=str,default='data/PEMS08',help='data path') 
parser.add_argument('--adjdata',type=str,default='data/PEMS08/adj_pems08.pkl',help='adj data path')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension') 
parser.add_argument('--input_dim',type=int,default=1,help='')
parser.add_argument('--num_nodes',type=int,default=170,help='')

parser.add_argument('--batch_size',type=int,default=64,help='batch size') 

parser.add_argument('--save',type=str,default='./garage/pems08',help='save path') 

parser.add_argument('--seq_length',type=int,default=12,help='') 
parser.add_argument('--nhid',type=int,default=64,help='') 
parser.add_argument('--learning_rate',type=float,default=0.01,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.0002,help='weight decay rate') 
parser.add_argument('--epochs',type=int,default=100,help='') 
parser.add_argument('--cl_decay_steps',type=int,default=2000,help='')
parser.add_argument('--print_every',type=int,default=100,help='')

parser.add_argument('--expid',type=int,default=1,help='experiment id') 
parser.add_argument('--blocks',type=int,default=2,help=)
parser.add_argument('--layers',type=int,default=3,help=)
parser.add_argument('--lr_decay_rate',type=int,default=0.6,help=)
parser.add_argument('--lr_step_size',type=int,default=10,help=)

parser.add_argument('--seq_len',type=int,default=12,help='')
parser.add_argument('--output_dim',type=int,default=1,help='')
parser.add_argument('--horizon',type=int,default=12,help='')
parser.add_argument('--rnn_units',type=int,default=64,help='')
parser.add_argument('--num_rnn_blocks',type=int,default=2,help='')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

def setup_seed(seed):
    # random.seed(seed)  # Python random module.
    # os.environ['PYTHONHASHSEED'] = str(seed)  
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True # cuDNN
seed = 1
setup_seed(seed)  

def main():

    #load data
    adj_mx, Ae, M = util.load_adj(args.adjdata) 
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    supports = [adj_mx.cuda()] 
    supports = supports + [Ae.cuda()]  
    M = M.cuda()

    engine = trainer(args, scaler, supports, M)

    print("start training...",flush=True)
    his_loss =[]
    val_time = []
    train_time = []
    batches_seen =  320*64
    for i in range(1,args.epochs+1): # epochs start
        #if i % 10 == 0:
            #lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
            #for g in engine.optimizer.param_groups:
                #g['lr'] = lr
        train_loss = [] 
        train_mape = [] 
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle() 
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()): 
            trainx = torch.Tensor(x).cuda()
            trainx= trainx.transpose(1, 3) 
            trainy = torch.Tensor(y).cuda()
            trainy = trainy.transpose(1, 3) 
            metrics = engine.train(trainx, trainy[:,0,:,:] ,batches_seen) 
            train_loss.append(metrics[0]) 
            train_mape.append(metrics[1]) 
            train_rmse.append(metrics[2]) 
            batches_seen += 1
            if iter % args.print_every == 0 : 
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
        t2 = time.time()
        train_time.append(t2-t1)
    
        engine.scheduler.step()  
    
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
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss) 
        mtrain_mape = np.mean(train_mape) 
        mtrain_rmse = np.mean(train_rmse)
    
        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss) 
    
        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
        torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time))) 




    bestid = np.argmin(his_loss) 
    engine.model.load_state_dict(torch.load(args.save+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))
    engine.model.eval()

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).cuda()
    realy = realy.transpose(1,3)[:,0,:,:]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).cuda()
        testx = testx.transpose(1,3) 
        with torch.no_grad(): 
            preds = engine.model(testx).transpose(1,3)
        outputs.append(preds.squeeze()) 

    yhat = torch.cat(outputs,dim=0) 
    yhat = yhat[:realy.size(0),...] 

    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid],4)))
    
    amae = []
    amape = []
    armse = []
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:,:,i]) 
        real = realy[:,:,i] 
        metrics = util.metric(pred,real) 
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
   

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    torch.cuda.empty_cache() 
    print("Total time spent: {:.4f}".format(t2-t1)) 
