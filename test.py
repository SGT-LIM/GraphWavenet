import util
import argparse
from model import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='data/2022년_부천시_대기행렬예측/대기행렬로_대기행렬예측',help='data path') #MA 안한걸로 test돌려야!
parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adj_mx_566.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension') #2
parser.add_argument('--num_nodes',type=int,default=572,help='number of nodes') #214
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--checkpoint',type=str,help='')
parser.add_argument('--plotheatmap',type=str,default='True',help='')


args = parser.parse_args()




def main():
    device = torch.device(args.device)

    _, _, adj_mx = util.load_adj(args.adjdata,args.adjtype) #adj_max는 Nx2N 인듯..?? 두 개의 NxN 행렬 나열해놓은거.
    print(len(adj_mx[0])) #원래는 207개임!
    supports = [torch.tensor(i).to(device) for i in adj_mx] #i는 adj_mx에 있는 두 행렬을 하나씩 순서대로 ㅇㅇ. torch.tensor(i)는 그 행렬을 tensor로 바꿔주는 거(모양은 그대로). 
    #그리고 cuda:0에 올리는거(?). supports는 그 두개의 tensor로 변환된 행렬들을 list에 담는거. 
    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0] #두 개의 행렬이 있는 리스트 중 첫번째거.

    if args.aptonly:
        supports = None


    
    model =  gwnet(device, args.num_nodes, args.dropout, supports=supports, gcn_bool=args.gcn_bool, addaptadj=args.addaptadj, aptinit=adjinit, in_dim=1)
    #model =  gwnet(device, args.num_nodes, args.dropout, supports=supports, gcn_bool=args.gcn_bool, addaptadj=args.addaptadj, aptinit=adjinit, in_dim=2, out_dim=12, residual_channels=32, dilation_channels=32, skip_channels=32 * 8, end_channels=32 * 16)
    #model = nn.DataParallel(model)

    model.to(device)
    #model.load_state_dict(torch.load(args.checkpoint), strict=False) 
    model.load_state_dict(torch.load("./train결과pth/대기행렬로_대기행렬예측/대기행렬로_대기행렬예측_epoch_2_0.0.pth"), strict=False)
    model.eval()
    
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    #print(model.names)

    print('model load successfully')

    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device) #size가 (157,12,47,2)
    
    realy = realy.transpose(1,3)[:,0,:,:] #(157,12,47,2) -> (157,2,47,12) -> (157,47,12). 0은 속도와 다른 변수들 중 속도만 취하기 위함인듯..?
    print('realy shape = ',realy.shape) #(157,47,12)

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds = model(testx).transpose(1,3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]
    print('yhat shape = ',yhat.shape)

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

    """
    if args.plotheatmap == "True":
        adp = F.softmax(F.relu(torch.mm(model.nodevec1, model.nodevec2)), dim=1)
        device = torch.device('cpu')
        adp.to(device)
        adp = adp.cpu().detach().numpy()
        adp = adp*(1/np.max(adp))
        df = pd.DataFrame(adp)
        sns.heatmap(df, cmap="RdYlBu")
        plt.savefig("./self_adaptive_adjacency_matrix/only_vel_인접행렬내가만듦_직접보고만든_imade_data47_without_dist_addaptadj_false"+ '.pdf')
    """
    for i in range(47):
        y12 = realy[:,i,11].cpu().detach().numpy()
        yhat12 = scaler.inverse_transform(yhat[:,i,11]).cpu().detach().numpy()

        y6 = realy[:,i,5].cpu().detach().numpy()
        yhat6 = scaler.inverse_transform(yhat[:,i,5]).cpu().detach().numpy()

        y3 = realy[:,i,2].cpu().detach().numpy()
        yhat3 = scaler.inverse_transform(yhat[:,i,2]).cpu().detach().numpy()

        df2 = pd.DataFrame({'real12':y12,'pred12':yhat12, 'real6':y6,'pred6':yhat6, 'real3': y3, 'pred3':yhat3})
        df2.to_csv('./Bucheon_214개_1월_test결과/2일치 대기행렬로 속도 예측/only_vel_인접행렬내가만듦/직접보고만든_extreme(1, 0)without_dist_addaptadj_false/Bucheon_onlyvel_imade_data47_withoutdist_withoutaddapt'+str(i)+'.csv',index=False)
  
   




if __name__ == "__main__":
    main()
