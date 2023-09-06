import torch
import random
import os
import numpy as np
from utils.models.unet import U_Net
import matplotlib.pyplot as plt
from utils.functions import destandarize
from utils.load_data import MyDataLoader
from matplotlib.ticker import MaxNLocator
import argparse

from utils.models import engine

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Using CPU')
else:
    print('CUDA is available. Using GPU')

device = torch.device("cuda:0" if train_on_gpu else "cpu")

def eval(opt):
    # LOAD DATA
    my_data_loader = MyDataLoader()
    x_test, y_test, y_mean, y_std, fs_test, r_test, z_test= my_data_loader.load_test_data()

    x_test = torch.tensor(x_test).float().to(device)
    y_test = torch.tensor(y_test).float().to(device)

    ####LOAD######
    model = U_Net(n1=opt.num_filters, dropout_rate=opt.dropout)
    model = torch.load(opt.weights)
    model.to(device)
    model.eval()

    # Eval 
    with torch.no_grad():
            output = model(x_test)

    x_test = x_test.cpu()
    y_test = y_test.cpu()

    y_test_pred = destandarize(output, y_mean, y_std)
    y_test_pred = y_test_pred.squeeze(1)

    y_test_pred = y_test_pred.cpu()

    #N = random.randint(0,len(x_test)-1)
    N = 11

    y_test[N][fs_test[N] <= 0.05e-6] = 0
    y_test_pred[N][fs_test[N] <= 0.05e-6] = 0
    print(N)

    plt.rcParams['figure.figsize'] = [12, 4]

    plt.subplot(171)
    plt.imshow(x_test[0,0,:,:], cmap = 'jet')#, vmax=x_test.max() , vmin=x_test.min())
    plt.title('$U-Net$')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(172)
    plt.imshow(x_test[0,1,:,:], cmap = 'jet')#, vmax=x_test.max(), vmin=x_test.min())
    plt.title('$U-Net$')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(173)
    plt.imshow(x_test[0,2,:,:], cmap = 'jet')#, vmax=x_test.max(), vmin=x_test.min())
    plt.title('$U-Net$')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(174)
    plt.imshow(y_test[N,:,:], cmap = 'jet', vmin=1500, vmax=2205)
    plt.title('$Groundtruth$')
    plt.colorbar()

    plt.subplot(175)
    plt.imshow(y_test_pred[N], cmap = 'jet', vmin=1500, vmax=2205)
    plt.title('$U-Net$')
    plt.colorbar()
    plt.tight_layout()

    abs_err = np.abs(y_test[N] - y_test_pred[N])
    plt.subplot(176)
    plt.imshow(abs_err, cmap = 'jet', vmin=0, vmax=50)
    plt.title('$\Delta T_{s}$')
    plt.colorbar(ticks=MaxNLocator(6))

    print('Abs. error max:', abs_err.max())
    print('Abs. error min:', abs_err.min())
    print('Abs. error mean:', abs_err.mean())
    print('Abs. error stddev:', abs_err.std())
    print('Abs. error %:', abs_err.mean()*100/y_test[N].mean())

    plt.show()

def eval_exp(opt):
    print("experiment")
    # LOAD DATA
    my_data_loader = MyDataLoader()
    _, _, y_mean, y_std, _, _, _= my_data_loader.load_test_data()
    Py_exp_interp,t_emi,t_bemi, r_emi, z_emi, t_emi, r, z = my_data_loader.load_data_exp()

    Py_exp_interp = torch.tensor(Py_exp_interp).float().to(device)


    ####LOAD######
    model = U_Net(n1=opt.num_filters, dropout_rate=opt.dropout)
    model = torch.load(opt.weights)
    model.to(device)
    model.eval()

    # Eval 
    with torch.no_grad():
            output = model(Py_exp_interp)

    Py_exp_interp = Py_exp_interp.cpu().numpy()

    print("ourput shape: ", output.shape)

    t_cgan_caseC = destandarize(output, y_mean, y_std)[0,0,:,:]
    t_cgan_caseC= t_cgan_caseC.cpu().numpy()

    mask = t_emi<1
    t_emi = np.ma.masked_where(mask, t_emi)
    t_bemi = np.ma.masked_where(mask, t_bemi)
    print("t_cgan_caseC shape: ", t_cgan_caseC.shape)
    t_cgan_caseC = t_cgan_caseC[::-1] #torch.flip(t_cgan_caseC, [0])

    t_cgan_caseC = np.ma.masked_where(mask,t_cgan_caseC)
    for i in range(3):
        Py_exp_interp[0,i,:,:] = np.ma.masked_where(mask,Py_exp_interp[0,i,:,:])

    plt.rcParams['figure.figsize'] = [10, 4]

    fig, ax = plt.subplots(1,6)
    axcontourf(ax[0],r,z, Py_exp_interp[0,0,:,:][::-1], 'R')
    axcontourf(ax[1],r,z, Py_exp_interp[0,1,:,:][::-1], 'G')
    axcontourf(ax[2],r,z, Py_exp_interp[0,2,:,:][::-1], 'B')
    
    im1=axcontourf(ax[3],r_emi,z_emi, t_emi,r'$T_{s}$(EMI)',levels=np.linspace(1500,2100,50))
    im2=axcontourf(ax[4],r,z, t_bemi,r'$T_{s}$ (BEMI)',levels=np.linspace(1500,2100,50))
    im3=axcontourf(ax[5],r,z,t_cgan_caseC,r'$T_{s}$(U-Net)',levels=np.linspace(1500,2100,50))

    ax[0].set_facecolor("darkblue")  
    ax[1].set_facecolor("darkblue")  
    ax[2].set_facecolor("darkblue")  
    ax[3].set_facecolor("darkblue")  
    ax[4].set_facecolor("darkblue")  
    ax[5].set_facecolor("darkblue")

    fig.colorbar(im1, ticks=MaxNLocator(6))
    fig.colorbar(im2, ticks=MaxNLocator(6))
    fig.colorbar(im3, ticks=MaxNLocator(6))
    fig.tight_layout()
 
    plt.rcParams['figure.figsize'] = [6, 4]
    fig, ax = plt.subplots(1,3)
    im1=axcontourf(ax[0],r_emi,z_emi, t_emi,r'$T_{s}(EMI)$',levels= np.linspace(1500,2100,50))
    im2=axcontourf(ax[1],r,z, t_bemi - t_emi,r'$\Delta_t$ BEMI',levels= np.linspace(-80,80,50),CMAP='bwr')
    abs_err = t_cgan_caseC - t_emi
    im3=axcontourf(ax[2],r,z,abs_err,'$\Delta_t$ U-Net',levels= np.linspace(-80,80,50),CMAP='bwr')
    abs_err[abs_err>100] = 100
    abs_err[abs_err<-100] = -100
    
    ax[0].set_facecolor("darkblue")  
    ax[1].set_facecolor("darkblue")  
    ax[2].set_facecolor("darkblue")  
    fig.colorbar(im1, ticks=MaxNLocator(6))
    fig.colorbar(im2, ticks=MaxNLocator(6))
    fig.colorbar(im3, ticks=MaxNLocator(6))
    fig.tight_layout()

    print('Abs. error max:', abs_err.max())
    print('Abs. error min:', abs_err.min())
    print('Abs. error mean:', abs_err.mean())
    print('Abs. error stddev:', abs_err.std())
    print('Abs. error %:', abs_err.mean()*100/t_emi.mean())

    plt.show()

def eval_exp_TRT(opt):
    print("experiment")
    # LOAD DATA
    my_data_loader = MyDataLoader()
    _, _, y_mean, y_std, _, _, _= my_data_loader.load_test_data()
    Py_exp_interp,t_emi,t_bemi, r_emi, z_emi, t_emi, r, z = my_data_loader.load_data_exp()

    Py_exp_interp = torch.tensor(Py_exp_interp).float().to(device)


    ####LOAD######
    current_directory = os.path.dirname(os.path.abspath(__file__))
    engine_path = os.path.join(current_directory,opt.weights)

    Engine = engine.TRTModule(engine_path, device)
    Engine.set_desired(['outputs'])
    model = Engine

    # Eval 
    with torch.set_grad_enabled(False):
            output = model(Py_exp_interp)

    Py_exp_interp = Py_exp_interp.cpu().numpy()

    print("ourput shape: ", output.shape)

    t_cgan_caseC = destandarize(output, y_mean, y_std)[0,0,:,:]
    t_cgan_caseC= t_cgan_caseC.cpu().numpy()

    mask = t_emi<1
    t_emi = np.ma.masked_where(mask, t_emi)
    t_bemi = np.ma.masked_where(mask, t_bemi)
    print("t_cgan_caseC shape: ", t_cgan_caseC.shape)
    t_cgan_caseC = t_cgan_caseC[::-1] #torch.flip(t_cgan_caseC, [0])

    t_cgan_caseC = np.ma.masked_where(mask,t_cgan_caseC)
    for i in range(3):
        Py_exp_interp[0,i,:,:] = np.ma.masked_where(mask,Py_exp_interp[0,i,:,:])

    plt.rcParams['figure.figsize'] = [10, 4]

    fig, ax = plt.subplots(1,6)
    axcontourf(ax[0],r,z, Py_exp_interp[0,0,:,:][::-1], 'R')
    axcontourf(ax[1],r,z, Py_exp_interp[0,1,:,:][::-1], 'G')
    axcontourf(ax[2],r,z, Py_exp_interp[0,2,:,:][::-1], 'B')
    
    im1=axcontourf(ax[3],r_emi,z_emi, t_emi,r'$T_{s}$(EMI)',levels=np.linspace(1500,2100,50))
    im2=axcontourf(ax[4],r,z, t_bemi,r'$T_{s}$ (BEMI)',levels=np.linspace(1500,2100,50))
    im3=axcontourf(ax[5],r,z,t_cgan_caseC,r'$T_{s}$(U-Net)',levels=np.linspace(1500,2100,50))

    ax[0].set_facecolor("darkblue")  
    ax[1].set_facecolor("darkblue")  
    ax[2].set_facecolor("darkblue")  
    ax[3].set_facecolor("darkblue")  
    ax[4].set_facecolor("darkblue")  
    ax[5].set_facecolor("darkblue")

    fig.colorbar(im1, ticks=MaxNLocator(6))
    fig.colorbar(im2, ticks=MaxNLocator(6))
    fig.colorbar(im3, ticks=MaxNLocator(6))
    fig.tight_layout()
 
    plt.rcParams['figure.figsize'] = [6, 4]
    fig, ax = plt.subplots(1,3)
    im1=axcontourf(ax[0],r_emi,z_emi, t_emi,r'$T_{s}(EMI)$',levels= np.linspace(1500,2100,50))
    im2=axcontourf(ax[1],r,z, t_bemi - t_emi,r'$\Delta_t$ BEMI',levels= np.linspace(-80,80,50),CMAP='bwr')
    abs_err = t_cgan_caseC - t_emi
    im3=axcontourf(ax[2],r,z,abs_err,'$\Delta_t$ U-Net',levels= np.linspace(-80,80,50),CMAP='bwr')
    abs_err[abs_err>100] = 100
    abs_err[abs_err<-100] = -100
    
    ax[0].set_facecolor("darkblue")  
    ax[1].set_facecolor("darkblue")  
    ax[2].set_facecolor("darkblue")  
    fig.colorbar(im1, ticks=MaxNLocator(6))
    fig.colorbar(im2, ticks=MaxNLocator(6))
    fig.colorbar(im3, ticks=MaxNLocator(6))
    fig.tight_layout()

    print('Abs. error max:', abs_err.max())
    print('Abs. error min:', abs_err.min())
    print('Abs. error mean:', abs_err.mean())
    print('Abs. error stddev:', abs_err.std())
    print('Abs. error %:', abs_err.mean()*100/t_emi.mean())

    plt.show()


def axcontourf(ax,r,z, data, title, levels=50, Y_MIN=1,Y_MAX=3.5,CMAP='jet'):
        x = ax.contourf(r, z,data,levels, cmap = CMAP)#,vmin =VMIN, vmax = VMAX)
        ax.set_xlabel('r (cm)')
        ax.set_ylabel('z (cm)')
        ax.set_title(title)
        ax.set_xlim(0,0.45)
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_xticks(ticks=[0, 0.25])#, labels=[0, T_0.shape[1]])
        return x 

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default = 32, type=int,help='batch size to train')
    parser.add_argument('--epochs', default = 100, type=int,help='epoch to train')
    parser.add_argument('--dropout', default = 0.089735, type=float,help='percentage dropout to use')
    parser.add_argument('--num_filters', default = 29, type=int,help='Canales de salida de la primera capa conv')
    parser.add_argument('--learning_rate', default = 0.000410, type=float, help='learning rate')
    parser.add_argument('--weights', default= 'weights/best.pth', type=str, help='path to weights')
    parser.add_argument('--experiment', action='store_true', help='si es experimento ')
    parser.add_argument('--TRT', action='store_true', help='si es experimento ')
    opt = parser.parse_args()
    return opt

def main(opt):

    if(opt.TRT):
        eval_exp_TRT(opt)
    elif(opt.experiment):
        eval_exp(opt)
    else:
        eval(opt)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)