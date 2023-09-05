import torch
import random
import os
import numpy as np
from models.unet import U_Net
import matplotlib.pyplot as plt
from utils.functions import destandarize
from utils.load_data import MyDataLoader
from matplotlib.ticker import MaxNLocator
import argparse

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

    plt.show()

    print('Abs. error max:', abs_err.max())
    print('Abs. error min:', abs_err.min())
    print('Abs. error mean:', abs_err.mean())
    print('Abs. error stddev:', abs_err.std())
    print('Abs. error %:', abs_err.mean()*100/y_test[N].mean())

def eval_exp(opt):
    print("experiment")
    # LOAD DATA
    my_data_loader = MyDataLoader()
    _, _, y_mean, y_std, _, _, _= my_data_loader.load_test_data()
    x_test, t_emi = my_data_loader.load_data_exp()

    x_test = torch.tensor(x_test).float().to(device)


    ####LOAD######
    model = U_Net(n1=opt.num_filters, dropout_rate=opt.dropout)
    model = torch.load(opt.weights)
    model.to(device)
    model.eval()

    # Eval 
    with torch.no_grad():
            output = model(x_test)

    x_test = x_test.cpu()

    y_test_pred = destandarize(output, y_mean, y_std)
    y_test_pred = y_test_pred.squeeze(1)

    y_test_pred = y_test_pred.cpu()

    t_emi = t_emi[::-1]
    mask = t_emi<1
    y_test_pred = np.ma.masked_where(mask, y_test_pred[0])

    plt.rcParams['figure.figsize'] = [12, 4]
    plt.subplot(141)
    plt.imshow(x_test[0,0,:,:], cmap = 'jet', vmax=x_test.max() , vmin=x_test.min())
    plt.title('$U-Net$')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(142)
    plt.imshow(x_test[0,1,:,:], cmap = 'jet', vmax=x_test.max(), vmin=x_test.min())
    plt.title('$U-Net$')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(143)
    plt.imshow(x_test[0,2,:,:], cmap = 'jet', vmax=x_test.max(), vmin=x_test.min())
    plt.title('$U-Net$')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(144)
    plt.imshow(y_test_pred, cmap = 'jet', vmin=1500, vmax=2205)
    plt.title('$U-Net$')
    plt.colorbar()
    plt.tight_layout()

    plt.show()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default = 32, type=int,help='batch size to train')
    parser.add_argument('--epochs', default = 100, type=int,help='epoch to train')
    parser.add_argument('--dropout', default = 0.089735, type=float,help='percentage dropout to use')
    parser.add_argument('--num_filters', default = 29, type=int,help='Canales de salida de la primera capa conv')
    parser.add_argument('--learning_rate', default = 0.000410, type=float, help='learning rate')
    parser.add_argument('--weights', default= 'weights/best.pth', type=str, help='path to weights')
    parser.add_argument('--experiment', action='store_true', help='si es experimento ')
    opt = parser.parse_args()
    return opt

def main(opt):

    if(opt.experiment):
        eval_exp(opt)
    else:
        eval(opt)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)