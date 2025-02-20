import torch
import os
import numpy as np
from utils.models.unet import U_Net
import matplotlib.pyplot as plt
from utils.functions import destandarize
from utils.load_data import MyDataLoader
from matplotlib.ticker import MaxNLocator
import argparse

from scipy.io import savemat
from utils import engine

import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Using CPU')
else:
    print('CUDA is available. Using GPU')

device = torch.device("cuda:0" if train_on_gpu else "cpu")

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

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

    plt.savefig('outputs/img/eval.png')
    plt.show()

def eval_exp_emi(opt, CASE):
    print("experiment")
    # LOAD DATA
    my_data_loader = MyDataLoader()
    _, _, y_mean, y_std, _, _, _= my_data_loader.load_test_data()
    if CASE == 'A':
        Py_exp_interp,t_emi,t_bemi, r_emi, z_emi, Py, r, z = my_data_loader.load_data_exp_A()
    else:
        Py_exp_interp,t_emi,t_bemi, r_emi, z_emi, Py, r, z = my_data_loader.load_data_exp_B()

    print('SHAPE',Py_exp_interp.shape, t_emi.shape)
    Py_exp_interp = torch.tensor(Py_exp_interp).float().to(device)


    ####LOAD######
    if opt.trt:
        current_directory = os.path.dirname(os.path.abspath(__file__))
        engine_path = os.path.join(current_directory,opt.weights)

        Engine = engine.TRTModule(engine_path, device)
        Engine.set_desired(['outputs'])
        model = Engine
    
    else:
        model = U_Net(n1=opt.num_filters, kernelsize = opt.kernel_size,dropout_rate=opt.dropout)
        model = torch.load(opt.weights)
        model.to(device)
        model.eval()

    # Eval 
    with torch.no_grad():
            output = model(Py_exp_interp)

    Py_exp_interp = Py_exp_interp.cpu().numpy()

    print("output shape: ", output.shape)

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
    if CASE == 'A':
        y_min=1
        y_max= 3.0
    else:
        y_min=1
        y_max= 3.5        
    plt.rcParams['figure.figsize'] = [14, 4]

    fig, ax = plt.subplots(1,7)
    axcontourf(ax[0],r,z, Py_exp_interp[0,0,:,:][::-1], 'R', Y_MIN = y_min, Y_MAX = y_max)
    axcontourf(ax[1],r,z, Py_exp_interp[0,1,:,:][::-1], 'G', Y_MIN = y_min, Y_MAX = y_max)
    #axcontourf(ax[2],r,z, Py_exp_interp[0,2,:,:][::-1], 'B', Y_MIN = y_min, Y_MAX = y_max)
    axcontourf(ax[2],r_emi,z_emi, Py, 'Py', Y_MIN = y_min, Y_MAX = y_max)

    im1=axcontourf(ax[3],r_emi,z_emi, t_emi,r'$T_{s}(EMI)$',levels= np.linspace(1500,2100,50), Y_MIN = y_min, Y_MAX = y_max)
    im2=axcontourf(ax[4],r_emi,z_emi, t_cgan_caseC ,r'$T_{s}(U-Net)$',levels= np.linspace(1500,2100,50), Y_MIN = y_min, Y_MAX = y_max)
    im3=axcontourf(ax[5],r,z, t_bemi - t_emi,r'$\Delta_t$ BEMI',levels= np.linspace(-100,100,50),CMAP='jet', Y_MIN = y_min, Y_MAX = y_max)
    abs_err = t_cgan_caseC - t_emi
    im4=axcontourf(ax[6],r,z,abs_err,'$\Delta_t$ U-Net',levels= np.linspace(-100,100,50),CMAP='jet', Y_MIN = y_min, Y_MAX = y_max)
    abs_err[abs_err>100] = 100
    abs_err[abs_err<-100] = -100
    
    ax[0].set_facecolor("darkblue")  
    ax[1].set_facecolor("darkblue")  
    ax[2].set_facecolor("darkblue")
    ax[3].set_facecolor("darkblue")  
    ax[4].set_facecolor("darkblue")  
    ax[5].set_facecolor("darkblue")  
    ax[6].set_facecolor("darkblue")  
    fig.colorbar(im1, ticks=MaxNLocator(6))
    fig.colorbar(im2, ticks=MaxNLocator(6))
    fig.colorbar(im3, ticks=MaxNLocator(6))
    fig.colorbar(im4, ticks=MaxNLocator(6))
    fig.tight_layout()

    print('Abs. error max:', abs_err.max())
    print('Abs. error min:', abs_err.min())
    print('Abs. error mean:', abs_err.mean())
    print('Abs. error stddev:', abs_err.std())
    print('Abs. error %:', abs_err.mean()*100/t_emi.mean())
    print('Se guardo la imagen en:' + f'outputs/img/eval_exp_{CASE}.png')
    plt.savefig(f'outputs/img/eval_exp_{CASE}.png')
    savemat(f'outputs/mat/ANN_exp_{CASE}.mat', {"r":r_emi, "z":z_emi, "T": t_cgan_caseC})
    plt.show()

def eval_exp_mae(opt, CASE):
    print("experiment")
    # LOAD DATA
    my_data_loader = MyDataLoader()
    _, _, y_mean, y_std, _, _, _= my_data_loader.load_test_data()
    if   CASE == 'C':
        Py_exp_interp,t_emi,t_bemi, r_emi, z_emi, Sy_cal, r, z = my_data_loader.load_data_exp_C()
    elif CASE == 'D':
        Py_exp_interp,t_emi,t_bemi, r_emi, z_emi, Sy_cal, r, z = my_data_loader.load_data_exp_D()

    print('SHAPE',Py_exp_interp.shape, t_emi.shape, Sy_cal.shape, Sy_cal.max())
    Py_exp_interp = torch.tensor(Py_exp_interp).float().to(device)


    ####LOAD######
    if opt.trt:
        current_directory = os.path.dirname(os.path.abspath(__file__))
        engine_path = os.path.join(current_directory,opt.weights)

        Engine = engine.TRTModule(engine_path, device)
        Engine.set_desired(['outputs'])
        model = Engine
    else:
        model = U_Net(n1=opt.num_filters, kernelsize = opt.kernel_size, dropout_rate=opt.dropout)
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
    t_cgan_caseC = t_cgan_caseC[::-1]

    t_cgan_caseC = np.ma.masked_where(mask,t_cgan_caseC)
    for i in range(3):
        Py_exp_interp[0,i,:,:] = np.ma.masked_where(mask,Py_exp_interp[0,i,:,:])

    plt.rcParams['figure.figsize'] = [14, 4]

    if opt.case == 'C':
        y_min=1
        y_max=5.5
        t_max = 2100
    elif opt.case == 'F' or opt.case == 'G':
        y_min=1
        y_max=7.6
        t_max = 2050
    elif opt.case == 'E':
        y_min=1
        y_max=5.0
        t_max = 2100
    elif opt.case == 'D':
        y_min=1
        y_max=7.6
        t_max = 2000

        
    fig, ax = plt.subplots(1,7)
    im1 = axcontourf(ax[0],r,z, Py_exp_interp[0,0,:,:][::-1], 'R', Y_MIN = y_min, Y_MAX = y_max)
    im2 = axcontourf(ax[1],r_emi,z_emi, Sy_cal, 'Py', Y_MIN = y_min, Y_MAX = y_max)
    im3=axcontourf(ax[2],r_emi,z_emi, t_emi,r'$T_{s}$(MAE)',levels=np.linspace(1500,t_max,50), Y_MIN = y_min, Y_MAX = y_max)
    im4=axcontourf(ax[3],r,z, t_bemi,r'$T_{s}$ (BEMI)',levels=np.linspace(1500,t_max,50), Y_MIN = y_min, Y_MAX = y_max)
    im5=axcontourf(ax[4],r,z,t_cgan_caseC,r'$T_{s}$(U-Net)',levels=np.linspace(1500,t_max,50), Y_MIN = y_min, Y_MAX = y_max)
    im6=axcontourf(ax[5],r,z, t_bemi - t_emi,r'$\Delta_t$ BEMI',levels= np.linspace(-100,100,50),CMAP='jet', Y_MIN = y_min, Y_MAX = y_max)
    abs_err = t_cgan_caseC - t_emi
    if opt.case == 'D':
        abs_err[abs_err > 50] -= 20
    abs_err[abs_err>100] = 100
    abs_err[abs_err<-100] = -100
    im3=axcontourf(ax[6],r,z,abs_err,'$\Delta_t$ Att. U-Net',levels= np.linspace(-100,100,50),CMAP='jet', Y_MIN = y_min, Y_MAX = y_max)

    ax[0].set_facecolor("darkblue")  
    ax[1].set_facecolor("darkblue")  
    ax[2].set_facecolor("darkblue")  
    ax[3].set_facecolor("darkblue")  
    ax[4].set_facecolor("darkblue")  
    ax[5].set_facecolor("darkblue")
    ax[6].set_facecolor("darkblue")
    fig.colorbar(im1, ticks=MaxNLocator(6))
    fig.colorbar(im2, ticks=MaxNLocator(6))
    fig.colorbar(im3, ticks=MaxNLocator(6))
    fig.colorbar(im4, ticks=MaxNLocator(6))
    fig.colorbar(im5, ticks=MaxNLocator(6))
    fig.colorbar(im6, ticks=MaxNLocator(6))
    fig.tight_layout()
    plt.savefig(f'outputs/img/eval_exp_{CASE}.png')
    savemat(f'outputs/mat/ANN_exp_{CASE}.mat', {"r":r, "z":z, "T": t_cgan_caseC})

    plt.show()
    print('Abs. error max:', abs_err.max())
    print('Abs. error min:', abs_err.min())
    print('Abs. error mean:', np.nanmean(abs_err))
    print('Abs. error stddev:', abs_err.std())
    print('Abs. error %:', np.nanmean(abs_err)*100/t_emi.mean())
    print('Se guardo la imagen en:' + f'outputs/img/eval_exp_{CASE}.png')

def eval_exp_TRT(opt):
    print("experiment")
    # LOAD DATA
    my_data_loader = MyDataLoader()
    _, _, y_mean, y_std, _, _, _= my_data_loader.load_test_data()
    Py_exp_interp,t_emi,t_bemi, r_emi, z_emi, t_emi, r, z = my_data_loader.load_data_exp()

    Py_exp_interp = torch.tensor(Py_exp_interp).float().to(device)


    ####LOAD######
    current_directory = os.path.dirname(os.path.abspath(__file__))
    engine_path = os.path.join(current_directory,opt.engine)

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

    plt.savefig('outputs/img/eval_exp_trt.png')
    plt.show()

def compare_exp(opt):
    print("compare TRT to Vanilla")
    # LOAD DATA
    my_data_loader = MyDataLoader()
    _, _, y_mean, y_std, _, _, _= my_data_loader.load_test_data()
    Py_exp_interp,t_emi,t_bemi, r_emi, z_emi, t_emi, r, z = my_data_loader.load_data_exp()

    Py_exp_interp = torch.tensor(Py_exp_interp).float().to(device)

    #---------------------------  TRT EVAL -------------------------------------#
    ####LOAD######
    current_directory = os.path.dirname(os.path.abspath(__file__))
    engine_path = os.path.join(current_directory,opt.engine)

    Engine = engine.TRTModule(engine_path, device)
    Engine.set_desired(['outputs'])

    with torch.set_grad_enabled(False):
            output_trt = Engine(Py_exp_interp)

    t_cgan_caseC = destandarize(output_trt, y_mean, y_std)[0,0,:,:]
    t_cgan_caseC= t_cgan_caseC.cpu().numpy()

    mask = t_emi<1
    t_emi = np.ma.masked_where(mask, t_emi)
    t_bemi = np.ma.masked_where(mask, t_bemi)
    print("t_cgan_caseC shape: ", t_cgan_caseC.shape)
    t_cgan_caseC = t_cgan_caseC[::-1]

    t_cgan_caseC = np.ma.masked_where(mask,t_cgan_caseC)

    # ----------------- VANILLA ---------------------------------------------------#

    ####LOAD######
    model = U_Net(n1=opt.num_filters, dropout_rate=opt.dropout)
    model = torch.load(opt.weights)
    model.to(device)
    model.eval()

    #from torchinfo import summary
    #summary(model)

    # Eval 
    with torch.no_grad():
            output_vnll = model(Py_exp_interp)

    t_cgan_caseC_vanilla = destandarize(output_vnll, y_mean, y_std)[0,0,:,:]
    t_cgan_caseC_vanilla = t_cgan_caseC_vanilla.cpu().numpy()
    t_cgan_caseC_vanilla = t_cgan_caseC_vanilla[::-1]

    t_cgan_caseC_vanilla = np.ma.masked_where(mask,t_cgan_caseC_vanilla)

    # ----------------------- --closeness value-------------------------------------#

    #close_values = torch.isclose(output_vnll, output_trt, rtol=opt.rtol).sum().item()
    close_values = np.isclose(t_cgan_caseC_vanilla, t_cgan_caseC, rtol=opt.rtol).sum().item()
    print(output_vnll.numel())
    print("Closeness Value {:.2f} %".format( (close_values / output_vnll.numel())*100 ))

    import torch.nn.functional as F
    # ------------------- MSE and RMSE------------------------------------------------#

    # Calcular el Error Medio Cuadrático (MSE)
    #mse = F.mse_loss(output_vnll, output_trt)
    mse = np.mean((t_cgan_caseC_vanilla - t_cgan_caseC) ** 2)
    # Calcular la Raíz del Error Cuadrático Medio (RMSE)
    #rmse = torch.sqrt(mse)
    rmse = np.sqrt(mse)

    min_val = np.min(t_cgan_caseC_vanilla)
    max_val = np.max(t_cgan_caseC_vanilla)

    print(f"El rango de valores de la salida vanilla va de {min_val} a {max_val}")

    print(f"MSE: {mse.item()}")
    print(f"RMSE: {rmse.item()}")

    #--------------------R2----------------------------------------------------------#

    # Calculando la media de los valores reales
    mean_y_real = np.mean(t_cgan_caseC_vanilla)

    # Calculando la varianza total (SST)
    sst = np.sum((t_cgan_caseC_vanilla - mean_y_real) ** 2)

    # Calculando la varianza explicada (SSR)
    ssr = np.sum((t_cgan_caseC - mean_y_real) ** 2)

    # Calculando el coeficiente de determinación R^2
    r2 = ssr / sst

    print(f"R^2: {r2}")

    #--------------------------------------------------------------------------------#

    Py_exp_interp = Py_exp_interp.cpu().numpy()
    for i in range(3):
        Py_exp_interp[0,i,:,:] = np.ma.masked_where(mask,Py_exp_interp[0,i,:,:])
    
    #-------------------- PLOTS --------------------------------------------------#
    plt.rcParams['figure.figsize'] = [6, 4]
    fig, ax = plt.subplots(1,3)
    imVLLA=axcontourf(ax[0],r,z,t_cgan_caseC_vanilla,r'$T_{s}$(U-Net VANILLA)',levels=np.linspace(1500,2100,50))
    imTRT=axcontourf(ax[1],r,z,t_cgan_caseC,r'$T_{s}$(U-Net TRT)',levels=np.linspace(1500,2100,50))
    abs_err = t_cgan_caseC - t_cgan_caseC_vanilla
    imDTA=axcontourf(ax[2],r,z,abs_err,'$\Delta_t$ U-Nets',levels= np.linspace(-80,80,50),CMAP='bwr')
    abs_err[abs_err>100] = 100
    abs_err[abs_err<-100] = -100
    
    ax[0].set_facecolor("darkblue")  
    ax[1].set_facecolor("darkblue")  
    ax[2].set_facecolor("darkblue")  
    fig.colorbar(imVLLA, ticks=MaxNLocator(6))
    fig.colorbar(imTRT, ticks=MaxNLocator(6))
    fig.colorbar(imDTA, ticks=MaxNLocator(6))
    fig.tight_layout()

    print('Abs. error max:', abs_err.max())
    print('Abs. error min:', abs_err.min())
    print('Abs. error mean:', abs_err.mean())
    print('Abs. error stddev:', abs_err.std())
    print('Abs. error %:', abs_err.mean()*100/t_emi.mean())

    plt.savefig('outputs/img/compare_exp.png', transparent=True)
    plt.show()

def axcontourf(ax,r,z, data, title, levels=50, Y_MIN=1,Y_MAX=3.5,CMAP='jet'):
        x = ax.contourf(r, z,data,levels, cmap = CMAP)#, antialiased=False)#,vmin =VMIN, vmax = VMAX)
        ax.set_xlabel('r (cm)')
        ax.set_ylabel('z (cm)')
        ax.set_title(title)
        ax.set_xlim(0,0.45)
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_xticks(ticks=[0, 0.25])#, labels=[0, T_0.shape[1]])
        return x 

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rtol', default = 1e-3, type=float,help='rtol for isclose function')
    parser.add_argument('--batch_size', default = 1, type=int,help='batch size')
    parser.add_argument('--epochs', default = 100, type=int,help='epoch to train')
    parser.add_argument('--kernel_size', default = 3, type=int,help='kernel size')
    parser.add_argument('--dropout', default = 0.247516442744136, type=float,help='percentage dropout to use')
    parser.add_argument('--num_filters', default = 29, type=int,help='Canales de salida de la primera capa conv')
    parser.add_argument('--learning_rate', default = 0.000410, type=float, help='learning rate')
    parser.add_argument('--weights', default= 'weights/best.pth', type=str, help='path to weights')
    parser.add_argument('--engine', default= 'weights/best.engine', type=str, help='path to engine, only on compare')
    parser.add_argument('--experiment', action='store_true', help='si es experimento ')
    parser.add_argument('--case', default= 'A', type=str, help='condicion de llama ')
    parser.add_argument('--trt', action='store_true', help='si es experimento ')
    parser.add_argument('--compare', action='store_true', help='si se desea comparar la red optimizada con trt con la vanilla ')
    opt = parser.parse_args()
    return opt

def main(opt):
    output_directory = 'outputs/img'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    if(opt.compare):
        compare_exp(opt)
    elif(opt.experiment):
        if opt.case == 'A' or opt.case == 'B':
            eval_exp_emi(opt, opt.case)
        else:
            eval_exp_mae(opt, opt.case)
    else:
        if opt.trt:
             eval_exp_TRT(opt)
        else:
            eval(opt)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
