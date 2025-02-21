import torch
import os
import numpy as np
from utils.models.unet import U_Net
from utils.models.attunet import AttentionUNet
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

def eval(opt, model):
    # LOAD DATA
    my_data_loader = MyDataLoader()
    x_test, y_test, y_mean, y_std, fs_test, r_test, z_test= my_data_loader.load_test_data()

    x_test = torch.tensor(x_test).float().to(device)
    y_test = torch.tensor(y_test).float().to(device)

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

def eval_exp_emi(opt, model):
    print("experiment emi")
    # LOAD DATA
    my_data_loader = MyDataLoader()
    _, _, y_mean, y_std, _, _, _= my_data_loader.load_test_data()
    if opt.case == 'A':
        Py_exp_interp,t_emi,t_bemi, r_emi, z_emi, Py, r, z = my_data_loader.load_data_exp_A()
    else:
        Py_exp_interp,t_emi,t_bemi, r_emi, z_emi, Py, r, z = my_data_loader.load_data_exp_B()

    print('SHAPE',Py_exp_interp.shape, t_emi.shape)
    Py_exp_interp = torch.tensor(Py_exp_interp).float().to(device)

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
    if opt.case == 'A':
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
    print('Se guardo la imagen en:' + f'outputs/img/eval_exp_{opt.model}_{opt.case}.png')
    plt.savefig(f'outputs/img/eval_exp_{opt.model}_{opt.case}.png')
    savemat(f'outputs/mat/ANN_exp_{opt.model}_{opt.case}.mat', {"r":r_emi, "z":z_emi, "T": t_cgan_caseC})
    plt.show()

def eval_exp_mae(opt, model):

    print("experiment mae")
    # LOAD DATA
    my_data_loader = MyDataLoader()
    _, _, y_mean, y_std, _, _, _= my_data_loader.load_test_data()
    if   opt.case == 'C':
        Py_exp_interp,t_emi,t_bemi, r_emi, z_emi, Sy_cal, r, z = my_data_loader.load_data_exp_C()
    #elif opt.case == 'D':
    #    Py_exp_interp,t_emi,t_bemi, r_emi, z_emi, Sy_cal, r, z = my_data_loader.load_data_exp_D()

    print('SHAPE',Py_exp_interp.shape, t_emi.shape, Sy_cal.shape, Sy_cal.max())
    Py_exp_interp = torch.tensor(Py_exp_interp).float().to(device)

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
    plt.savefig(f'outputs/img/eval_exp_{opt.model}_{opt.case}.png')
    savemat(f'outputs/mat/ANN_exp_{opt.model}_{opt.case}.mat', {"r":r, "z":z, "T": t_cgan_caseC})

    plt.show()
    print('Abs. error max:', abs_err.max())
    print('Abs. error min:', abs_err.min())
    print('Abs. error mean:', np.nanmean(abs_err))
    print('Abs. error stddev:', abs_err.std())
    print('Abs. error %:', np.nanmean(abs_err)*100/t_emi.mean())
    print('Se guardo la imagen en:' + f'outputs/img/eval_exp_{opt.model}_{opt.case}.png')

def compare(opt,model1,model2):
    print("compare two models")

    # LOAD DATA
    my_data_loader = MyDataLoader()
    _, _, y_mean, y_std, _, _, _ = my_data_loader.load_test_data()
    if opt.case == 'A':
        Py_exp_interp,t_emi,t_bemi, r_emi, z_emi, Py, r, z = my_data_loader.load_data_exp_A()
    elif opt.case == 'B':
        Py_exp_interp,t_emi,t_bemi, r_emi, z_emi, Py, r, z = my_data_loader.load_data_exp_B()
    elif opt.case == 'C':
        Py_exp_interp,t_emi,t_bemi, r_emi, z_emi, Sy_cal, r, z = my_data_loader.load_data_exp_C()
    #elif opt.case == 'D':
    #    Py_exp_interp,t_emi,t_bemi, r_emi, z_emi, Sy_cal, r, z = my_data_loader.load_data_exp_D()     
    else: # data test
        Py_exp_interp,t_emi,t_bemi, r_emi, z_emi, t_emi, r, z = my_data_loader.load_data_exp()

    Py_exp_interp = torch.tensor(Py_exp_interp).float().to(device)

    mask = t_emi<1
    t_emi = np.ma.masked_where(mask, t_emi)
    t_bemi = np.ma.masked_where(mask, t_bemi)

    #---------------------------  model 1 -------------------------------------#
    import time
    with torch.no_grad():
        start_time = time.time()
        output_1 = model1(Py_exp_interp)
        end_time = time.time()
    print(f"Tiempo de ejecución model1: {end_time - start_time} segundos")
    #print("output 1 shape: ", output_1.shape)

    t_cgan_caseC_1 = destandarize(output_1, y_mean, y_std)[0,0,:,:]
    t_cgan_caseC_1 = t_cgan_caseC_1.cpu().numpy()
    #print("t_cgan_caseC 1 shape: ", t_cgan_caseC_1.shape)
    t_cgan_caseC_1 = t_cgan_caseC_1[::-1]
    t_cgan_caseC_1 = np.ma.masked_where(mask,t_cgan_caseC_1)

    # ----------------- model 2 ---------------------------------------------------#

    with torch.no_grad():
        start_time = time.time() 
        output_2 = model2(Py_exp_interp)
        end_time = time.time() 

    print(f"Tiempo de ejecución model2: {end_time - start_time} segundos")
    #print("output 2 shape: ", output_2.shape)

    t_cgan_caseC_2 = destandarize(output_2, y_mean, y_std)[0,0,:,:]
    t_cgan_caseC_2 = t_cgan_caseC_2.cpu().numpy()
    #print("t_cgan_caseC 2 shape: ", t_cgan_caseC_2.shape)
    t_cgan_caseC_2 = t_cgan_caseC_2[::-1]
    t_cgan_caseC_2 = np.ma.masked_where(mask,t_cgan_caseC_2)

    # ----------------------- --closeness value-------------------------------------#

    #close_values = torch.isclose(output_vnll, output_trt, rtol=opt.rtol).sum().item()
    close_values = np.isclose(t_cgan_caseC_2, t_cgan_caseC_1, rtol=opt.rtol).sum().item()
    print(output_2.numel())
    print("Closeness Value {:.2f} %".format( (close_values / output_2.numel())*100 ))

    import torch.nn.functional as F
    # ------------------- MSE and RMSE------------------------------------------------#

    # Calcular el Error Medio Cuadrático (MSE)
    #mse = F.mse_loss(output_vnll, output_trt)
    mse = np.mean((t_cgan_caseC_2 - t_cgan_caseC_1) ** 2)
    # Calcular la Raíz del Error Cuadrático Medio (RMSE)
    #rmse = torch.sqrt(mse)
    rmse = np.sqrt(mse)

    min_val = np.min(t_cgan_caseC_2)
    max_val = np.max(t_cgan_caseC_2)

    print(f"El rango de valores de la salida vanilla va de {min_val} a {max_val}")

    print(f"MSE: {mse.item()}")
    print(f"RMSE: {rmse.item()}")

    #--------------------R2----------------------------------------------------------#

    # Calculando la media de los valores reales
    mean_y_real = np.mean(t_cgan_caseC_2)

    # Calculando la varianza total (SST)
    sst = np.sum((t_cgan_caseC_2 - mean_y_real) ** 2)

    # Calculando la varianza explicada (SSR)
    ssr = np.sum((t_cgan_caseC_1 - mean_y_real) ** 2)

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
    imVLLA=axcontourf(ax[0],r,z,t_cgan_caseC_2,r'$T_{s}$(U-Net model 2)',levels=np.linspace(1500,2100,50))
    imTRT=axcontourf(ax[1],r,z,t_cgan_caseC_1,r'$T_{s}$(U-Net model 1)',levels=np.linspace(1500,2100,50))
    abs_err = t_cgan_caseC_1 - t_cgan_caseC_2
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

    plt.savefig('outputs/img/compare.png')#, transparent=True)
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

def load_model(opt,model_name, weight):

    if model_name == 'tensorrt':
        current_directory = os.path.dirname(os.path.abspath(__file__))
        engine_path = os.path.join(current_directory, weight)
        model = engine.TRTModule(engine_path, device)
        model.set_desired(['outputs'])
    elif model_name == 'unet':
        model = U_Net(n1=opt.num_filters, kernelsize=opt.kernel_size, dropout_rate=opt.dropout)
    elif model_name == 'attunet':
        model = AttentionUNet(first_filters=opt.num_filters, kernelsize=opt.kernel_size, batchnorm=True, dropout_rate=opt.dropout)
    else:
        print(f'ERROR: especifica un modelo válido, opciones: tensorrt, unet, attunet. Modelo dado: {opt.model}')
        return None
    
    if model_name == 'unet' or model_name == 'attunet':
        model = torch.load(weight)
        model.to(device)
        model.eval()

    return model

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rtol', default = 1e-3, type=float,help='rtol for isclose function')
    parser.add_argument('--batch_size', default = 1, type=int,help='batch size')
    parser.add_argument('--epochs', default = 100, type=int,help='epoch to train')
    parser.add_argument('--kernel_size', default = 4, type=int,help='kernel size')
    parser.add_argument('--dropout', default = 0.119372, type=float,help='percentage dropout to use')
    parser.add_argument('--num_filters', default = 20, type=int,help='Canales de salida de la primera capa conv')
    parser.add_argument('--learning_rate', default = 0.001112, type=float, help='learning rate')
    parser.add_argument('--model', default= 'unet', type=str, help='modelo a evaluar, puede ser tensorrt, unet o attunet, if compare, use a string separate with a space with the two models to compare.')
    parser.add_argument('--weights', default= 'weights/best.pth', type=str, help='path to weights, if compare, use a string separate with a space with the two paths of weights to compare')
    parser.add_argument('--experiment', action='store_true', help='si es experimento ')
    parser.add_argument('--case', default= 'A', type=str, help='condicion de llama, puede ser A, B o C')
    parser.add_argument('--compare', action='store_true', help='si se desea comparar la red optimizada con trt con la vanilla ')
    opt = parser.parse_args()
    return opt

def main(opt):
    output_directory = 'outputs/img'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if opt.compare:
        models = opt.model.split()  # Divide el string para obtener los dos modelos
        weights = opt.weights.split()
        if len(models) != 2 or len(weights) != 2:
            print("ERROR: Debes especificar exactamente dos (modelos y pesos) separados por un espacio.")
        else:
            model1 = load_model(opt, models[0], weights[0])
            model2 = load_model(opt, models[1], weights[1])
            if model1 is None or model2 is None:
                print("Error en la carga de modelos. Revisa las especificaciones.")

            compare(opt, model1, model2)
    else:
        if len(opt.model.split()) != 1 or len( opt.weights.split()) != 1:
            print("ERROR: Debes especificar exactamente uno (modelos y pesos) si no va a comparar.")
        else:
            print(opt.model)
            model = load_model(opt, opt.model, opt.weights)
            if model is None:
                print("Error en la carga del modelo. Revisa las especificaciones.")
            if(opt.experiment):
                if opt.case == 'A' or opt.case == 'B':
                    eval_exp_emi(opt, model)
                else:
                    eval_exp_mae(opt, model)
            else:
                eval(opt, model) ##  Revisar funcionalidad de eval, parece mal

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)