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
import time
from matplotlib.gridspec import GridSpec

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

# Aumentar tamaño de fuente globalmente
plt.rcParams.update({
    'font.size': 14,  # Tamaño de fuente global
    'axes.titlesize': 16,  # Tamaño de los títulos de los ejes
    'axes.labelsize': 16,  # Tamaño de las etiquetas de los ejes
    'xtick.labelsize': 14,  # Tamaño de los valores en el eje X
    'ytick.labelsize': 14,  # Tamaño de los valores en el eje Y
    'legend.fontsize': 14,  # Tamaño de la fuente en la leyenda
    'figure.titlesize': 14  # Tamaño del título de la figura
})

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

    if opt.case == 'A':
        y_min=1
        y_max= 3.0
    elif opt.case == 'B':
        y_min=1
        y_max= 3.5  
    elif opt.case == 'C':
        y_min=1
        y_max=5.5
        t_max = 2100

        
    plt.rcParams['figure.figsize'] = [6, 4]
    fig, ax = plt.subplots(1,3)
    imVLLA=axcontourf(ax[0],r,z,t_cgan_caseC_2,r'$T_{s}$(U-Net model 2)',levels=np.linspace(1500,2100,50))
    imTRT=axcontourf(ax[1],r,z,t_cgan_caseC_1,r'$T_{s}$(U-Net model 1)',levels=np.linspace(1500,2100,50))
    abs_err = t_cgan_caseC_1 - t_cgan_caseC_2
    imDTA=axcontourf(ax[2],r,z,abs_err,'$\Delta_t$ U-Nets',levels= np.linspace(-30,30,50),CMAP='bwr')
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

def axcontourf(ax, r, z, data, title, levels=50, Y_MIN=1, Y_MAX=3.5, CMAP='jet',show_axes=True,ftitle=None):
    data = np.clip(data, levels[0], levels[-1])  # Recorta los valores fuera del rango
    x = ax.contourf(r, z, data, levels, cmap=CMAP)
    ax.set_title(title)
    ax.set_xlim(0, 0.45)
    ax.set_ylim(Y_MIN, Y_MAX)
    if show_axes:
        ax.set_xticks(ticks=[0, 0.25])
        ax.set_xlabel('r (cm)')
        ax.set_ylabel('z (cm)')
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel('')
        ax.set_ylabel('')
    if isinstance(ftitle, str) and ftitle:
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()
        ax.set_ylabel(ftitle, fontsize=16, rotation=-90, labelpad=15,fontweight='bold')
    return x

def get_model_size_MB(model_path):
    return os.path.getsize(model_path) / (1024 * 1024) 

def get_parameters_vanilla(model):
    total_capas = sum(1 for _ in model.modules())
    total_parametros = sum(p.numel() for p in model.parameters())
    return total_capas, total_parametros

def get_layers(model_name, model_path):
    # para que funcione como sudo es necesario correr desde el path del enviroment env/bin/polygraphy
    if model_name == 'tensorrt':
        cmd = f"env/bin/polygraphy inspect model {model_path}"
    else:
        cmd = f"env/bin/polygraphy inspect model {(model_path).replace('.engine', '.onnx')} --display-as=trt"

    # Ejecuta el comando y captura la salida
    import subprocess
    import re
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Decodifica la salida a texto
    output = stdout.decode()

    # Usa una expresión regular para encontrar el número de capas
    match = re.search(r"---- (\d+) Layer\(s\) ----", output)
    # Extrae el número de capas si se encuentra el patrón
    if match:
        num_layers = int(match.group(1))
        return num_layers
    else:
        print("No se encontró el número de capas")
        return 0

def get_parametros(model_type, model_path):
    if model_type == 'tensorrt':
        cmd = f"env/bin/python utils/param_counter.py --engine ../{model_path}"
    else:
        cmd = f"env/bin/onnx_opcounter {(model_path).replace('.engine', '.onnx')}"

    # Ejecuta el comando y captura la salida
    import subprocess
    import re
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Decodifica la salida a texto
    output = stdout.decode()

    # Usa una expresión regular para encontrar el número de capas
    match = re.search(r"Number of parameters in the model: (\d+)", output)
    if match:
        num_parameters = int(match.group(1))
        return num_parameters
    else:
        print("No se encontró el número de parametros")
        return 0

def load_model(opt,model_name, weight):
    #print('----------------------------------------------------------\n')
    #print('Modelo ', model_name, 'path = ', weight,'\n')

    if model_name == 'tensorrt':
        current_directory = os.path.dirname(os.path.abspath(__file__))
        engine_path = os.path.join(current_directory, weight)
        model = engine.TRTModule(engine_path, device)
        model.set_desired(['outputs'])    
        
        #print("# capas onnx = ",get_layers(model_name,weight))
        #print("# parametro onnx = ",get_parametros(model_name,weight))
    elif model_name == 'unet' or model_name == 'attunet':
        model = torch.load(weight)
        model.to(device)
        model.eval()
        #print("# capas base = ", get_parameters_vanilla(model)[0])
        #print("# parametros base = ", get_parameters_vanilla(model)[1])
    else:
        print(f'ERROR: especifica un modelo válido, opciones: tensorrt, unet, attunet. Modelo dado: {opt.model}')
        return None
    #print("tamaño = ",get_model_size_MB(weight), " MB")
    return model

def compare_extended(opt, model_unet, model_attention_unet, 
                     model_unet_trt_fp32, model_unet_trt_fp16, model_unet_trt_int8, 
                     model_attention_unet_trt_fp32,
                     model_attention_unet_trt_fp16,
                     model_attention_unet_trt_int8):
    print("Comparando modelos base y optimizados")

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

    def model_inference(model, data, name=""):
        with torch.no_grad():
            start_time = time.time()
            output = model(data)
            end_time = time.time()
            print(f"Tiempo de ejecución {name}: {end_time - start_time:.4f} s")
        output = destandarize(output, y_mean, y_std)[0, 0, :, :].cpu().numpy()[::-1]
        return np.ma.masked_where(mask, output)

    # Inferencias UNet
    unet_output = model_inference(model_unet, Py_exp_interp, "UNet Base")
    unet_trt_fp32_output = model_inference(model_unet_trt_fp32, Py_exp_interp, "UNet TRT fp32")
    unet_trt_fp16_output = model_inference(model_unet_trt_fp16, Py_exp_interp, "UNet TRT fp16")
    unet_trt_int8_output = model_inference(model_unet_trt_int8, Py_exp_interp, "UNet TRT int8")

    # Inferencias Attention UNet
    attention_unet_output = model_inference(model_attention_unet, Py_exp_interp, "Attention UNet Base")
    attention_unet_trt_fp32_output = model_inference(model_attention_unet_trt_fp32, Py_exp_interp, "Attention UNet TRT FP32")
    attention_unet_trt_fp16_output = model_inference(model_attention_unet_trt_fp16, Py_exp_interp, "Attention UNet TRT fp16")
    attention_unet_trt_int8_output = model_inference(model_attention_unet_trt_int8, Py_exp_interp, "Attention UNet TRT int8")

    # Función para calcular error absoluto
    def abs_error(base, optimized):
        error = base - optimized
        error = np.clip(error, -100, 100)
        return error

    ##  Configurar figura
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 8, width_ratios=[0.1,0.3,1, 1, 1, 1,0.2,0.1],wspace=0.1,hspace=0.3)
    axes = [[fig.add_subplot(gs[i, j]) for j in range(8)] for i in range(2)]

    for row in axes:
        for ax in row:
            ax.set_facecolor("darkgray")

    for i in [0, 1]:
        for j in [0,1,6,7]:
            axes[i][j].axis("off")  
            axes[i][j].set_frame_on(False)

    if opt.case == 'A':
        y_min, y_max, t_max = 1, 3.0, 2200
    elif opt.case == 'B':
        y_min, y_max, t_max = 1, 3.5, 2200
    elif opt.case == 'C':
        y_min, y_max, t_max = 1, 5.5, 2100

    # Plot para UNet
    unet = axcontourf(axes[0][2], r, z, unet_output, 'Modelo Base', levels=np.linspace(1500, t_max, 50),Y_MAX=y_max, Y_MIN=y_min)
    unet_fp32 = axcontourf(axes[0][3], r, z, abs_error(unet_output, unet_trt_fp32_output), '$\Delta_t$ TRT fp32', levels=np.linspace(-30, 30, 50), CMAP='bwr',Y_MAX=y_max, Y_MIN=y_min,show_axes=False)
    unet_fp16 = axcontourf(axes[0][4], r, z, abs_error(unet_output, unet_trt_fp16_output), '$\Delta_t$ TRT fp16', levels=np.linspace(-30, 30, 50), CMAP='bwr',Y_MAX=y_max, Y_MIN=y_min,show_axes=False)
    unet_int8 = axcontourf(axes[0][5], r, z, abs_error(unet_output, unet_trt_int8_output), '$\Delta_t$ TRT int8', levels=np.linspace(-30, 30, 50), CMAP='bwr',Y_MAX=y_max, Y_MIN=y_min,show_axes=False,ftitle="U-Net")

    # Plot para Attention UNet
    attunet = axcontourf(axes[1][2], r, z, attention_unet_output, 'Modelo Base', levels=np.linspace(1500, t_max, 50),Y_MAX=y_max, Y_MIN=y_min)
    attunet_fp32 = axcontourf(axes[1][3], r, z, abs_error(attention_unet_output, attention_unet_trt_fp32_output), '$\Delta_t$ TRT fp32', levels=np.linspace(-30, 30, 50), CMAP='bwr',Y_MAX=y_max, Y_MIN=y_min,show_axes=False)
    attunet_fp16 = axcontourf(axes[1][4], r, z, abs_error(attention_unet_output, attention_unet_trt_fp16_output), '$\Delta_t$ TRT fp16', levels=np.linspace(-30, 30, 50), CMAP='bwr',Y_MAX=y_max, Y_MIN=y_min,show_axes=False)
    attunet_int8 = axcontourf(axes[1][5], r, z, abs_error(attention_unet_output, attention_unet_trt_int8_output), '$\Delta_t$ TRT int8', levels=np.linspace(-30, 30, 50), CMAP='bwr',Y_MAX=y_max, Y_MIN=y_min,show_axes=False,ftitle="Att. U-Net")

    # Colorbars
    cbar_ax_left = fig.add_subplot(gs[:, 0])  # Colorbar izquierdo en ambas filas
    cbar_ax_right = fig.add_subplot(gs[:, 7])  # Colorbar derecho en ambas filas
    
    cbar_unet = fig.colorbar(unet, cax=cbar_ax_left, ticks=MaxNLocator(6))
    cbar_unet.ax.yaxis.set_ticks_position('left') 
    cbar_unet.ax.set_title(r'$T$ [K]', fontsize=16)
    cbar_trt = fig.colorbar(unet_int8, cax=cbar_ax_right, ticks=MaxNLocator(6))
    cbar_trt.ax.set_title(r'$\Delta T$ [K]', fontsize=16)

    fig.tight_layout()
    #plt.savefig('outputs/img/compare_extended.png')
    plt.savefig(f'outputs/img/compare_extended_case_{opt.case}.pdf', format='pdf', bbox_inches='tight')
    plt.show()

def compare_all(opt):
    model_unet = load_model(opt,'unet', 'weights/unet.pth')
    model_attention_unet = load_model(opt,'attunet', 'weights/attunet.pth')
    unet_fp32 = load_model(opt,'tensorrt', 'weights/unet_fp32.engine')
    unet_fp16 = load_model(opt,'tensorrt', 'weights/unet_fp16.engine')
    unet_int8 = load_model(opt,'tensorrt', 'weights/unet_int8.engine')

    attunet_fp32 = load_model(opt,'tensorrt', 'weights/attunet_fp32.engine')
    attunet_fp16 = load_model(opt,'tensorrt', 'weights/attunet_fp16.engine')
    attunet_int8 = load_model(opt,'tensorrt', 'weights/attunet_int8.engine')

    compare_extended(opt, model_unet, model_attention_unet, 
                     unet_fp32, unet_fp16, unet_int8, 
                     attunet_fp32,attunet_fp16,attunet_int8)

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
    parser.add_argument('--compare_all', action='store_true', help='Compara todos los modelos para un solo caso en especifico')
    opt = parser.parse_args()
    return opt

def main(opt):
    output_directory = 'outputs/img'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    if opt.compare_all:
        compare_all(opt)
    elif opt.compare:
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
