import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.functions import destandarize
from utils.load_data import MyDataLoader
from matplotlib.ticker import MaxNLocator
import argparse
import time
from matplotlib.gridspec import GridSpec

from utils.processing import process_llamas

from PIL import Image

import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

from utils.utils import *

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

def eval_exp(opt, model):
    model.eval()
    # Data experimental de la condicion de llama (ya sea EMI o MAE)
    data = preprocess_data(opt) 
    # Definir valores de Y_MIN, Y_MAX y t_max según el tipo de experimento y el caso
    exp_name = "EMI"
    if opt.case == 'A':
        y_min, y_max, t_max = 1, 3.0, 2200
    elif opt.case == 'B':
        y_min, y_max, t_max = 1, 3.5, 2200
    elif opt.case == 'C':
        y_min, y_max, t_max = 1, 5.5, 2100
        exp_name = "MAE"

    if opt.dataset_experimental:
        rmse_values = []
        imagenes = sorted(os.listdir(opt.dataset))
        for img_name in imagenes:
            img_path = os.path.join(opt.dataset, img_name)
            img_data = process_experimental_input(opt, img_path)
            img_data = torch.tensor([img_data]).float().to(device)
            if torch.isnan(img_data[0]).any() or torch.isinf(img_data[0]).any():
                #print("\n-------------------------\ncontinue\n--------------------------------\n")
                continue

            with torch.no_grad():
                output = model(img_data)
            t_cgan_caseC = destandarize(output, data["y_mean"], data["y_std"])[0,0,:,:].cpu().numpy()
            t_cgan_caseC = np.ma.masked_where(data["mask"], t_cgan_caseC[::-1])

            #print("Valores de salida (mín, máx):", t_cgan_caseC.min().item(), t_cgan_caseC.max().item())
            rmse = root_mean_squared_error(data["t_emi"], t_cgan_caseC)
            rmse_values.append(rmse)

        # Cálculo de estadísticas
        rmse_values = np.array(rmse_values)
        rmse_promedio = np.mean(rmse_values)
        rmse_std = np.std(rmse_values)
        rmse_max = np.max(rmse_values)
        rmse_min = np.min(rmse_values)

        print(f"RMSE Promedio: {rmse_promedio:.4f}")
        print(f"Desviación Estándar del RMSE: {rmse_std:.4f}")
        print(f"Máximo RMSE: {rmse_max:.4f}")
        print(f"Mínimo RMSE: {rmse_min:.4f}")

    else:
        # Realiza una evaluacion unicamente sobre una imagen experimental contenida en data.
        with torch.no_grad():
            output = model(torch.tensor(data["Py_exp_interp"]).float().to(device))
        t_cgan_caseC = destandarize(output, data["y_mean"], data["y_std"])[0,0,:,:].cpu().numpy()
        t_cgan_caseC = np.ma.masked_where(data["mask"], t_cgan_caseC[::-1])
        
        rmse = root_mean_squared_error(data["t_emi"], t_cgan_caseC)
        print("RMSE: ", rmse)

    # graficara la ultima figura evaluada, si es que se decide evaluar todas las figuras del conjunto de datos experimental
    abs_err = t_cgan_caseC - data["t_emi"]
    abs_err = np.clip(abs_err, -100, 100)
            
    # Graficar
    fig = plt.figure(figsize=(7, 4))
    gs = GridSpec(1, 5, width_ratios=[0.1, 0.2, 0.5, 0.5, 0.5], wspace=0.1, hspace=0.35)
    axes = [fig.add_subplot(gs[0, i]) for i in range(5)]

    axes[1].axis("off")  
    axes[1].set_frame_on(False)
    for ax in axes:
        ax.set_facecolor("darkgrey")

    title = rf'$T_{{s}}$ ' + opt.weights.split("/")[-1].split(".")[0]
    referencia = axcontourf(axes[2], data["r_emi"], data["z_emi"], data["t_emi"], rf'$T_{{s}}$ {exp_name}', levels=np.linspace(1500, t_max, 50), Y_MIN=y_min, Y_MAX=y_max)
    axcontourf(axes[3], data["r"], data["z"], t_cgan_caseC, title, levels=np.linspace(1500, t_max, 50), Y_MIN=y_min, Y_MAX=y_max, show_axes=False)
    diferencia = axcontourf(axes[4], data["r"], data["z"], abs_err, r'$\Delta_{T_{s}}$', levels=np.linspace(-100, 100, 50), CMAP='bwr', Y_MIN=y_min, Y_MAX=y_max, show_axes=False)
    
    cbar_ref = fig.colorbar(referencia, cax=axes[0], location='left', ticks=MaxNLocator(6))
    cbar_ref.ax.yaxis.set_ticks_position('left')
    fig.colorbar(diferencia, ticks=MaxNLocator(6))

    plt.savefig(f'outputs/img/eval_experiment_{title}_{opt.case}.pdf', format='pdf', bbox_inches='tight')
    plt.show()

def compare(opt,model1,model2):
    print("compare two models")
    data = preprocess_data(opt)    
    #---------------------------  model 1 -------------------------------------#
    with torch.no_grad():
        start_time = time.time()
        output_1 = model1(torch.tensor(data["Py_exp_interp"]).float().to(device))
        end_time = time.time()
    print(f"Tiempo de ejecución modelo 1: {end_time - start_time} segundos")

    t_cgan_caseC_1 = destandarize(output_1, data["y_mean"], data["y_std"])[0,0,:,:].cpu().numpy()
    t_cgan_caseC_1 = np.ma.masked_where(data["mask"], t_cgan_caseC_1[::-1])
    
    rmse = root_mean_squared_error(data["t_emi"], t_cgan_caseC_1)
    print("RMSE 1: ", rmse)

    # ----------------- model 2 ---------------------------------------------------#
    with torch.no_grad():
        start_time = time.time() 
        output_2 = model2(torch.tensor(data["Py_exp_interp"]).float().to(device))
        end_time = time.time() 

    print(f"Tiempo de ejecución modelo 2: {end_time - start_time} segundos")
    
    t_cgan_caseC_2 = destandarize(output_2, data["y_mean"], data["y_std"])[0,0,:,:].cpu().numpy()
    t_cgan_caseC_2 = np.ma.masked_where(data["mask"], t_cgan_caseC_2[::-1])
    
    rmse = root_mean_squared_error(data["t_emi"], t_cgan_caseC_2)
    print("RMSE 2: ", rmse)
    
    #-------------------- PLOTS --------------------------------------------------#

    if opt.case == 'A':
        y_min, y_max, t_max = 1, 3.0, 2200
    elif opt.case == 'B':
        y_min, y_max, t_max = 1, 3.5, 2200
    elif opt.case == 'C':
        y_min, y_max, t_max = 1, 5.5, 2100

    # Graficar
    abs_err = t_cgan_caseC_1 - t_cgan_caseC_2

    fig = plt.figure(figsize=(7, 4))
    gs = GridSpec(1, 5, width_ratios=[0.1, 0.2, 0.5, 0.5, 0.5], wspace=0.1, hspace=0.35)
    axes = [fig.add_subplot(gs[0, i]) for i in range(5)]

    axes[1].axis("off")  
    axes[1].set_frame_on(False)
    for ax in axes:
        ax.set_facecolor("darkgrey")

    referencia = axcontourf(axes[2], data["r_emi"], data["z_emi"], t_cgan_caseC_1, rf'$T_{{s}}$ model 1', levels=np.linspace(1500, t_max, 50), Y_MIN=y_min, Y_MAX=y_max)
    axcontourf(axes[3], data["r"], data["z"], t_cgan_caseC_2, rf'$T_{{s}}$ model 2', levels=np.linspace(1500, t_max, 50), Y_MIN=y_min, Y_MAX=y_max, show_axes=False)
    diferencia = axcontourf(axes[4], data["r"], data["z"], abs_err, r'$\Delta_{T_{s}}$', levels=np.linspace(-100, 100, 50), CMAP='bwr', Y_MIN=y_min, Y_MAX=y_max, show_axes=False)
    
    cbar_ref = fig.colorbar(referencia, cax=axes[0], location='left', ticks=MaxNLocator(6))
    cbar_ref.ax.yaxis.set_ticks_position('left')
    fig.colorbar(diferencia, ticks=MaxNLocator(6))

    plt.savefig(f'outputs/img/diferencia_modelos_{opt.case}.pdf', format='pdf', bbox_inches='tight')
    plt.show()

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
    fig = plt.figure(figsize=(12, 7))
    gs = GridSpec(2, 8, width_ratios=[0.1,0.3,0.5, 0.5, 0.5, 0.5,0.2,0.1],wspace=0.1,hspace=0.35)
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

def closeness(opt, model, engines):
    model.eval()
    porcentajes = [0.005, 0.01, 0.1, 0.2, 0.5, 1]
    engine_stats = {name: {'correct': 0, 'total': 0, 'close_counts': [0]*len(porcentajes)} for name in engines.keys()}
    outputs_all_list = []
    imagenes = sorted(os.listdir(opt.dataset))
    
    for img_name in imagenes:
        img_path = os.path.join(opt.dataset, img_name)
        data = process_llamas(img_path)
        data = torch.tensor([data]).float().to(device)
        #if torch.isnan(data).any() or torch.isinf(data).any():
        #    continue
        
        with torch.no_grad():
            output_vanilla = model(data)
        
        outputs_all_list.append(output_vanilla.cpu())

        for name, engine in engines.items():
            engine.eval()
            with torch.no_grad():
                output_engine = engine(data)
            outputs_all_list.append(output_engine.cpu())

    outputs_all = torch.cat(outputs_all_list).flatten()
    max_value = np.percentile(np.abs(outputs_all.numpy()), 90)
    rtols = [p * max_value for p in porcentajes]
    
    for img_name in imagenes:
        img_path = os.path.join(opt.dataset, img_name)
        data = process_llamas(img_path)    
        data = torch.tensor([data]).float().to(device)
        #if torch.isnan(data).any() or torch.isinf(data).any():
        #    continue

        with torch.no_grad():
            output_vanilla = model(data)
        
        num_elementos_por_imagen = output_vanilla.numel()  # Obtiene el número de elementos en la salida del modelo

        for name, engine in engines.items():
            engine.eval()
            with torch.no_grad():
                output_engine = engine(data)
            engine_stats[name]['total'] += 1  # Cuenta el número de imágenes procesadas
            
            for idx, rtol in enumerate(rtols):
                close_values = torch.isclose(output_vanilla, output_engine, atol=rtol, rtol=0)
                engine_stats[name]['close_counts'][idx] += close_values.sum().item()
    
    rtols_header = [f"atol {porcentajes[i]}={rtols[i]:.5f}" for i in range(len(porcentajes))]
    table = "| engine | " + " | ".join(rtols_header) + " |\n" + "|--------" * (len(rtols) + 1) + "|\n"
    
    for name in engines.keys():
        total_comparaciones = engine_stats[name]['total'] * num_elementos_por_imagen  # Total de valores comparados
        if total_comparaciones > 0:
            close_percentages = [f"{100.0 * count / total_comparaciones:.2f}%" for count in engine_stats[name]['close_counts']]
        else:
            close_percentages = ["0.00%" for _ in porcentajes]  # En caso de no haber comparaciones, evitar división por 0
        
        table += f"| {name} | " + " | ".join(close_percentages) + " |\n"
    
    print(table)
    
def load_closeness(opt):
    model_unet = load_model(opt,'unet', 'weights/unet.pth')
    model_attention_unet = load_model(opt,'attunet', 'weights/attunet.pth')
    unet_fp32 = load_model(opt,'tensorrt', 'weights/unet_fp32.engine')
    unet_fp16 = load_model(opt,'tensorrt', 'weights/unet_fp16.engine')
    unet_int8 = load_model(opt,'tensorrt', 'weights/unet_int8.engine')

    attunet_fp32 = load_model(opt,'tensorrt', 'weights/attunet_fp32.engine')
    attunet_fp16 = load_model(opt,'tensorrt', 'weights/attunet_fp16.engine')
    attunet_int8 = load_model(opt,'tensorrt', 'weights/attunet_int8.engine')

    closeness(opt, model_unet, {'fp32':unet_fp32,'fp16':unet_fp16,'int8':unet_int8})  
    closeness(opt, model_attention_unet, {'fp32':attunet_fp32,'fp16':attunet_fp16,'int8':attunet_int8})

def latencia(opt):
    model = load_model(opt, opt.model, opt.weights)
    model.eval()  # Asegurar que el modelo está en modo de evaluación

    tiempos_procesamiento = []
    
    # Recorrer todas las imágenes en el dataset
    imagenes = sorted(os.listdir(opt.dataset))
    batch_size = opt.batch_size if hasattr(opt, 'batch_size') else 1
    
    for i in range(0, len(imagenes), batch_size):
        batch_imgs = imagenes[i:i + batch_size]
        batch_data = []
        
        try:
            for img_name in batch_imgs:
                img_path = os.path.join(opt.dataset, img_name)
                data = process_llamas(img_path)  # Preprocesar la imagen
                batch_data.append(data)
            batch_data = torch.tensor(batch_data).float()

            start_time = time.time()

            batch_data = batch_data.to(device)
            with torch.no_grad():
                output = model(batch_data)  # Realizar la inferencia
                torch.cuda.synchronize()
                output = output.cpu()
                
            end_time = time.time()
            
            tiempos_procesamiento.append(end_time - start_time)
        except Exception as e:
            print(f"Error procesando imágenes {batch_imgs}: {e}")
    
    if tiempos_procesamiento:
        if batch_size == 1:
            # Si el batch size es 1, calcular latencia promedio y máxima
            latencia_maxima = max(tiempos_procesamiento)
            latencia_promedio = sum(tiempos_procesamiento) / len(tiempos_procesamiento)
            return latencia_maxima, latencia_promedio
        else:
            # Si el batch size es mayor a 1, calcular throughput en inferencias por segundo
            throughput = [batch_size / t for t in tiempos_procesamiento if t > 0]
            throughput_promedio = sum(throughput) / len(throughput)
            throughput_maximo = max(throughput)
            return throughput_maximo, throughput_promedio
    else:
        return None, None

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default = 1, type=int,help='batch size')
    parser.add_argument('--dataset', default = 'datasets/img_preprocess', type=str,help='folder to dataset to evaluate latency or thr. images in TIFF format.')
    parser.add_argument('--model', default= 'attunet', type=str, help='modelo a evaluar, puede ser tensorrt, unet o attunet, if compare, use a string separate with a space with the two models to compare.')
    parser.add_argument('--weights', default= 'weights/attunet.pth', type=str, help='path to weights, if compare, use a string separate with a space with the two paths of weights to compare')
    parser.add_argument('--experiment', action='store_true', help='si es experimento ')
    parser.add_argument('--case', default= 'A', type=str, help='condicion de llama, puede ser A, B o C')
    parser.add_argument('--compare', action='store_true', help='si se desea comparar la red optimizada con trt con la vanilla ')
    parser.add_argument('--compare_all', action='store_true', help='Compara todos los modelos para un solo caso en especifico')
    parser.add_argument('--latency', action='store_true', help='Realiza una evaluacion de la latencia, si el batch size uno o thr si el batch size es mayor a uno, tomando en cuenta el dataset en el directorio --dataset')
    parser.add_argument('--closeness', action='store_true', help='Realiza una evaluacion del closeness sobre el --dataset. Para todos los modelos posibles.')
    parser.add_argument('--dataset_experimental','-de',action='store_true',help='Hace la evaluacion sobre todo el datset experimental especificaod en --dataset')
    opt = parser.parse_args()
    return opt

def main(opt):
    output_directory = 'outputs/img'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    if opt.latency:
        l_max,l_ave = latencia(opt)
        if opt.batch_size > 1:
            print("Throughput ave: ", l_ave," inf/s")
        elif opt.batch_size == 1:
            print("Latenica max: ", l_max," s\nLatenica ave: ", l_ave," s")
        else:
            print("Error: Ingrese un batch size valido.")
    elif opt.closeness:
        load_closeness(opt)
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
    elif not (opt.latency or opt.closeness):
        if len(opt.model.split()) != 1 or len( opt.weights.split()) != 1:
            print("ERROR: Debes especificar exactamente uno (modelos y pesos) si no va a comparar.")
        else:
            print(opt.model)
            model = load_model(opt, opt.model, opt.weights)
            if model is None:
                print("Error en la carga del modelo. Revisa las especificaciones.")
            if(opt.experiment):
                eval_exp(opt, model)
            else:
                eval(opt, model) ##  Revisar funcionalidad de eval, parece mal

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
