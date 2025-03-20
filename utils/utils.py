import os
import torch
import numpy as np
import subprocess
import re
from utils.load_data import MyDataLoader
from utils import engine
import cv2

from scipy.interpolate import interp2d

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available. Using CPU')
else:
    print('CUDA is available. Using GPU')
device = torch.device("cuda:0" if train_on_gpu else "cpu")

def preprocess_data(opt):
    """
    Preprocesa los datos de entrada según el caso especificado en opt.case.
    
    Args:
        opt: objeto con la configuración del experimento.
        
    Returns:
        Diccionario con los datos preprocesados.
    """
    my_data_loader = MyDataLoader()
    _, _, y_mean, y_std, _, _, _ = my_data_loader.load_test_data()
    
    if opt.case == 'A':
        Py_exp_interp, t_emi, t_bemi, r_emi, z_emi, Py, r, z = my_data_loader.load_data_exp_A()
    elif opt.case == 'B':
        Py_exp_interp, t_emi, t_bemi, r_emi, z_emi, Py, r, z = my_data_loader.load_data_exp_B()
    elif opt.case == 'C':
        Py_exp_interp, t_emi, t_bemi, r_emi, z_emi, Sy_cal, r, z = my_data_loader.load_data_exp_C()
    else:
        raise ValueError(f"Caso {opt.case} no soportado.")
    
    Py_exp_interp = torch.tensor(Py_exp_interp).float().to(device)
    mask = t_emi < 1
    
    t_emi = np.ma.masked_where(mask, t_emi)
    t_bemi = np.ma.masked_where(mask, t_bemi)
    
    Py_exp_interp = Py_exp_interp.cpu().numpy()
    
    for i in range(Py_exp_interp.shape[1]):
        Py_exp_interp[0, i, :, :] = np.ma.masked_where(mask, Py_exp_interp[0, i, :, :])
    
    data = {
        "mask": mask,
        "Py_exp_interp": Py_exp_interp,
        "t_emi": t_emi,
        "t_bemi": t_bemi,
        "r_emi": r_emi,
        "z_emi": z_emi,
        "r": r,
        "z": z,
        "y_mean": y_mean,
        "y_std": y_std,
    }
    
    if opt.case in ['A', 'B']:
        data["Py"] = Py
    elif opt.case == 'C':
        data["Sy_cal"] = Sy_cal
    
    return data

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
    print('----------------------------------------------------------\n')
    print('Modelo ', model_name, 'path = ', weight,'\n')

    if model_name == 'tensorrt':
        current_directory = os.path.dirname(os.path.abspath(__file__))
        parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        engine_path = os.path.join(parent_directory, weight)
        model = engine.TRTModule(engine_path, device)
        model.set_desired(['outputs'])    
        
        print("# capas onnx = ",get_layers(model_name,weight))
        print("# parametro onnx = ",get_parametros(model_name,weight))
    elif model_name == 'unet' or model_name == 'attunet':
        model = torch.load(weight)
        model.to(device)
        model.eval()
        print("# capas base = ", get_parameters_vanilla(model)[0])
        print("# parametros base = ", get_parameters_vanilla(model)[1])
    else:
        print(f'ERROR: especifica un modelo válido, opciones: tensorrt, unet, attunet. Modelo dado: {opt.model}')
        return None
    print("tamaño = ",get_model_size_MB(weight), " MB")
    return model

def axcontourf(ax, r, z, data, title, levels=50, Y_MIN=1, Y_MAX=3.5, CMAP='jet',show_axes=True,ftitle=None):
    if isinstance(levels, list) and len(levels) > 1:
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

def root_mean_squared_error(t_emi,t_cgan_caseC):
    def mse(actual, predicted):
        differences = actual -  predicted
        differences[differences > 100] = 0
        differences[differences < -100] = 0
        squared_differences = np.square(differences)
        return np.nanmean(squared_differences)
    mse_value = mse(t_emi,t_cgan_caseC)
    rmse = round(np.sqrt(mse_value),2)
    return rmse

def process_experimental_input(opt,image_dir):

    if opt.case == "A":
        Py_exp_interp = process_condA(image_dir)
    elif opt.case == "B":
        Py_exp_interp = process_condB(image_dir)
    elif opt.case == "C":
        Py_exp_interp = process_condC(image_dir)
    else:
        print("Condicion no implementada")
        return
    return Py_exp_interp

def process_condA(image_dir):
    NPY_DIR2 = 'datasets/npy-PS44'
    INPUT_1 = 'R'
    INPUT_2 = 'G'
    INPUT_3 = 'B'

    ## PRe PRE procesamiento-------------------------------------------------------#
    x1 = np.load(os.path.abspath(os.path.join(NPY_DIR2, INPUT_1 + '.npy')))[20:,:,:]
    x2 = np.load(os.path.abspath(os.path.join(NPY_DIR2, INPUT_2 + '.npy')))[20:,:,:]
    x3 = np.load(os.path.abspath(os.path.join(NPY_DIR2, INPUT_3 + '.npy')))[20:,:,:]

    x1 = x1[:,:,:32]
    x2 = x2[:,:,:32]
    x3 = x3[:,:,:32]

    for i in range(len(x2)):
        x_max = np.max([x1[i].max(),x2[i].max(),x3[i].max()])
        x2[i] = x2[i][::-1]/x_max
        x3[i] = x3[i][::-1]/x_max
        x1[i] = x1[i][::-1]/x_max
        
    x1_mean = np.mean(x1[:].mean())
    x1_std = np.mean(x1[:].std())

    x2_mean = np.mean(x2[:].mean())
    x2_std = np.mean(x2[:].std())

    x3_mean = np.mean(x3[:].mean())
    x3_std = np.mean(x3[:].std())

    ## autentica carga de datos-----------------------------------------------------#

    ss = cv2.imread(image_dir,cv2.IMREAD_UNCHANGED)

    ss_new = np.zeros((600,2048,3))
    ss_new[:,:,0] = ss[800:1400,:, 0]
    ss_new[:,:,1] = ss[800:1400,:, 1]
    ss_new[:,:,2] = ss[800:1400,:, 2]

    R_exp=np.zeros_like(ss_new)
    BG = 25.318288541666668
    CONST1 = 0.24409398022534526
    R_exp[:,:,0]=((ss_new[:,:,2]- BG))*CONST1

    CONST2 = 0.6647056891885273
    R_exp[:,:,1]=((ss_new[:,:,1]- BG))*CONST2

    CONST3 = 1.7375992007958856
    R_exp[:,:,2]=((ss_new[:,:,0]- BG))*CONST3

    Py_rot = np.zeros((2048,1536,3))
    Py_rot[:,:,0] = np.rot90(ss[:,:, 0], k=1, axes=(0, 1))
    Py_rot[:,:,1] = np.rot90(ss[:,:, 1], k=1, axes=(0, 1))
    Py_rot[:,:,2] = np.rot90(ss[:,:, 2], k=1, axes=(0, 1))
    r_x = int(150)
    r1 = int(1000)
    r2 = int(1140)
    h_px = int(1300)
    m =  np.where(Py_rot[h_px,r1:r2,1] == Py_rot[h_px,r1:r2,1].min())[0][0]
    center_x = r1 + m
    border_x  = center_x + r_x
    
    Py_rgb  = np.zeros((3,2048,border_x - center_x)) 
    Py_rgb[0,:,:] = Py_rot[:,center_x:border_x, 0]
    Py_rgb[1,:,:] = Py_rot[:,center_x:border_x, 1]
    Py_rgb[2,:,:] = Py_rot[:,center_x:border_x, 2]

    scale = 37294.15914879467
    offset_z = -0.3/100
    nx = r_x     
    Zmin = int(2048)  # initial height to consider
    Zmax = 0  # max height to consider
    offset_z = -0.3/100
    nz = Zmin - Zmax

    r_exp = np.linspace(0, nx - 1, nx) / scale
    z_exp = np.linspace(Zmin - Zmax, 0, nz) / scale + offset_z

    Py_exp_interp = np.empty((3,128,32))

    r, z, Py_exp_interp[0,:,:] = resize_temp(r_exp, z_exp, Py_rgb[0,:,:])
    r, z, Py_exp_interp[1,:,:] = resize_temp(r_exp, z_exp, Py_rgb[1,:,:])
    r, z, Py_exp_interp[2,:,:] = resize_temp(r_exp, z_exp, Py_rgb[2,:,:])

    epsilon = 1e-10  # Pequeña constante para evitar división por cero
    #x_max = x_max if x_max > 0 else epsilon
    Py_exp_interp[0,:,:] = Py_exp_interp[0,:,:][::-1]/x_max
    Py_exp_interp[1,:,:] = Py_exp_interp[1,:,:][::-1]/x_max
    Py_exp_interp[2,:,:] = Py_exp_interp[2,:,:][::-1]/x_max

    Py_exp_interp[0,:,:] = standarize(Py_exp_interp[0,:,:] , x1_mean, x1_std)
    Py_exp_interp[1,:,:] = standarize(Py_exp_interp[1,:,:] , x2_mean, x2_std)
    Py_exp_interp[2,:,:] = standarize(Py_exp_interp[2,:,:] , x3_mean, x3_std)

    #print("Py_exp_interp shape: ", Py_exp_interp.shape)
    return Py_exp_interp

def process_condB(image_dir):
    Py_exp_interp = []
    return Py_exp_interp

def process_condC(image_dir):
    Py_exp_interp = []
    return Py_exp_interp

def standarize(data, mean, std):
    return (data - mean) / std

def resize_temp(r, z, Tp):
    # Definir la nueva cuadrícula con dimensiones de 128x40
    new_r = np.linspace(0, 0.32, 32)
    new_z = np.linspace(1, 7.6, 128)
    f = interp2d(r, z, Tp, kind='linear', copy=True, bounds_error=False, fill_value=None)
    new_temp = f(new_r, new_z)
    return new_r, new_z, new_temp