import os
import torch
import numpy as np
import subprocess
import re
from utils.load_data import MyDataLoader
from utils import engine

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