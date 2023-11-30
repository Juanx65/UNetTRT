import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.notebook import tqdm

TITLE = 'U-Net_B2040_syn_A'
RESULTS_DIR = './results/'

def standarize(data, mean, std):
    return (data - mean) / std

def destandarize(data, mean, std):
    return (data * std) + mean

# Definir MAE con destandarización
def mae_destandarize(y_true, y_pred, y_mean, y_std):
    y_true_d = destandarize(y_true, y_mean, y_std)
    y_pred_d = destandarize(y_pred, y_mean, y_std)
    return torch.mean(torch.abs(y_pred_d - y_true_d))

# Definir MAE en porcentaje
def mae_percentage(y_true, y_pred, y_mean, y_std):
    y_true_d = destandarize(y_true, y_mean, y_std)
    y_pred_d = destandarize(y_pred, y_mean, y_std)
    mae = torch.mean(torch.abs(y_pred_d - y_true_d))
    return (mae * 100) / y_mean

def save_results(data, name):
    try:
        os.mkdir(RESULTS_DIR)
    except:
        pass
    try:
        os.mkdir(os.path.join(RESULTS_DIR, TITLE))
    except:
        pass
    dir_name = os.path.join(RESULTS_DIR, TITLE, name)
    try:
        os.mkdir(dir_name)
    except:
        pass
    for i in tqdm(range(data.shape[0]), total=data.shape[0], desc=name, unit=' files'):
        np.savetxt(os.path.join(dir_name, name + '_' + str(i) + '.dat'), data[i], delimiter='\n')

def add_noise(data, rng,NOISE_STDDEV):
    assert np.size(NOISE_STDDEV) == 1 or np.size(NOISE_STDDEV) == 2
    if np.size(NOISE_STDDEV) == 1:
        data += rng.normal(0, NOISE_STDDEV, data.shape)
    else:
        s = rng.uniform(NOISE_STDDEV[0], NOISE_STDDEV[1], len(data))
        for i, d in enumerate(data):
            d += rng.normal(0, s[i], d.shape)
    return data

def format_to_train(x1,x2,x3):
    x1_t = np.transpose(x1, [0,3, 1, 2])
    x2_t = np.transpose(x2, [0,3, 1, 2])
    x3_t = np.transpose(x3, [0,3, 1, 2])
    xd = np.concatenate((x1_t,x2_t, x3_t), axis=1)
    #xd = np.transpose(xd, [0,1,2, 3]) # para entrenar con pytorch size: (N, C, H, W)
    return xd

def max_mean(B1):
    B_max = list()
    for i in range(len(B1)):
        B_max.append(B1[i,:,:].max())
    B_max_mean = np.array(B_max).mean()
    B_max_std = np.array(B_max).std()

    print('max_mean: ',B_max_mean, 'Std: ',B_max_std, 'Min: ', np.array(B_max).min(), 'Max: ', np.array(B_max).max())
    return B_max_mean, B_max_std 


def plot_img(x_train, y_train, INPUT1, INPUT2, INPUT3, OUTPUT,  label):
    plt.rcParams['figure.figsize'] = [18,4]
    plt.subplot(141)
    plt.imshow(x_train[:,:,0], cmap= 'jet')
    plt.title(INPUT1+'-'+str(label))
    plt.colorbar()
    plt.subplot(142)
    plt.imshow(x_train[:,:,1], cmap= 'jet')
    plt.title(INPUT2+'-'+str(label))
    plt.colorbar()
    plt.subplot(143)
    plt.imshow(x_train[:,:,2], cmap= 'jet')
    plt.title(INPUT3+'-'+str(label))
    plt.colorbar()
    plt.subplot(144)
    plt.imshow(y_train, cmap= 'jet', vmin=y_train.max()*0.6, vmax=y_train.max())
    plt.title(OUTPUT+'-'+str(label))
    plt.colorbar()
    plt.show()
    
def destandarize_py(x_train,x1_mean,x1_std,x2_mean,x2_std,x3_mean,x3_std):
    d1 = x_train[:,:,0]
    d1 = destandarize(d1, x1_mean,x1_std)
    d2 = x_train[:,:,1]
    d2 = destandarize(d2, x2_mean,x2_std)
    d3 = x_train[:,:,2]
    d3 = destandarize(d3, x3_mean,x3_std)
    d1 = d1.reshape(d1.shape[0],d1.shape[1], 1)
    d2 = d2.reshape(d2.shape[0],d2.shape[1], 1)
    d3 = d3.reshape(d3.shape[0],d3.shape[1], 1)

    d = np.concatenate((d1,d2,d3), axis =2)
    return d