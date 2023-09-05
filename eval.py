import torch
import random
import os
import numpy as np
from models.unet import U_Net
import matplotlib.pyplot as plt
from utils.functions import destandarize
from utils.load_data import MyDataLoader
from matplotlib.ticker import MaxNLocator

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Using CPU')
else:
    print('CUDA is available. Using GPU')

device = torch.device("cuda:0" if train_on_gpu else "cpu")

# Hyperparameters
num_filters = 29
val_dropout = 0.0
learning_rate = 0.001
batch_size = 32

# LOAD DATA
my_data_loader = MyDataLoader()
#x_test, y_test, y_mean, y_std, fs_test, r_test, z_test= my_data_loader.load_test_data()
_, _, y_mean, y_std, _, _, _= my_data_loader.load_test_data()
x_test, t_emi = my_data_loader.load_data_exp()

x_test = torch.tensor(x_test).float().to(device)
#y_test = torch.tensor(y_test).float().to(device)


####LOAD######
model = U_Net(n1=num_filters, dropout_rate=val_dropout)
model = torch.load('weights/best.pth')
model.to(device)
model.eval()

# Eval 
with torch.no_grad():
        output = model(x_test)

x_test = x_test.cpu()
#y_test = y_test.cpu()

y_test_pred = destandarize(output, y_mean, y_std)
y_test_pred = y_test_pred.squeeze(1)

y_test_pred = y_test_pred.cpu()

print("x_test size", x_test.size())

#N = random.randint(0,len(x_test)-1)
N = 11

""" y_test[N][fs_test[N] <= 0.05e-6] = 0
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
plt.colorbar(ticks=MaxNLocator(6)) """


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

""" print('Abs. error max:', abs_err.max())
print('Abs. error min:', abs_err.min())
print('Abs. error mean:', abs_err.mean())
print('Abs. error stddev:', abs_err.std())
print('Abs. error %:', abs_err.mean()*100/y_test[N].mean()) """