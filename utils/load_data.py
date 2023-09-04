import numpy as np 
import os
from utils.functions import max_mean, add_noise,standarize,format_to_train
from sklearn.model_selection import train_test_split

import random
import matplotlib.pyplot as plt

INPUT_1 = 'R'
INPUT_2 = 'G'
INPUT_3 = 'B'

OUTPUT = 'ts'

NPY_DIR2 = 'npy-PS44'

PERCENT_NOISE = 0.0025
ADD_NOISE = True

def load_data():
    x1 = np.load(os.path.abspath(os.path.join(NPY_DIR2, INPUT_1 + '.npy')))[20:,:,:]
    x2 = np.load(os.path.abspath(os.path.join(NPY_DIR2, INPUT_2 + '.npy')))[20:,:,:]
    x3 = np.load(os.path.abspath(os.path.join(NPY_DIR2, INPUT_3 + '.npy')))[20:,:,:]
    y = np.load(os.path.abspath(os.path.join(NPY_DIR2, OUTPUT + '.npy')))[20:,:,:]
    fs = np.load(os.path.abspath(os.path.join(NPY_DIR2, 'fs.npy')))[20:,:,:]
    r = np.load(os.path.abspath(os.path.join(NPY_DIR2, 'r.npy')))[20:,:]
    z = np.load(os.path.abspath(os.path.join(NPY_DIR2, 'z.npy')))[20:,:]

    x1 = x1[:,:88,:32]
    x2 = x2[:,:88,:32]
    x3 = x3[:,:88,:32]
    y = y[:,:88,:32]
    fs = fs[:,:88,:32]
    r = r[:,:32]
    z = z[:,:88]

    for i in range(len(x2)):
        x_max = np.max([x1[i].max(),x2[i].max(),x3[i].max()])
        x2[i] = x2[i][::-1]/x_max
        x3[i] = x3[i][::-1]/x_max
        x1[i] = x1[i][::-1]/x_max
        y[i]= y[i][::-1]
        fs[i] = fs[i][::-1]
        z[i] = z[i][::-1]
        
    x1_mean = np.mean(x1[:].mean())
    x1_std = np.mean(x1[:].std())

    x2_mean = np.mean(x2[:].mean())
    x2_std = np.mean(x2[:].std())

    x3_mean = np.mean(x3[:].mean())
    x3_std = np.mean(x3[:].std())

    y_mean = np.mean(y[:].mean())
    y_std = np.mean(y[:].std())

    x1_max_mean, _ = max_mean(x1)
    x2_max_mean, _  = max_mean(x2)
    x3_max_mean, _  = max_mean(x3)
    #### NOISE
    if ADD_NOISE:
        NOISE_STDDEV = [0, x3_max_mean*PERCENT_NOISE]
        x1 = add_noise(x1, np.random.RandomState(0),NOISE_STDDEV)
        x2 = add_noise(x2, np.random.RandomState(1),NOISE_STDDEV)
        x3 = add_noise(x3, np.random.RandomState(2),NOISE_STDDEV)
    ####

    x1 = np.expand_dims(x1, axis=3)
    x2 = np.expand_dims(x2, axis=3)
    x3 = np.expand_dims(x3, axis=3)

    x = format_to_train(x1,x2,x3)

    print(x.shape)
    print('Flame images:', x.shape[0])

    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, shuffle=True)

    x_train[:,:,:,0] = standarize(x_train[:,:,:,0], x1_mean, x1_std)
    x_valid[:,:,:,0] = standarize(x_valid[:,:,:,0], x1_mean, x1_std)
    x_train[:,:,:,1] = standarize(x_train[:,:,:,1], x2_mean, x2_std)
    x_valid[:,:,:,1] = standarize(x_valid[:,:,:,1], x2_mean, x2_std)
    x_train[:,:,:,2] = standarize(x_train[:,:,:,2], x3_mean, x3_std)
    x_valid[:,:,:,2] = standarize(x_valid[:,:,:,2], x3_mean, x3_std)

    y_train = standarize(y_train, y_mean, y_std)
    y_valid = standarize(y_valid, y_mean, y_std)
    del x1
    del x2
    del x3

    """ L = 50
    for _ in range(3):
        plt.rcParams['figure.figsize'] = [10, 4]
        n = random.randint(0,len(x_train))
        print(n)
        print("r shape: ", r[n].shape) #r shape:  (32, 40)
        print("z shape: ", z[n].shape) #z shape:  (88,)
        print("r shape: ", x[n,0,:,:]) #x shape:(3, 88)
        print("y shape: ", y[n]) #x shape:(3, 88)
        print("y shape: ", fs[n]) #x shape:(3, 88)

        plt.subplot(151)
        plt.imshow(x[n,0,:,:], cmap= 'jet'), plt.title('$P_{R}$')
        plt.subplot(152)
        plt.imshow(x[n,1,:,:], cmap= 'jet'), plt.title('$P_{G}$')
        plt.subplot(153)
        plt.imshow(x[n,2,:,:], cmap= 'jet'), plt.title('$P_{B}$')
        plt.subplot(154)
        plt.imshow(y[n], cmap = 'jet'), plt.title('$T_s$')
        plt.subplot(155)
        plt.imshow(fs[n]*1e6, cmap = 'jet'), plt.title('$f_{s}$')
        #plt.colorbar()
        plt.show() """

    return x_train,x_valid, y_train, y_valid, y_mean, y_std

def load_test_data():

    #############################################################
    x1 = np.load(os.path.abspath(os.path.join(NPY_DIR2, INPUT_1 + '.npy')))[20:,:,:]
    x2 = np.load(os.path.abspath(os.path.join(NPY_DIR2, INPUT_2 + '.npy')))[20:,:,:]
    x3 = np.load(os.path.abspath(os.path.join(NPY_DIR2, INPUT_3 + '.npy')))[20:,:,:]
    y = np.load(os.path.abspath(os.path.join(NPY_DIR2, OUTPUT + '.npy')))[20:,:,:]
    fs = np.load(os.path.abspath(os.path.join(NPY_DIR2, 'fs.npy')))[20:,:,:]
    r = np.load(os.path.abspath(os.path.join(NPY_DIR2, 'r.npy')))[20:,:]
    z = np.load(os.path.abspath(os.path.join(NPY_DIR2, 'z.npy')))[20:,:]

    x1 = x1[:,:88,:32]
    x2 = x2[:,:88,:32]
    x3 = x3[:,:88,:32]
    y = y[:,:88,:32]
    fs = fs[:,:88,:32]
    r = r[:,:32]
    z = z[:,:88]

    for i in range(len(x2)):
        x_max = np.max([x1[i].max(),x2[i].max(),x3[i].max()])
        x2[i] = x2[i][::-1]/x_max
        x3[i] = x3[i][::-1]/x_max
        x1[i] = x1[i][::-1]/x_max
        y[i]= y[i][::-1]
        fs[i] = fs[i][::-1]
        z[i] = z[i][::-1]
        
    x1_mean = np.mean(x1[:].mean())
    x1_std = np.mean(x1[:].std())

    x2_mean = np.mean(x2[:].mean())
    x2_std = np.mean(x2[:].std())

    x3_mean = np.mean(x3[:].mean())
    x3_std = np.mean(x3[:].std())

    y_mean = np.mean(y[:].mean())
    y_std = np.mean(y[:].std())

    x1_max_mean, _ = max_mean(x1)
    x2_max_mean, _  = max_mean(x2)
    x3_max_mean, _  = max_mean(x3)

    x1_test = np.load(os.path.abspath(os.path.join(NPY_DIR2, INPUT_1 + '.npy')))[:20,:,:]
    x2_test = np.load(os.path.abspath(os.path.join(NPY_DIR2, INPUT_2 + '.npy')))[:20,:,:]
    x3_test = np.load(os.path.abspath(os.path.join(NPY_DIR2, INPUT_3 + '.npy')))[:20,:,:]
    y_test = np.load(os.path.abspath(os.path.join(NPY_DIR2, OUTPUT + '.npy')))[:20,:,:]
    fs_test = np.load(os.path.abspath(os.path.join(NPY_DIR2, 'fs.npy')))[:20,:,:]
    r_test = np.load(os.path.abspath(os.path.join(NPY_DIR2, 'r.npy')))[:20,:]
    z_test = np.load(os.path.abspath(os.path.join(NPY_DIR2, 'z.npy')))[:20,:]

    for i in range(len(x2_test)):
        x_max = np.max([x1_test[i].max(),x2_test[i].max(),x3_test[i].max()])
        x1_test[i] = x1_test[i][::-1]/x_max
        x2_test[i] = x2_test[i][::-1]/x_max
        x3_test[i] = x3_test[i][::-1]/x_max
        y_test[i] = y_test[i][::-1]
        fs_test[i] = fs_test[i][::-1]
        z_test[i] = z_test[i][::-1]
    
    x1_test = x1_test[:,128-88:,:32]
    x2_test = x2_test[:,128-88:,:32]
    x3_test = x3_test[:,128-88:,:32]
    y_test = y_test[:,128-88:,:32]
    fs_test = fs_test[:,128-88:,:32]
    r_test = r_test[:,:32]
    z_test = z_test[:,128-88:]

    ################################################################

    ### NOISE
    if ADD_NOISE:
        NOISE_STDDEV = x3_max_mean * PERCENT_NOISE
        #NOISE_STDDEV = [0, x3_max_mean*percent_noise]
        x1_test = add_noise(x1_test, np.random.RandomState(0),NOISE_STDDEV)
        x2_test = add_noise(x2_test, np.random.RandomState(1),NOISE_STDDEV)
        x3_test = add_noise(x3_test, np.random.RandomState(2),NOISE_STDDEV)
    ###
    x1_test[x1_test<0] = 0
    x2_test[x2_test<0] = 0
    x3_test[x3_test<0] = 0 
        
    x1_test = np.expand_dims(x1_test, axis=3)
    x2_test = np.expand_dims(x2_test, axis=3)
    x3_test = np.expand_dims(x3_test, axis=3)

    x_test = format_to_train(x1_test,x2_test,x3_test)

    x_test[:,:,:,0] = standarize(x_test[:,:,:,0], x1_mean, x1_std)
    x_test[:,:,:,1] = standarize(x_test[:,:,:,1], x2_mean, x2_std)
    x_test[:,:,:,2] = standarize(x_test[:,:,:,2], x3_mean, x3_std)

    return x_test, y_test, y_mean, y_std, fs_test, r_test, z_test
