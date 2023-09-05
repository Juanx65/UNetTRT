import numpy as np 
import os
from utils.functions import max_mean, add_noise,standarize,format_to_train
from sklearn.model_selection import train_test_split

import random
import matplotlib.pyplot as plt

from scipy.interpolate import interp2d

import scipy.io

INPUT_1 = 'R'
INPUT_2 = 'G'
INPUT_3 = 'B'

OUTPUT = 'ts'

NPY_DIR2 = 'npy-PS44'

PERCENT_NOISE = 0.0025
ADD_NOISE = True

class MyDataLoader():
    def __init__(self):
        super(MyDataLoader, self).__init__()
        self.x1 = np.load(os.path.abspath(os.path.join(NPY_DIR2, INPUT_1 + '.npy')))[20:,:,:]
        self.x2 = np.load(os.path.abspath(os.path.join(NPY_DIR2, INPUT_2 + '.npy')))[20:,:,:]
        self.x3 = np.load(os.path.abspath(os.path.join(NPY_DIR2, INPUT_3 + '.npy')))[20:,:,:]
        self.y = np.load(os.path.abspath(os.path.join(NPY_DIR2, OUTPUT + '.npy')))[20:,:,:]
        self.fs = np.load(os.path.abspath(os.path.join(NPY_DIR2, 'fs.npy')))[20:,:,:]
        self.r = np.load(os.path.abspath(os.path.join(NPY_DIR2, 'r_m.npy')))[20:,:]
        self.z = np.load(os.path.abspath(os.path.join(NPY_DIR2, 'z.npy')))[20:,:]

        self.x1 = self.x1[:,:,:32]
        self.x2 = self.x2[:,:,:32]
        self.x3 = self.x3[:,:,:32]
        self.y = self.y[:,:,:32]
        self.fs = self.fs[:,:,:32]
        self.r = self.r[:,:32]
        self.z = self.z[:,:]

        for i in range(len(self.x2)):
            x_max = np.max([self.x1[i].max(),self.x2[i].max(),self.x3[i].max()])
            self.x2[i] = self.x2[i][::-1]/x_max
            self.x3[i] = self.x3[i][::-1]/x_max
            self.x1[i] = self.x1[i][::-1]/x_max
            self.y[i]= self.y[i][::-1]
            self.fs[i] = self.fs[i][::-1]
            self.z[i] = self.z[i][::-1]
            
        self.x1_mean = np.mean(self.x1[:].mean())
        self.x1_std = np.mean(self.x1[:].std())

        self.x2_mean = np.mean(self.x2[:].mean())
        self.x2_std = np.mean(self.x2[:].std())

        self.x3_mean = np.mean(self.x3[:].mean())
        self.x3_std = np.mean(self.x3[:].std())

        self.y_mean = np.mean(self.y[:].mean())
        self.y_std = np.mean(self.y[:].std())

        self.x1_max_mean, _ = max_mean(self.x1)
        self.x2_max_mean, _  = max_mean(self.x2)
        self.x3_max_mean, _  = max_mean(self.x3)

    def load_data(self):
        #### NOISE
        if ADD_NOISE:
            NOISE_STDDEV = [0, self.x3_max_mean*PERCENT_NOISE]
            x1 = add_noise(self.x1, np.random.RandomState(0),NOISE_STDDEV)
            x2 = add_noise(self.x2, np.random.RandomState(1),NOISE_STDDEV)
            x3 = add_noise(self.x3, np.random.RandomState(2),NOISE_STDDEV)
        ####

        x1 = np.expand_dims(x1, axis=3)
        x2 = np.expand_dims(x2, axis=3)
        x3 = np.expand_dims(x3, axis=3)

        x = format_to_train(x1,x2,x3)

        print(x.shape)
        print('Flame images:', x.shape[0])

        x_train, x_valid, y_train, y_valid = train_test_split(x, self.y, test_size=0.2, shuffle=True)

        x_train[:,0,:,:] = standarize(x_train[:,0,:,:], self.x1_mean, self.x1_std)
        x_valid[:,0,:,:] = standarize(x_valid[:,0,:,:], self.x1_mean, self.x1_std)
        x_train[:,1,:,:] = standarize(x_train[:,1,:,:], self.x2_mean, self.x2_std)
        x_valid[:,1,:,:] = standarize(x_valid[:,1,:,:], self.x2_mean, self.x2_std)
        x_train[:,2,:,:] = standarize(x_train[:,2,:,:], self.x3_mean, self.x3_std)
        x_valid[:,2,:,:] = standarize(x_valid[:,2,:,:], self.x3_mean, self.x3_std)

        y_train = standarize(y_train, self.y_mean, self.y_std)
        y_valid = standarize(y_valid, self.y_mean, self.y_std)
        del x1
        del x2
        del x3

        L = 50
        for _ in range(3):
            plt.rcParams['figure.figsize'] = [10, 4]
            n = random.randint(0,len(x_train))
            print(n)
            print("r shape: ", self.r[n].shape) #r shape:  (32, 40)
            print("z shape: ", self.z[n].shape) #z shape:  (88,)
            print("x shape: ", x[n,0,:,:].shape) #x shape:(3, 88)
            print("y shape: ", self.y[n].shape) #x shape:(3, 88)
            print("fs shape: ", self.fs[n].shape) #x shape:(3, 88)

            plt.subplot(151)
            plt.imshow(x[n,0,:,:], cmap= 'jet'), plt.title('$P_{R}$')
            plt.subplot(152)
            plt.imshow(x[n,1,:,:], cmap= 'jet'), plt.title('$P_{G}$')
            plt.subplot(153)
            plt.imshow(x[n,2,:,:], cmap= 'jet'), plt.title('$P_{B}$')
            plt.subplot(154)
            plt.imshow(self.y[n], cmap = 'jet'), plt.title('$T_s$')
            plt.subplot(155)
            plt.imshow(self.fs[n]*1e6, cmap = 'jet'), plt.title('$f_{s}$')
            #plt.colorbar()
            plt.show()

        return x_train,x_valid, y_train, y_valid, self.y_mean, self.y_std

    def load_test_data(self):

        x1_test = np.load(os.path.abspath(os.path.join(NPY_DIR2, INPUT_1 + '.npy')))[:20,:,:]
        x2_test = np.load(os.path.abspath(os.path.join(NPY_DIR2, INPUT_2 + '.npy')))[:20,:,:]
        x3_test = np.load(os.path.abspath(os.path.join(NPY_DIR2, INPUT_3 + '.npy')))[:20,:,:]
        y_test = np.load(os.path.abspath(os.path.join(NPY_DIR2, OUTPUT + '.npy')))[:20,:,:]
        fs_test = np.load(os.path.abspath(os.path.join(NPY_DIR2, 'fs.npy')))[:20,:,:]
        r_test = np.load(os.path.abspath(os.path.join(NPY_DIR2, 'r_m.npy')))[:20,:]
        z_test = np.load(os.path.abspath(os.path.join(NPY_DIR2, 'z.npy')))[:20,:]

        x1_test = x1_test[:,:,:32]
        x2_test = x2_test[:,:,:32]
        x3_test = x3_test[:,:,:32]
        y_test = y_test[:,:,:32]
        fs_test = fs_test[:,:,:32]
        r_test = r_test[:,:32]
        z_test = z_test[:,:]
        #%n = 0
        #%print(r_test[n].shape, z_test[n].shape,x1_test[n].shape)
        #%plt.contourf(r_test[n], z_test[n],x1_test[n], cmap = 'jet', levels = 50)
        #%plt.show()

        for i in range(len(x2_test)):
            x_max = np.max([x1_test[i].max(),x2_test[i].max(),x3_test[i].max()])
            x1_test[i] = x1_test[i][::-1]/x_max
            x2_test[i] = x2_test[i][::-1]/x_max
            x3_test[i] = x3_test[i][::-1]/x_max
            y_test[i] = y_test[i][::-1]
            fs_test[i] = fs_test[i][::-1]
            z_test[i] = z_test[i][::-1]

        ################################################################

        ### NOISE
        if ADD_NOISE:
            NOISE_STDDEV = self.x3_max_mean * PERCENT_NOISE
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

        print(x1_test.shape)
        x_test = format_to_train(x1_test,x2_test,x3_test)

        x_test[:,0,:,:] = standarize(x_test[:,0,:,:], self.x1_mean, self.x1_std)
        x_test[:,1,:,:] = standarize(x_test[:,1,:,:], self.x2_mean, self.x2_std)
        x_test[:,2,:,:] = standarize(x_test[:,2,:,:], self.x3_mean, self.x3_std)


        print("x_test shape: ", x_test.shape)
        #n = 0
        #print(r_test[n].shape, z_test[n].shape,x1_test[n,:,:,0].shape)
        #plt.contourf(r_test[n], z_test[n],x1_test[n,:,:,0], cmap = 'jet', levels = 50)
        #plt.show()
        
        for _ in range(3):
            plt.rcParams['figure.figsize'] = [10, 4]
            n = random.randint(0,20)
            print(n)
            print("r shape: ", r_test[n].shape) #r shape:  (32, 40)
            print("z shape: ", z_test[n].shape) #z shape:  (88,)
            print("x_test shape: ", x_test[n].shape) #x shape:(3, 88)
            print("y_test shape: ", y_test[n].shape) #x shape:(3, 88)
            print("fs_test shape: ", fs_test[n].shape) #x shape:(3, 88)

            plt.subplot(151)
            plt.contourf(r_test[n], z_test[n],x_test[n,0,:,:], cmap= 'jet', levels = 50), plt.title('$P_{R}$')
            plt.subplot(152)
            plt.contourf(r_test[n], z_test[n],x_test[n,1,:,:], cmap= 'jet', levels = 50), plt.title('$P_{G}$')
            plt.subplot(153)
            plt.contourf(r_test[n], z_test[n],x_test[n,2,:,:], cmap= 'jet', levels = 50), plt.title('$P_{B}$')
            plt.subplot(154)
            plt.contourf(r_test[n], z_test[n],y_test[n], cmap = 'jet', levels = 50), plt.title('$T_s$')
            plt.subplot(155)
            plt.contourf(r_test[n], z_test[n],fs_test[n]*1e6, cmap = 'jet', levels = 50), plt.title('$f_{s}$')
            #plt.colorbar()
            plt.show()

        return x_test, y_test, self.y_mean, self.y_std, fs_test, r_test, z_test

    def load_data_exp(self):
        Y_MIN = 1
        Y_MAX = 3.5
        percentage_noise = 0.0
        mat_BEMI = os.path.abspath('npy-PS44/ts_BEMI_B2040_case_A.mat')
        mat_EMI = os.path.abspath('npy-PS44/ts_EMI_case_A.mat')

        mat = scipy.io.loadmat(mat_EMI)
        ts_emi_A = mat.get('EMI')
        Py = mat.get('Py')
        r_emi_A = mat.get('r')[0,:]
        z_emi_A = mat.get('z')[0,:]
        r_emi, z_emi, t_emi = resize_temp(r_emi_A ,z_emi_A , ts_emi_A)
        r_emi,z_emi, Py = resize_temp(r_emi_A,z_emi_A , Py)
        
        mat = scipy.io.loadmat(mat_BEMI)
        Py_exp = mat.get('Py_rgb')
        r_exp = mat.get('r')
        z_exp = mat.get('z')
        t_bemi = mat.get('BEMI')[:,:,3]

        #*****************************************************************+++++++++++++
        # Select the P_B channel and calculate the range of the values in the matrix
        m_range = np.max(Py_exp[:,:, 1]) #- np.min(Py_rgb[:,:, 0])
        # Calculate the standard deviation of the Gaussian noise (0.5% of the range)
        #noise_std = percentage_noise * m_range
        noise_std = (percentage_noise/100) * m_range
        noise = np.random.normal(loc=0, scale=noise_std, size=Py_exp[:,:, 0].shape)
        Py_exp[:,:, 0] += noise
        Py_exp[:,:, 1] += noise
        Py_exp[:,:, 2] += noise

        #*****************************************************************+++++++++++++

        r, z, t_bemi = resize_temp(r_exp, z_exp, t_bemi)
        Py_exp_interp = np.empty((3,128,32))
        r, z, Py_exp_interp[0,:,:] = resize_temp(r_exp, z_exp, Py_exp[:,:,0])
        r, z, Py_exp_interp[1,:,:] = resize_temp(r_exp, z_exp, Py_exp[:,:,1])
        r, z, Py_exp_interp[2,:,:] = resize_temp(r_exp, z_exp, Py_exp[:,:,2])
        del Py_exp

        x_max = np.max([Py_exp_interp])
        Py_exp_interp[:,0,:,:]= Py_exp_interp[:,0,:,:] [::-1]/x_max
        Py_exp_interp[:,1,:,:]  = Py_exp_interp[:,1,:,:][::-1]/x_max
        Py_exp_interp[:,2,:,:] = Py_exp_interp[:,2,:,:] [::-1]/x_max

        Py_exp_interp[:,0,:,:] = standarize(Py_exp_interp[:,0,:,:] , self.x1_mean, self.x1_std)
        Py_exp_interp[:,1,:,:] = standarize(Py_exp_interp[:,1,:,:] , self.x2_mean, self.x2_std)
        Py_exp_interp[:,2,:,:]  = standarize(Py_exp_interp[:,2,:,:] , self.x3_mean, self.x3_std)

        Py_exp_interp = np.expand_dims(Py_exp_interp, axis=0)

        print("x shape: ", Py_exp_interp.shape)

        return Py_exp_interp, t_emi
    

def resize_temp(r, z, Tp):
        # Definir la nueva cuadrícula con dimensiones de 128x40
        new_r = np.linspace(0, 0.32, 32)
        new_z = np.linspace(1, 7.6, 128)
        f = interp2d(r, z, Tp, kind='linear', copy=True, bounds_error=False, fill_value=None)
        new_temp = f(new_r, new_z)
        return new_r, new_z, new_temp
            
