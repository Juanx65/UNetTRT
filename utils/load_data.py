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
INPUT_4 = 'r_axis'

NPY_DIR1 = 'dataset-combustion/npy-PS44'
NPY_DIR2 = 'dataset-combustion/npy-PSB40-4'
NPY_DIR3 = 'dataset-combustion/npy-PSB60'
NPY_DIR4 = 'dataset-combustion/npy-PSB80'
#NPY_DIR1 = '/home/jorge/mnt/dataset-combustion/Simulations_coflame_yale/npy-PS44'
#NPY_DIR2 = '/home/jorge/mnt/dataset-combustion/Simulations_coflame_yale/npy-PSB40-4'
#NPY_DIR3 = '/home/jorge/mnt/dataset-combustion/Simulations_coflame_yale/npy-PSB60'
#NPY_DIR4 = '/home/jorge/mnt/dataset-combustion/Simulations_coflame_yale/npy-PSB80F'

PERCENT_NOISE = 0.0025
ADD_NOISE = True
n_samples = 7500

class MyDataLoader():
    def __init__(self):
        super(MyDataLoader, self).__init__()
        self.x1_A = np.load(os.path.abspath(os.path.join(NPY_DIR1, INPUT_1 + '.npy')))[20:n_samples,:,:]
        self.x2_A = np.load(os.path.abspath(os.path.join(NPY_DIR1, INPUT_2 + '.npy')))[20:n_samples,:,:]
        self.x3_A = np.load(os.path.abspath(os.path.join(NPY_DIR1, INPUT_3 + '.npy')))[20:n_samples,:,:]
        self.y_A = np.load(os.path.abspath(os.path.join(NPY_DIR1, OUTPUT + '.npy')))[20:n_samples,:,:]
        self.fs_A = np.load(os.path.abspath(os.path.join(NPY_DIR1, 'fs.npy')))[20:n_samples,:,:]
        self.r_A = np.load(os.path.abspath(os.path.join(NPY_DIR1, INPUT_4  +'.npy')))[20:n_samples,0,:]
        #self.r_A = np.load(os.path.abspath(os.path.join(NPY_DIR1, INPUT_4  +'.npy')))[20:n_samples,:]
        
        self.z_A = np.load(os.path.abspath(os.path.join(NPY_DIR1, 'z.npy')))[20:n_samples,:]

        self.x1_B = np.load(os.path.abspath(os.path.join(NPY_DIR2, INPUT_1 + '.npy')))[20:5010,:,:]
        self.x2_B = np.load(os.path.abspath(os.path.join(NPY_DIR2, INPUT_2 + '.npy')))[20:5010,:,:]
        self.x3_B = np.load(os.path.abspath(os.path.join(NPY_DIR2, INPUT_3 + '.npy')))[20:5010,:,:]
        self.y_B = np.load(os.path.abspath(os.path.join(NPY_DIR2, OUTPUT + '.npy')))[20:5010,:,:]
        self.fs_B = np.load(os.path.abspath(os.path.join(NPY_DIR2, 'fs.npy')))[20:5010,:,:]
        #self.r_B = np.load(os.path.abspath(os.path.join(NPY_DIR2, INPUT_4  +'.npy')))[20:5010,:]
        self.r_B = np.load(os.path.abspath(os.path.join(NPY_DIR2, INPUT_4  +'.npy')))[20:5010,0,:]
        self.z_B = np.load(os.path.abspath(os.path.join(NPY_DIR2, 'z.npy')))[20:5010,:]

        self.x1_C = np.load(os.path.abspath(os.path.join(NPY_DIR3, INPUT_1 + '.npy')))[20:n_samples,:,:]
        self.x2_C = np.load(os.path.abspath(os.path.join(NPY_DIR3, INPUT_2 + '.npy')))[20:n_samples,:,:]
        self.x3_C = np.load(os.path.abspath(os.path.join(NPY_DIR3, INPUT_3 + '.npy')))[20:n_samples,:,:]
        self.y_C = np.load(os.path.abspath(os.path.join(NPY_DIR3, OUTPUT + '.npy')))[20:n_samples,:,:]
        self.fs_C = np.load(os.path.abspath(os.path.join(NPY_DIR3, 'fs.npy')))[20:n_samples,:,:]
        self.r_C = np.load(os.path.abspath(os.path.join(NPY_DIR3, INPUT_4  +'.npy')))[20:n_samples,0,:]
        #self.r_C = np.load(os.path.abspath(os.path.join(NPY_DIR3, INPUT_4  +'.npy')))[20:n_samples,:]
        self.z_C = np.load(os.path.abspath(os.path.join(NPY_DIR3, 'z.npy')))[20:n_samples,:]

        self.x1_D = np.load(os.path.abspath(os.path.join(NPY_DIR4, INPUT_1 + '.npy')))[20:n_samples,:128,:]
        self.x2_D = np.load(os.path.abspath(os.path.join(NPY_DIR4, INPUT_2 + '.npy')))[20:n_samples,:128,:]
        self.x3_D = np.load(os.path.abspath(os.path.join(NPY_DIR4, INPUT_3 + '.npy')))[20:n_samples,:128,:]
        self.y_D = np.load(os.path.abspath(os.path.join(NPY_DIR4, OUTPUT + '.npy')))[20:n_samples,:128,:]
        self.fs_D = np.load(os.path.abspath(os.path.join(NPY_DIR4, 'fs.npy')))[20:n_samples,:128,:]
        self.r_D = np.load(os.path.abspath(os.path.join(NPY_DIR4, INPUT_4  +'.npy')))[20:n_samples,:]
        self.z_D = np.load(os.path.abspath(os.path.join(NPY_DIR4, 'z.npy')))[20:n_samples,:128]
    
        self.x1 = np.concatenate([self.x1_A,self.x1_B,self.x1_C,self.x1_D], axis = 0)
        self.x2 = np.concatenate([self.x2_A,self.x2_B,self.x2_C,self.x2_D], axis = 0)
        self.x3 = np.concatenate([self.x3_A,self.x3_B,self.x3_C,self.x3_D], axis = 0)
        self.y = np.concatenate([self.y_A,self.y_B,self.y_C,self.y_D], axis = 0)
        self.fs = np.concatenate([self.fs_A,self.fs_B,self.fs_C,self.fs_D], axis = 0)
        self.r = np.concatenate([self.r_A,self.r_B,self.r_C,self.r_D], axis = 0)
        self.z = np.concatenate([self.z_A,self.z_B,self.z_C,self.z_D], axis = 0)

        del self.x1_A, self.x1_B, self.x1_C, self.x1_D
        del self.x2_A, self.x2_B, self.x2_C, self.x2_D
        del self.x3_A, self.x3_B, self.x3_C, self.x3_D
        del self.y_A, self.y_B, self.y_C, self.y_D
        del self.fs_A, self.fs_B, self.fs_C, self.fs_D
        del self.r_A, self.r_B, self.r_C, self.r_D
        del self.z_A, self.z_B, self.z_C, self.z_D

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

        """ L = 50
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
            plt.show() """

        return x_train,x_valid, y_train, y_valid, self.y_mean, self.y_std

    def load_test_data(self):
        self.x1_A_test = np.load(os.path.abspath(os.path.join(NPY_DIR1, INPUT_1 + '.npy')))[:20,:,:]
        self.x2_A_test = np.load(os.path.abspath(os.path.join(NPY_DIR1, INPUT_2 + '.npy')))[:20,:,:]
        self.x3_A_test = np.load(os.path.abspath(os.path.join(NPY_DIR1, INPUT_3 + '.npy')))[:20,:,:]
        self.y_A_test = np.load(os.path.abspath(os.path.join(NPY_DIR1, OUTPUT + '.npy')))[:20,:,:]
        self.fs_A_test = np.load(os.path.abspath(os.path.join(NPY_DIR1, 'fs.npy')))[:20,:,:]
        self.r_A_test = np.load(os.path.abspath(os.path.join(NPY_DIR1, INPUT_4 + '.npy')))[:20,0,:]
        #r_A_test = np.load(os.path.abspath(os.path.join(NPY_DIR1, INPUT_4 + '.npy')))[:20,:]
        self.z_A_test = np.load(os.path.abspath(os.path.join(NPY_DIR1, 'z.npy')))[:20,:]

        self.x1_B_test = np.load(os.path.abspath(os.path.join(NPY_DIR2, INPUT_1 + '.npy')))[:20,:,:]
        self.x2_B_test = np.load(os.path.abspath(os.path.join(NPY_DIR2, INPUT_2 + '.npy')))[:20,:,:]
        self.x3_B_test = np.load(os.path.abspath(os.path.join(NPY_DIR2, INPUT_3 + '.npy')))[:20,:,:]
        self.y_B_test = np.load(os.path.abspath(os.path.join(NPY_DIR2, OUTPUT + '.npy')))[:20,:,:]
        self.fs_B_test = np.load(os.path.abspath(os.path.join(NPY_DIR2, 'fs.npy')))[:20,:,:]
        self.r_B_test = np.load(os.path.abspath(os.path.join(NPY_DIR2, 'r_m.npy')))[:20,0,:]
        #r_B_test = np.load(os.path.abspath(os.path.join(NPY_DIR2, INPUT_4 + '.npy')))[:20,:]
        self.z_B_test = np.load(os.path.abspath(os.path.join(NPY_DIR2, 'z.npy')))[:20,:]

        self.x1_C_test = np.load(os.path.abspath(os.path.join(NPY_DIR3, INPUT_1 + '.npy')))[:20,:,:]
        self.x2_C_test = np.load(os.path.abspath(os.path.join(NPY_DIR3, INPUT_2 + '.npy')))[:20,:,:]
        self.x3_C_test = np.load(os.path.abspath(os.path.join(NPY_DIR3, INPUT_3 + '.npy')))[:20,:,:]
        self.y_C_test = np.load(os.path.abspath(os.path.join(NPY_DIR3, OUTPUT + '.npy')))[:20,:,:]
        self.fs_C_test = np.load(os.path.abspath(os.path.join(NPY_DIR3, 'fs.npy')))[:20,:,:]
        #r_C_test = np.load(os.path.abspath(os.path.join(NPY_DIR3, INPUT_4 +'.npy')))[:20,:]
        self.r_C_test = np.load(os.path.abspath(os.path.join(NPY_DIR3, INPUT_4 +'.npy')))[:20,0,:]
        self.z_C_test = np.load(os.path.abspath(os.path.join(NPY_DIR3, 'z.npy')))[:20,:]

        self.x1_D_test = np.load(os.path.abspath(os.path.join(NPY_DIR4, INPUT_1 + '.npy')))[:20,:128,:]
        self.x2_D_test = np.load(os.path.abspath(os.path.join(NPY_DIR4, INPUT_2 + '.npy')))[:20,:128,:]
        self.x3_D_test = np.load(os.path.abspath(os.path.join(NPY_DIR4, INPUT_3 + '.npy')))[:20,:128,:]
        self.y_D_test = np.load(os.path.abspath(os.path.join(NPY_DIR4, OUTPUT + '.npy')))[:20,:128,:]
        self.fs_D_test = np.load(os.path.abspath(os.path.join(NPY_DIR4, 'fs.npy')))[:20,:128,:]
        #r_D_test = np.load(os.path.abspath(os.path.join(NPY_DIR4, 'r_m.npy')))[:20,:]
        self.r_D_test = np.load(os.path.abspath(os.path.join(NPY_DIR4, INPUT_4 + '.npy')))[:20,:]
        self.z_D_test = np.load(os.path.abspath(os.path.join(NPY_DIR4, 'z.npy')))[:20,:128]

        self.x1_test = np.concatenate([self.x1_A_test,self.x1_B_test,self.x1_C_test,self.x1_D_test], axis = 0)
        self.x2_test = np.concatenate([self.x2_A_test,self.x2_B_test,self.x2_C_test,self.x2_D_test], axis = 0)
        self.x3_test = np.concatenate([self.x3_A_test,self.x3_B_test,self.x3_C_test,self.x3_D_test], axis = 0)
        self.y_test = np.concatenate([self.y_A_test,self.y_B_test,self.y_C_test,self.y_D_test], axis = 0)
        self.fs_test = np.concatenate([self.fs_A_test,self.fs_B_test,self.fs_C_test,self.fs_D_test], axis = 0)
        self.r_test = np.concatenate([self.r_A_test,self.r_B_test,self.r_C_test,self.r_D_test], axis = 0)
        self.z_test = np.concatenate([self.z_A_test,self.z_B_test,self.z_C_test,self.z_D_test], axis = 0)

        self.x1_test = self.x1_test[:,:,:32]
        self.x2_test = self.x2_test[:,:,:32]
        self.x3_test = self.x3_test[:,:,:32]
        self.y_test = self.y_test[:,:,:32]
        self.fs_test = self.fs_test[:,:,:32]
        self.r_test = self.r_test[:,:32]
        self.z_test = self.z_test[:,:]

        for i in range(len(x2_test)):
            x_max = np.max([self.x1_test[i].max(),self.x2_test[i].max(),self.x3_test[i].max()])
            self.x1_test[i] = self.x1_test[i][::-1]/x_max
            self.x2_test[i] = self.x2_test[i][::-1]/x_max
            self.x3_test[i] = self.x3_test[i][::-1]/x_max
            self.y_test[i] = self.y_test[i][::-1]
            self.fs_test[i] = self.fs_test[i][::-1]
            self.z_test[i] = self.z_test[i][::-1]

        ################################################################

        ### NOISE
        if ADD_NOISE:
            NOISE_STDDEV = self.x3_max_mean * PERCENT_NOISE
            #NOISE_STDDEV = [0, x3_max_mean*percent_noise]
            x1_test = add_noise(self.x1_test, np.random.RandomState(0),NOISE_STDDEV)
            x2_test = add_noise(self.x2_test, np.random.RandomState(1),NOISE_STDDEV)
            x3_test = add_noise(self.x3_test, np.random.RandomState(2),NOISE_STDDEV)
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
        
        return x_test, y_test, self.y_mean, self.y_std, fs_test, r_test, z_test
    
    def load_data_exp_A(self):
        percentage_noise = 0.0
        mat_BEMI = os.path.abspath('data_experimental/BEMI/32/ts_BEMI_B2040_case_A.mat')
        mat_EMI = os.path.abspath('data_experimental/MAE-EMI/32/ts_EMI_case_A.mat')

        mat = scipy.io.loadmat(mat_EMI)
        ts_emi_A = mat.get('EMI')
        Py = mat.get('Py')
        r_emi_A = mat.get('r')[0,:]
        z_emi_A = mat.get('z')[0,:]
        r_emi, z_emi, t_emi = resize_temp(r_emi_A ,z_emi_A-0.12 , ts_emi_A)
        r_emi,z_emi, Py = resize_temp(r_emi_A,z_emi_A-0.12 , Py)
        
        mat = scipy.io.loadmat(mat_BEMI)
        Py_exp = mat.get('Py_rgb')
        r_exp = mat.get('r')
        z_exp = mat.get('z')
        t_bemi = mat.get('BEMI')[:,:,3]

        #*****************************************************************+++++++++++++
        # Select the P_B channel and calculate the range of the values in the matrix
        m_range = np.max(Py_exp[:,:,1]) #- np.min(Py_rgb[:,:, 0])
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
        Py_exp_interp[0,:,:] = Py_exp_interp[0,:,:][::-1]/x_max
        Py_exp_interp[1,:,:] = Py_exp_interp[1,:,:][::-1]/x_max
        Py_exp_interp[2,:,:] = Py_exp_interp[2,:,:][::-1]/x_max

        Py_exp_interp[0,:,:] = standarize(Py_exp_interp[0,:,:] , self.x1_mean, self.x1_std)
        Py_exp_interp[1,:,:] = standarize(Py_exp_interp[1,:,:] , self.x2_mean, self.x2_std)
        Py_exp_interp[2,:,:]  = standarize(Py_exp_interp[2,:,:] , self.x3_mean, self.x3_std)

        Py_exp_interp = np.expand_dims(Py_exp_interp, axis=0)

        print("x shape: ", Py_exp_interp.shape)

        return Py_exp_interp, t_emi, t_bemi, r_emi, z_emi, Py, r, z

    def load_data_exp_B(self):
        percentage_noise = 0.0
        mat_BEMI = os.path.abspath('data_experimental/BEMI/40/ts_BEMI_B2040_case_B.mat')
        mat_EMI = os.path.abspath('data_experimental/MAE-EMI/40/ts_EMI_case_B.mat')
        
        mat = scipy.io.loadmat(mat_EMI)
        ts_emi_B = mat.get('EMI')
        Py = mat.get('Py')
        Py = Py/Py.max()
        r_emi_B = mat.get('r')[0,:]
        z_emi_B = mat.get('z')[0,:]
        _, _, Py = resize_temp(r_emi_B ,z_emi_B , Py)
        r_emi, z_emi, t_emi = resize_temp(r_emi_B ,z_emi_B , ts_emi_B)

        mat = scipy.io.loadmat(mat_BEMI)
        Py_exp = mat.get('Py_rgb')
        Py_exp  = Py_exp/Py_exp.max()
        r_exp = mat.get('r')
        z_exp = mat.get('z')
        t_bemi = mat.get('BEMI')[:,:,3]

        m_range = np.max(Py_exp[:,:, 1])
        noise_std = (percentage_noise/100) * m_range
        noise = np.random.normal(loc=0, scale=noise_std, size=Py_exp[:,:, 0].shape)
        Py_exp[:,:, 0] += noise
        Py_exp[:,:, 1] += noise
        Py_exp[:,:, 2] += noise
        #*****************************************************************+++++++++++++
        r, z, t_bemi = resize_temp(r_exp, z_exp, t_bemi)
        Py_exp_interp = np.empty((3,128,32))
        r, z, Py_exp_interp[0,:,:] = resize_temp(r_exp-0.034, z_exp+0.135, Py_exp[:,:,0])
        r, z, Py_exp_interp[1,:,:] = resize_temp(r_exp-0.034, z_exp+0.135, Py_exp[:,:,1])
        r, z, Py_exp_interp[2,:,:] = resize_temp(r_exp-0.034, z_exp+0.135, Py_exp[:,:,2])

        del Py_exp

        x_max = np.max([Py_exp_interp])
        Py_exp_interp[0,:,:] = Py_exp_interp[0,:,:][::-1]/x_max
        Py_exp_interp[1,:,:] = Py_exp_interp[1,:,:][::-1]/x_max
        Py_exp_interp[2,:,:] = Py_exp_interp[2,:,:][::-1]/x_max

        Py_exp_interp[0,:,:] = standarize(Py_exp_interp[0,:,:], self.x1_mean, self.x1_std)
        Py_exp_interp[1,:,:] = standarize(Py_exp_interp[1,:,:], self.x2_mean, self.x2_std)
        Py_exp_interp[2,:,:] = standarize(Py_exp_interp[2,:,:], self.x3_mean, self.x3_std)

        Py_exp_interp = np.expand_dims(Py_exp_interp, axis=0)
        print("x shape: ", Py_exp_interp.shape)
        
        return Py_exp_interp, t_emi, t_bemi, r_emi, z_emi, Py, r, z
    
    def load_data_exp_C(self):
        percentage_noise = 0.0
        mat_BEMI = os.path.abspath('data_experimental/BEMI/60/ts_BEMI_B2040_case_C.mat')
        
        path_mae = 'data_experimental/MAE-EMI/'
        mat = scipy.io.loadmat(path_mae + '60/T.mat')
        T_mae = mat.get('T')[:,:,0]
        mat = scipy.io.loadmat(path_mae + '60/fv.mat')
        fv = mat.get('fv')[:,:,0]
        mat = scipy.io.loadmat(path_mae + '60/r.mat')
        r_mae = mat.get('r')[0,:]*100
        mat = scipy.io.loadmat(path_mae + '60/z.mat')
        z_mae = mat.get('z')[0,:]*100
        mat = scipy.io.loadmat(path_mae + '60/Sy_cal.mat')
        Sy_cal = mat.get('Sy_cal')
        Sy_cal = Sy_cal[:,:,1]/Sy_cal[:,:,1].max()

        print(r_mae.shape,z_mae.shape, fv.shape)
        _,_,fv = resize_temp(r_mae,z_mae, fv)
        _,_,Sy_cal = resize_temp(r_mae,z_mae, Sy_cal)
        r_mae,z_mae,T_mae = resize_temp(r_mae,z_mae, T_mae)
    

        
        mat = scipy.io.loadmat(mat_BEMI)
        Py_exp = mat.get('Py_rgb')
        Py_exp  = Py_exp/Py_exp.max()
        r_exp = mat.get('r')
        z_exp = mat.get('z')
        t_bemi = mat.get('BEMI')[:,:,3]

        m_range = np.max(Py_exp[:,:, 1])
        noise_std = (percentage_noise/100) * m_range
        noise = np.random.normal(loc=0, scale=noise_std, size=Py_exp[:,:, 0].shape)
        Py_exp[:,:, 0] += noise
        Py_exp[:,:, 1] += noise
        Py_exp[:,:, 2] += noise
        #*****************************************************************+++++++++++++
        r, z, t_bemi = resize_temp(r_exp, z_exp, t_bemi)
        Py_exp_interp = np.empty((3,128,32))
        r, z, Py_exp_interp[0,:,:] = resize_temp(r_exp, z_exp-0.1, Py_exp[:,:,0])
        r, z, Py_exp_interp[1,:,:] = resize_temp(r_exp, z_exp-0.1, Py_exp[:,:,1])
        r, z, Py_exp_interp[2,:,:] = resize_temp(r_exp, z_exp-0.1, Py_exp[:,:,2])

        del Py_exp

        x_max = np.max([Py_exp_interp])
        Py_exp_interp[0,:,:] = Py_exp_interp[0,:,:][::-1]/x_max
        Py_exp_interp[1,:,:] = Py_exp_interp[1,:,:][::-1]/x_max
        Py_exp_interp[2,:,:] = Py_exp_interp[2,:,:][::-1]/x_max

        Py_exp_interp[0,:,:] = standarize(Py_exp_interp[0,:,:], self.x1_mean, self.x1_std)
        Py_exp_interp[1,:,:] = standarize(Py_exp_interp[1,:,:], self.x2_mean, self.x2_std)
        Py_exp_interp[2,:,:] = standarize(Py_exp_interp[2,:,:], self.x3_mean, self.x3_std)

        Py_exp_interp = np.expand_dims(Py_exp_interp, axis=0)
        print("x shape: ", Py_exp_interp.shape)
        
        return Py_exp_interp, T_mae, t_bemi, r_mae, z_mae, Sy_cal, r, z


def resize_temp(r, z, Tp):
        # Definir la nueva cuadrícula con dimensiones de 128x40
        new_r = np.linspace(0, 0.32, 32)
        new_z = np.linspace(1, 7.6, 128)
        f = interp2d(r, z, Tp, kind='linear', copy=True, bounds_error=False, fill_value=None)
        new_temp = f(new_r, new_z)
        return new_r, new_z, new_temp

