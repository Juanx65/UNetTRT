    import glob
    import scipy.io    
    from scipy.interpolate import interp2d
    from matplotlib.ticker import MaxNLocator
    def axcontourf(ax,r,z, data, title):
        x = ax.contourf(r, z,data,levels, cmap = CMAP)#,vmin =VMIN, vmax = VMAX)
        ax.set_xlabel('r (cm)')
        ax.set_ylabel('z (cm)')
        ax.set_title(title)
        ax.set_xlim(0,0.45)
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_xticks(ticks=[0, 0.25])#, labels=[0, T_0.shape[1]])
        return x 
    def resize_temp(r, z, Tp):
        # Definir la nueva cuadrícula con dimensiones de 128x40
        new_r = np.linspace(0, 0.32, 32)
        new_z = np.linspace(1, 7.6, 128)
        f = interp2d(r, z, Tp, kind='linear', copy=True, bounds_error=False, fill_value=None)
        new_temp = f(new_r, new_z)
        return new_r, new_z, new_temp

    MODEL_FILE = './'+TITLE +'/'+ TITLE + '2.h5'
    print(MODEL_FILE)
    
    
    Y_MIN = 1
    Y_MAX = 3.5
    percentage_noise = 0.0
    path_emi = '/home/jorge/flame/final-new/Paper_Measurements_2023/Campaña_Yale/MAE/'

    mat = scipy.io.loadmat(path_emi+ 'ts_EMI_case_A.mat')
    ts_emi_A = mat.get('EMI')
    #ts_emi_A[ts_emi_A>2100] = np.nan
    #ts_emi_A[ts_emi_A<1500] = np.nan
    Py = mat.get('Py')
    r_emi_A = mat.get('r')[0,:]
    z_emi_A = mat.get('z')[0,:]
    r_emi, z_emi, t_emi = resize_temp(r_emi_A ,z_emi_A , ts_emi_A)
    r_emi,z_emi, Py = resize_temp(r_emi_A,z_emi_A , Py)
    #z_emi = z_emi - 0.163
    mat = scipy.io.loadmat('/home/jorge/flame/final-new/Paper_Measurements_2023/
                           Campaña_Yale/BEMI/results/ts_BEMI_B2040_case_A.mat')
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
    Py_exp_interp = np.empty((128,32,3))
    r, z, Py_exp_interp[:,:,0] = resize_temp(r_exp, z_exp, Py_exp[:,:,0])
    r, z, Py_exp_interp[:,:,1] = resize_temp(r_exp, z_exp, Py_exp[:,:,1])
    r, z, Py_exp_interp[:,:,2] = resize_temp(r_exp, z_exp, Py_exp[:,:,2])
    del Py_exp

    x_max = np.max([Py_exp_interp])
    Py_exp_interp[:,:,0] = Py_exp_interp[:,:,0][::-1]/x_max
    Py_exp_interp[:,:,1] = Py_exp_interp[:,:,1][::-1]/x_max
    Py_exp_interp[:,:,2] = Py_exp_interp[:,:,2][::-1]/x_max

    Py_exp_interp[:,:,0] = standarize(Py_exp_interp[:,:,0], x1_mean, x1_std)
    Py_exp_interp[:,:,1] = standarize(Py_exp_interp[:,:,1], x2_mean, x2_std)
    Py_exp_interp[:,:,2] = standarize(Py_exp_interp[:,:,2], x3_mean, x3_std)

    Py_exp_interp = np.expand_dims(Py_exp_interp, axis=0)
    ####LOAD######
    model = tf.keras.models.load_model(
        MODEL_FILE,
        custom_objects={'mae_destandarize': mae_destandarize, 'mae_percentage': mae_percentage}
    )

    model.compile(optimizer= Adam(lr = 0.0005), loss='mse', metrics=[mae_percentage])

    t_cgan_case = model.predict(Py_exp_interp)
    t_cgan_caseC = destandarize(t_cgan_case, y_mean, y_std)[0,:,:,0]
    t_cgan_caseC2 = destandarize(t_cgan_case, y_mean, y_std)[0,:,:,0]

    mask = t_emi<1
    t_emi = np.ma.masked_where(mask, t_emi)
    t_bemi = np.ma.masked_where(mask, t_bemi)
    t_cgan_caseC = t_cgan_caseC[::-1]
    t_cgan_caseC = np.ma.masked_where(mask,t_cgan_caseC)
    for i in range(3):
        Py_exp_interp[0,:,:,i] = np.ma.masked_where(mask,Py_exp_interp[0,:,:,i])

    plt.rcParams['figure.figsize'] = [10, 4]
    levels = 50
    fig, ax = plt.subplots(1,6)
    CMAP = 'jet'
    im = axcontourf(ax[0],r,z, Py_exp_interp[0,:,:,0][::-1], 'R')
    axcontourf(ax[1],r,z, Py_exp_interp[0,:,:,1][::-1], 'G')
    axcontourf(ax[2],r,z, Py_exp_interp[0,:,:,2][::-1], 'B')
    
    levels = np.linspace(1500,2100,50)
    axcontourf(ax[3],r_emi,z_emi, t_emi,r'$T_{s}$(EMI)')
    im3 = axcontourf(ax[4],r,z, t_bemi,r'$T_{s}$ (BEMI)')
    axcontourf(ax[5],r,z,t_cgan_caseC,r'$T_{s}$(U-Net)')

    ax[0].set_facecolor("darkblue")  
    ax[1].set_facecolor("darkblue")  
    ax[2].set_facecolor("darkblue")  
    ax[3].set_facecolor("darkblue")  
    ax[4].set_facecolor("darkblue")  
    ax[5].set_facecolor("darkblue")    
    cbar = fig.colorbar(im3, ticks=MaxNLocator(6))#.tick_values(1500,2300))
    fig.tight_layout()
    #fig.savefig(f'./results_U-Net_case_A_A.jpg', bbox_inches='tight', dpi = 300, quality = 100)
    #scipy.io.savemat('./results/ts_UNET-CBAM_BEMI_B2040_dataA_case_A_training_noise_'+str(percent_noise)+'.mat', {'UNET':t_cgan_caseC2,'r':r,'z':z})

    plt.rcParams['figure.figsize'] = [6, 4]
    fig, ax = plt.subplots(1,3)
    CMAP = 'jet'
    levels = np.linspace(1500,2100,50)
    axcontourf(ax[0],r_emi,z_emi, t_emi,r'$T_{s}(EMI)$')
    CMAP = 'bwr'
    levels = np.linspace(-80,80,50)
    im3 = axcontourf(ax[1],r,z, t_bemi - t_emi,r'$\Delta_t$ BEMI')
    abs_err = t_cgan_caseC - t_emi
    axcontourf(ax[2],r,z,abs_err,'$\Delta_t$ U-Net')
    abs_err[abs_err>100] = 100
    abs_err[abs_err<-100] = -100
    
    ax[0].set_facecolor("darkblue")  
    ax[1].set_facecolor("darkblue")  
    ax[2].set_facecolor("darkblue")  
    cbar = fig.colorbar(im3, ticks=MaxNLocator(6))#.tick_values(1500,2300))
    fig.tight_layout()
    
    print('Abs. error max:', abs_err.max())
    print('Abs. error min:', abs_err.min())
    print('Abs. error mean:', abs_err.mean())
    print('Abs. error stddev:', abs_err.std())
    print('Abs. error %:', abs_err.mean()*100/t_emi.mean())