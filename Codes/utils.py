import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import librosa  
from matrices import *
def gpu_test():
    if tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
        return True

def mix_fixed_SNR(Signal,Noise,Desired_SNR_dB):
    Npts = len(Signal); # Number of input time samples
    Signal_Power = np.sum(abs(Signal)*abs(Signal))/Npts
    Noise_Power = np.sum(abs(Noise)*abs(Noise))/Npts
    Initial_SNR = 10*(np.log10(Signal_Power/Noise_Power))
    K = (Signal_Power/Noise_Power)*10**(-Desired_SNR_dB/10) # Scale factor

    New_Noise = np.sqrt(K)*Noise; # Change Noise level    
    Noisy_Signal = Signal + New_Noise
    return Noisy_Signal

def check_SNR_non_merged(Xtest,Ytest,SNR,model):
    est=model.predict(np.array(Xtest))
    shape=est.shape[0]
    snr_array=np.zeros(shape)
    prd_array=np.zeros(shape)
    rmse_array=np.zeros(shape)

    for L in range(shape):
        snr=SI_SNR(est[L],Ytest[L])
        prd1=prd(Xtest[L],est[L])
        rmse1=rmse(est[L],Ytest[L])
        snr_array[L]=snr
        prd_array[L]=prd1
        rmse_array[L]=rmse1
    print("------")
    print('REAL_SNR')
    print(SNR)
    print('PREDICTED_SNR')
    print(np.mean(snr_array))
    print('PRD_SNR')
    print(np.mean(prd_array))
    print('RMSE_SNR')
    print(np.mean(rmse_array))
    
    return est[1]


def mergeX(X_train,seg):
    index=0
    merge=[]
    for L in range(int(X_train.shape[0]/seg)):
        temp=X_train[index]
        for i in range(index,index+seg-1):
            temp=np.concatenate([temp,X_train[i+1]])
        temp = temp/max(abs(temp))   
        merge.append(temp)
        
        index+=seg
    x = np.array(merge)
    
    return x
