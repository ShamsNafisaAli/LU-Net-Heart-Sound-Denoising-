import math
import numpy as np
import pandas as pd

def SI_SNR(est, egs):
    '''
         Calculate the SNR indicator between the two audios. 
         The larger the value, the better the separation.
         input:
               _s: Generated audio
               s:  Ground Truth audio
         output:
               SNR value 
    '''
    est=np.squeeze(est)
    egs=np.squeeze(egs)
    length = len(est)
    _s = est[0:length]
    s =egs[0:length]
    noise=_s-s
    noise_power=sum(pow(noise,2))/len(noise)
    pre_signal_power=sum(pow(s,2))/len(s)
    snr = 10*math.log10(pre_signal_power/noise_power)
    return snr
  
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
def prd(noisy, targets):
    return np.sqrt((((noisy - targets) ** 2).mean())/(((noisy) ** 2).mean()))

def SDR(est, egs):
    '''
        calculate SDR
        est: Network generated audio
        egs: Ground Truth
    '''
    length = len(est)
    print(length)
    _s = est[0:length]
    print(_s.shape)
    s =egs[0:length]
    print(s.shape)
    (sdr, isr, sir, sar, perm) = museval.metrics.bss_eval(s,_s)
    return sdr,isr,sar
