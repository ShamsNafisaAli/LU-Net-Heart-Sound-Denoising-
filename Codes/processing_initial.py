import numpy as np
import os
import soundfile as sf
from scipy import signal
import librosa
from utils import mix_fixed_SNR

def get_files_and_resample(sampling_rate_new, desired_length_seconds,locH,locN ,db_SNR = 0, mode=0):

    print("xxxxxxxxxxxxxxxxxxxxxxxxxxx------start-------xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print(mode)
    startpath = os.path.abspath(locH)
    noisepath1 = os.path.abspath(locN)
    list_with_file_names1 = os.listdir(startpath) 
    print(list_with_file_names1)
    snr=[-6,-3,0,3,6]
    duration_samples = int(desired_length_seconds * sampling_rate_new)
    print(duration_samples)
    y_list = []
    x_list =[]
    label=[]
    index = 0
    lab=-1
    noise_i=0
    
    for file1 in list_with_file_names1:  
        completePath1 = os.path.join(startpath,file1) 
        list_with_file_names2 = sorted(os.listdir(completePath1))
        list_with_file_names21 = sorted(os.listdir(noisepath1))
        snr_index=0
        index = 0
        lab += 1
        for snrx in snr:
            noise_i=0
            for file2 in list_with_file_names2: 
                completePath2 = os.path.join(completePath1,file2)
                completePath21 = os.path.join(noisepath1,list_with_file_names21[noise_i])
                input_file,sampling_rate_orig = librosa.load(completePath2, sampling_rate_new)
                input_file1,sampling_rate_orig = librosa.load(completePath21, sampling_rate_new, duration=len(input_file)/sampling_rate_new)
                k=int(len(input_file)/duration_samples)
          
                if len(input_file) == 9990 or len(input_file) == 10000 or len(input_file) == 3500 or len(input_file) == 4995 :
                    xtem=[]
                    ytem=[]
                    ltem=[]
                    skip=0
                    for i in range(k):
                
                        x_files=input_file[i*duration_samples:duration_samples*(i+1)]
                        real_signal=(x_files/max(abs(x_files)))
                        y_files=input_file1[i*duration_samples:duration_samples*(i+1)] 
                        if mode==0:
                            noise_signal=mix_fixed_SNR(real_signal,(y_files/max(abs(y_files))),snrx)
                        elif mode ==1:
                            noise_signal=mix_fixed_SNR(real_signal,(y_files/max(abs(y_files))),db_SNR)
                        if np.isnan(noise_signal.max()):
                            skip=1
                            continue
                        else:
                            xtem.append(noise_signal)
                            ytem.append(real_signal)
                            ltem.append(lab)
                    if skip==0:
                        x_list=x_list+xtem
                        y_list=y_list+ytem
                        label=label+ltem
                noise_i += 1

    print('XXXXXXXXXXXXXXXXXXXX-------end---------XXXXXXXXXXXXXXXXXXXXXXXX')
    x = np.array(x_list)
    y = np.array(y_list)
    labelk=np.array(label)  
    return x[...,np.newaxis],y[...,np.newaxis],labelk
