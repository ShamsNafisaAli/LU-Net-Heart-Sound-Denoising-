import tensorflow as tf
from tensorflow import keras
from processing_initial import get_files_and_resample

from config import *
from utilis import *
model=keras.models.load_model('check_points/lunet.h5')
for snr in [-6,-3,0,3,6]:
    XtestL,YtestL,label1= get_files_and_resampleP(1000,0.8,locH=pathheartVal,locN=pathhospitalval, db_SNR =snr,mode=1)
    est_testL=check_SNR_non_merged(XtestL,YtestL,snr,model)
