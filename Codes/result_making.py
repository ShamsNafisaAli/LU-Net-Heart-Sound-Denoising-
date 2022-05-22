import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from processing_initial import get_files_and_resample,get_files_and_resamplePascal
from config import *
from utils import *
import pandas as pd

#Use the pretrained weights from Models folder or use your own check point's directory if you train by yourself
model = keras.models.load_model('LU-Net.h5') 
# Result making step for Open access heart sound dataset
for snr in [-6,-3, 0,3,6]:
    XtestL,YtestL,label1= get_files_and_resample(1000, 0.8, locH = pathheartVal,locN = pathhospitalval, db_SNR = snr, mode = 1)
    est_testL=check_SNR_non_merged(XtestL,YtestL,snr,model)
# Result making step for Pascal heart sound dataset
testX,testY= get_files_and_resamplePascal(1000,.8, locH=pathPascal)
# est_min5 = model.predict(np.array(testXE1))
est_test=mergeX(testXE1,seg=3)
est_test = np.reshape(est_test, (est_test.shape[0], -1))
df1 = pd.DataFrame(est_test1)
df1.to_csv('lunet_denoised_pascal.csv',index=True)

