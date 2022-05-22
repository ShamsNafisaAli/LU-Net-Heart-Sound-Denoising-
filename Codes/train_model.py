import tensorflow
from tensorflow import keras 
from tensorflow.keras.callbacks import ModelCheckpoint
from model import enhancement_model
from processing_initial import get_files_and_resampleP
from processing_spect_generate_save import get_files_and_resample_spect
from config import *
import os


X1,Y1,labeltrain = get_files_and_resampleP(sampling_rate_new, window_size,locH=path_Heart_Train,locN=path_Lung_Train,mode=0)
print(X1.shape)  
checkpoint = ModelCheckpoint(check, monitor='val_loss', verbose=1, 
                             save_best_only=True, save_weights_only=False, mode='auto')

model =enhancement_model(name_model=name_model, input_shape=input_shape,output_shape=output_shape,loss_function='mse').model
history=model.fit(X1,Y1, epochs=20,batch_size=128, verbose=1, validation_split=0.1,callbacks=[checkpoint])

# X1 = get_files_and_resample_spect(sampling_rate_new, window_size,locH=path_Heart_Train,locN=path_Lung_Train,mode=0)
        
