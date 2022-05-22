"""
This File contains everything to train the LSTM UNet model for HS enhancement.
For running the training see "training.py".
To run evaluation with the provided pretrained model see "inference.py".
Author: 
    Shams Nafisa Ali (snafisa.bme.buet@gmail.com)
    Samiul Based Shuvo (sbshuvo.bme.buet@gmail.com)
Version: 24.05.2022
This code is licensed under the terms of the MIT-license.
"""



from tensorflow.keras.layers import Input, Conv2D,LeakyReLU,Dropout,Conv2DTranspose,Activation,Conv1D, Bidirectional, LSTM, UpSampling1D, Concatenate, BatchNormalization
from tensorflow.keras import Model
import tensorflow as tf
from tensorflow import keras

class enhancement_model:
    def __init__(self,name_model, input_shape, output_shape, loss_function):
        self.input=input_shape
        self.output=output_shape
        self.model_name=name_model
        self.model=None
        self.loss=loss_function

        if self.model_name=="lunet":
            self.lstm_unet_model()
        elif self.model_name=="unet":
            self.base_unet_model()
        elif self.model_name=='fcn':
            self.fcn_dae()
          
            

    def lstm_unet_model(self): #LU-Net

        input_sig = Input(batch_shape=(None,self.input,1))

        x = Conv1D(16,31, strides=1,activation='relu', padding='same')(input_sig)
        rnn_1 = Bidirectional(LSTM(8, kernel_initializer="he_normal", return_sequences=True))(x)
        x1 = Conv1D(32,31, strides=2,activation='relu', padding='same')(x)
        rnn_2 = Bidirectional(LSTM(16, kernel_initializer="he_normal", return_sequences=True))(x1)
        x2 = Conv1D(32,31, strides=2,activation='relu', padding='same')(x1)
        rnn_3 = Bidirectional(LSTM(16, kernel_initializer="he_normal", return_sequences=True))(x2)
        x3 = Conv1D(64,31,strides=2, activation='relu', padding='same')(x2)
        rnn_4 = Bidirectional(LSTM(32, kernel_initializer="he_normal", return_sequences=True))(x3)
        x4 = Conv1D(64,31,strides=2, activation='relu', padding='same')(x3)
        rnn_5 = Bidirectional(LSTM(32, kernel_initializer="he_normal", return_sequences=True))(x4)
        x5 = Conv1D(128,31,strides=2, activation='relu', padding='same')(x4)

        d5 = Conv1D(64,31,strides=1, activation='relu', padding='same')(x5)
        d5 = UpSampling1D(2)(d5)
        d6 = Concatenate(axis=2)([rnn_5,d5])

        d7 = Conv1D(64,31,strides=1, activation='relu', padding='same')(d6)
        d7 = UpSampling1D(2)(d7)
        d8 = Concatenate(axis=2)([rnn_4,d7])

        d9 = Conv1D(32,31,strides=1, activation='relu', padding='same')(d8)
        d9 = UpSampling1D(2)(d9)
        d10 = Concatenate(axis=2)([rnn_3 ,d9])

        d11 = Conv1D(32,31,strides=1, activation='relu', padding='same')(d10)
        d11 = UpSampling1D(2)(d11)
        d12 = Concatenate(axis=2)([rnn_2,d11])

        d13 = Conv1D(32,31,strides=1, activation='relu', padding='same')(d12)
        d13 = UpSampling1D(2)(d13)
        d14 = Concatenate(axis=2)([rnn_1,d13])
        final = Conv1D(1,31,strides=1, activation='tanh', padding='same')(d14)

        self.model= Model(input_sig,final)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=self.loss)
        self.model.summary()
        
        
    def base_unet_model(self): #U-Net (Baseline-2)

        input_sig = Input(batch_shape=(None,self.input,1)) #1600
        x = Conv1D(16,31, strides=1,activation='relu', padding='same')(input_sig)

        x1 = Conv1D(32,31, strides=2,activation='relu', padding='same')(x)
        #32
        x2 = Conv1D(32,31, strides=2,activation='relu', padding='same')(x1)
        #x2=Dropout(0.2)(x2)
        #16
        #x1 = MaxPooling1D(2)(x)
        x3 = Conv1D(64,31,strides=2, activation='relu', padding='same')(x2)
        #8
        x4 = Conv1D(64,31,strides=2, activation='relu', padding='same')(x3)
        #4
        x5 = Conv1D(128,31,strides=2, activation='relu', padding='same')(x4)
        #x5=Dropout(0.2)(x5)

        d5 = Conv1D(64,31,strides=1, activation='relu', padding='same')(x5)
        d5 = UpSampling1D(2)(d5)
        d6 = Concatenate(axis=2)([x4,d5])

        d7 = Conv1D(64,31,strides=1, activation='relu', padding='same')(d6)
        d7 = UpSampling1D(2)(d7)
        d8 = Concatenate(axis=2)([x3,d7])

        d9 = Conv1D(32,31,strides=1, activation='relu', padding='same')(d8)
        d9 = UpSampling1D(2)(d9)
        d10 = Concatenate(axis=2)([x2,d9])

        d11 = Conv1D(32,31,strides=1, activation='relu', padding='same')(d10)
        d11 = UpSampling1D(2)(d11)
        d12 = Concatenate(axis=2)([x1,d11])

        d13 = Conv1D(32,31,strides=1, activation='relu', padding='same')(d12)
        d13 = UpSampling1D(2)(d13)
        d14 = Concatenate(axis=2)([x,d13])

        final = Conv1D(1,31,strides=1, activation='tanh', padding='same')(d14)

        self.model= Model(input_sig,final)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=self.loss)
        self.model.summary()

 
    def fcn_dae(self):  #FCN (Baseline-1)
        input_sig = Input(batch_shape=(None,800,1)) 
        x1 = Conv1D(40, 16, strides=2, activation='elu', padding='same')(input_sig)
        #x1 = BatchNormalization()(x1)
        #x1 = Dropout(0.5)(x1)

        x2 = Conv1D(20, 16, strides=2, activation='elu', padding='same')(x1)
        x3 = Conv1D(20, 16, strides=2, activation='elu', padding='same')(x2)
        x4 = Conv1D(20, 16, strides=2, activation='elu', padding='same')(x3)
        x5 = Conv1D(40, 16, strides=2, activation='elu', padding='same')(x4)
        x6 = Conv1D(1, 16, strides=1, activation='elu', padding='same')(x5)
        x6 = Dropout(0.5)(x6)
        
        d6 = Conv1D(1,16,strides=1, activation='elu', padding='same')(x6)
        d6 = UpSampling1D(2)(d6)
        d5 = Conv1D(40,16,strides=2, activation='elu', padding='same')(d6)
        d5 = UpSampling1D(2)(d5)
        d4 = Conv1D(20,16,strides=1, activation='elu', padding='same')(d5)
        d4 = UpSampling1D(2)(d4)
        d3 = Conv1D(20,16,strides=1, activation='elu', padding='same')(d4)
        d3 = UpSampling1D(2)(d3)
        d2 = Conv1D(20,16,strides=1, activation='elu', padding='same')(d3)
        d2 = UpSampling1D(2)(d2)
        d1 = Conv1D(40,16,strides=1, activation='elu', padding='same')(d2)
        d1 = UpSampling1D(2)(d1)
        final = Conv1D(1, 16, strides=1, padding='same')(d1)
        
        self.model= Model(input_sig,final)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=self.loss)
        self.model.summary()
