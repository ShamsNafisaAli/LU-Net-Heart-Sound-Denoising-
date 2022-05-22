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
        elif self.model_name=="cunet":
            self.convulation_unet_model()
        elif self.model_name=="unet":
            self.base_unet_model()
        elif self.model_name=='unet2d':
            self.unet2D()
        elif self.model_name=='fcn':
            self.fcn_dae()

            
            
            """
            The unet2D module is constructed based on the following work:
            Jansson, Andreas, Eric J. Humphrey, Nicola Montecchio, Rachel M. Bittner, Aparna Kumar and Tillman Weyde.
            Singing Voice Separation with Deep U-Net Convolutional Networks. ISMIR (2017).
            [https://ejhumphrey.com/assets/pdf/jansson2017singing.pdf]
            """
            
    def unet2D(self):
        print('Unet 2D')
        inputs=Input(batch_shape=(None,512, 128,1))
        conv1 = Conv2D(16, 5, strides=2, padding='same')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = LeakyReLU(alpha=0.2)(conv1)

        conv2 = Conv2D(32, 5, strides=2, padding='same')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = LeakyReLU(alpha=0.2)(conv2)

        conv3 = Conv2D(64, 5, strides=2, padding='same')(conv2)
        conv3 = BatchNormalization()(conv3)
        conv3 = LeakyReLU(alpha=0.2)(conv3)

        conv4 = Conv2D(128, 5, strides=2, padding='same')(conv3)
        conv4 = BatchNormalization()(conv4)
        conv4 = LeakyReLU(alpha=0.2)(conv4)

        conv5 = Conv2D(256, 5, strides=2, padding='same')(conv4)
        conv5 = BatchNormalization()(conv5)
        conv5 = LeakyReLU(alpha=0.2)(conv5)

        conv6 = Conv2D(512, 5, strides=2, padding='same')(conv5)
        conv6 = BatchNormalization()(conv6)
        conv6 = LeakyReLU(alpha=0.2)(conv6)

        deconv7 = Conv2DTranspose(256, 5, strides=2, padding='same')(conv6)
        deconv7 = BatchNormalization()(deconv7)
        deconv7 = Dropout(0.5)(deconv7)
        deconv7 = Activation('relu')(deconv7)

        deconv8 = Concatenate(axis=3)([deconv7, conv5])
        deconv8 = Conv2DTranspose(128, 5, strides=2, padding='same')(deconv8)
        deconv8 = BatchNormalization()(deconv8)
        deconv8 = Dropout(0.5)(deconv8)
        deconv8 = Activation('relu')(deconv8)

        deconv9 = Concatenate(axis=3)([deconv8, conv4])
        deconv9 = Conv2DTranspose(64, 5, strides=2, padding='same')(deconv9)
        deconv9 = BatchNormalization()(deconv9)
        deconv9 = Dropout(0.5)(deconv9)
        deconv9 = Activation('relu')(deconv9)

        deconv10 = Concatenate(axis=3)([deconv9, conv3])
        deconv10 = Conv2DTranspose(32, 5, strides=2, padding='same')(deconv10)
        deconv10 = BatchNormalization()(deconv10)
        deconv10 = Activation('relu')(deconv10)

        deconv11 = Concatenate(axis=3)([deconv10, conv2])
        deconv11 = Conv2DTranspose(16, 5, strides=2, padding='same')(deconv11)
        deconv11 = BatchNormalization()(deconv11)
        deconv11 = Activation('relu')(deconv11)

        deconv12 = Concatenate(axis=3)([deconv11, conv1])
        deconv12 = Conv2DTranspose(1, 5, strides=2, padding='same')(deconv12)
        deconv12 = Activation('sigmoid')(deconv12)
        self.model = Model(inputs=inputs, outputs=deconv12)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=self.loss)
        self.model.summary()
        
        
    def lstm_unet_model(self):

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
        
        
    def base_unet_model(self):

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

        #d16=Dropout(0.2)(d16)

        final = Conv1D(1,31,strides=1, activation='tanh', padding='same')(d14)




        self.model= Model(input_sig,final)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=self.loss)
        self.model.summary()

 
    def fcn_dae(self):
        input_sig = Input(batch_shape=(None,800,1)) 
        x1 = Conv1D(40, 16, strides=2, activation='elu', padding='same')(input_sig)
        #x1 = BatchNormalization()(x1)
        #x1 = Dropout(0.5)(x1)

        x2 = Conv1D(20, 16, strides=2, activation='elu', padding='same')(x1)
        #x2 = BatchNormalization()(x2)
        #x2 = Dropout(0.5)(x2)

        x3 = Conv1D(20, 16, strides=2, activation='elu', padding='same')(x2)
        #x3 = BatchNormalization()(x3)
        #x3 = Dropout(0.5)(x3)

        x4 = Conv1D(20, 16, strides=2, activation='elu', padding='same')(x3)
        #x4 = BatchNormalization()(x4)
        #x4 = Dropout(0.5)(x4)

        x5 = Conv1D(40, 16, strides=2, activation='elu', padding='same')(x4)
        #x5 = BatchNormalization()(x5)
        #x5 = Dropout(0.5)(x5)

        x6 = Conv1D(1, 16, strides=1, activation='elu', padding='same')(x5)
        #x6 = BatchNormalization()(x6)
        x6 = Dropout(0.5)(x6)
        #x6 = UpSampling1D(2)(x6)

        d6 = Conv1D(1,16,strides=1, activation='elu', padding='same')(x6)
        #d6 = BatchNormalization()(d6)
        #d6 = Dropout(0.5)(d6)
        d6 = UpSampling1D(2)(d6)

        d5 = Conv1D(40,16,strides=2, activation='elu', padding='same')(d6)
        #d5 = BatchNormalization()(d5)
        #d5 = Dropout(0.5)(d5)
        d5 = UpSampling1D(2)(d5)

        d4 = Conv1D(20,16,strides=1, activation='elu', padding='same')(d5)
        #d4 = BatchNormalization()(d4)
        #d4 = Dropout(0.5)(d4)
        d4 = UpSampling1D(2)(d4)

        d3 = Conv1D(20,16,strides=1, activation='elu', padding='same')(d4)
        #d3 = BatchNormalization()(d3)
        #d3 = Dropout(0.5)(d3)
        d3 = UpSampling1D(2)(d3)

        d2 = Conv1D(20,16,strides=1, activation='elu', padding='same')(d3)
        #d2 = BatchNormalization()(d2)
        #d2 = Dropout(0.5)(d2)
        d2 = UpSampling1D(2)(d2)

        d1 = Conv1D(40,16,strides=1, activation='elu', padding='same')(d2)
        #d1 = BatchNormalization()(d1)
        #d1 = Dropout(0.5)(d1)
        d1 = UpSampling1D(2)(d1)

        final = Conv1D(1, 16, strides=1, padding='same')(d1)
        
        self.model= Model(input_sig,final)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=self.loss)
        self.model.summary()


    def skip1(self,x):
        #x = tf.keras.activations.relu(x)
        
        x = Conv1D(x.shape[-1], 31, activation='relu', padding='same')(x)
        #x = Add()([x, y])
        #return BatchNormalization()(x)
        return x
    def convulation_unet_model(self):
        input_sig = Input(batch_shape=(None,self.input, 1)) #1600
        x = Conv1D(16,31, strides=1,activation='relu', padding='same')(input_sig)
        x = BatchNormalization()(x)
        x1 = Conv1D(32,31, strides=2,activation='relu', padding='same')(x)
        x1 = BatchNormalization()(x1)
        x2 = Conv1D(32,31, strides=2,activation='relu', padding='same')(x1)
        x2 = BatchNormalization()(x2)

        x3 = Conv1D(64,31,strides=2, activation='relu', padding='same')(x2)
        x3 = BatchNormalization()(x3)
        #8
        x4 = Conv1D(64,31,strides=2, activation='relu', padding='same')(x3)
        x4 = BatchNormalization()(x4)
        #4
        x5 = Conv1D(128,31,strides=2, activation='relu', padding='same')(x4)
        x5 = BatchNormalization()(x5)
            
        s1 = self.skip1(x)
        s2 = self.skip1(x1)
        s3 = self.skip1(x2)
        s4 = self.skip1(x3)
        s5 = self.skip1(x4)
        s6 = self.skip1(x5)
            
        d5 = Conv1D(64,31,strides=1, activation='relu', padding='same')(x5)
        d5 = BatchNormalization()(d5)
        d5 = UpSampling1D(2)(d5)
        d6 = Concatenate(axis=2)([s5,d5])

        d7 = Conv1D(64,31,strides=1, activation='relu', padding='same')(d6)
        d7 = BatchNormalization()(d7)
        d7 = UpSampling1D(2)(d7)
        d8 = Concatenate(axis=2)([s4,d7])

        d9 = Conv1D(32,31,strides=1, activation='relu', padding='same')(d8)
        d9 = BatchNormalization()(d9)
        d9 = UpSampling1D(2)(d9)
        d10 = Concatenate(axis=2)([s3,d9])

        d11 = Conv1D(32,31,strides=1, activation='relu', padding='same')(d10)
        d11 = BatchNormalization()(d11)
        d11 = UpSampling1D(2)(d11)
        d12 = Concatenate(axis=2)([s2,d11])

        d13 = Conv1D(32,31,strides=1, activation='relu', padding='same')(d12)
        d13 = BatchNormalization()(d13)
        d13 = UpSampling1D(2)(d13)
        d14 = Concatenate(axis=2)([s1,d13])

        final = Conv1D(1,31,strides=1, activation='tanh', padding='same')(d14)



        self.model= Model(input_sig,final)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=self.loss)
        self.model.summary()
