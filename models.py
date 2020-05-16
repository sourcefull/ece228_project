from keras.models import Model, Sequential
from keras.layers import Conv2D, Dense, Activation, Flatten, BatchNormalization, Dropout
from keras import optimizers


import keras 



import tensorflow as tf 

def baseline_model(height, width, channels):
    model = Sequential()

   # model.add((BatchNormalization(epsilon=0.001, axis=1, input_shape=(height, width, channels))))

    model.add(Conv2D(32, (3,3), strides=(2,2), activation='relu'   , input_shape=(height, width, channels)))
    #model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), strides=(2,2), activation='relu'))
    #model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), strides=(2,2), activation='relu'))
    #model.add(BatchNormalization())   
    #model.add(Dropout(0.15))   
    #model.add(Conv2D(64, (3,3), strides=(1,1), activation='elu'))
    #model.add(BatchNormalization())   
    dropout_rate =    0.7        

    model.add(Flatten())
   

    model.add(Dropout(dropout_rate))   

    model.add(Dense(512, activation='relu'))
    #model.add(Dropout(dropout_rate ))

    #model.add(Dense(30, activation='relu'))
    #model.add(Dropout(dropout_rate-.3))

    model.add(Dense(10, activation='softmax'))
    #model.add(Dense(1, activation='linear'))

    model.compile(  loss='categorical_crossentropy' , optimizer=optimizers.Adam(lr=0.001 ), metrics = ['accuracy' , keras.metrics.TopKCategoricalAccuracy(  k=1, name="top_k_categorical_accuracy", dtype=None ) ]   )
    #model.compile(optimizer = optimizers.SGD(learning_rate=0.0005, momentum=0.0, nesterov=True), loss='mse')
    
    model.summary()

    return model
