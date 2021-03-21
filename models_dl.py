import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv1D, AveragePooling1D, AveragePooling2D, Dense, Flatten, Dropout, BatchNormalization, MaxPool1D, Activation, concatenate, GlobalAveragePooling1D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import keras
from tensorflow.keras.models import load_model
from keras.regularizers import l2
tf.config.run_functions_eagerly(True)
import tensorflow_addons as tfa




def ANN(shape=(55,300), layer_1=256, layer_2=256, layer_3=128, dropout=0.2,
       layer_4=16, layer_5=128, layer_6=64, activation_1='elu', activation_2='relu',
       optimizer='nadam', loss='mse'):
    
    # This returns a tensor
    inputs = Input(shape=shape)
    input_head = Input(shape=(6,))

    x = Dense(layer_1)(inputs)
    x = Activation(activation_1)(x) #  tfa.activations.mish
    
    x = Dense(layer_2)(x)
    x = Activation(activation_1)(x)
    
    x = Dense(layer_3)(x)
    x = BatchNormalization()(x)
    x = Activation(activation_1)(x)

    x = Dropout(dropout)(x)
    x = Flatten()(x)

    # a layer instance is callable on a tensor, and returns a tensor
    x2 = Dense(layer_4)(input_head) 
    #x2 = BatchNormalization()(x2)
    x2 = Activation(activation_1)(x2)

    x = concatenate([x,x2])
    x = Dense(layer_5)(x)
    x = BatchNormalization()(x)
    x = Activation(activation_1)(x)
    x = Dropout(dropout)(x)

    x = Dense(layer_6, activation=activation_2)(x)
    #x = Dense(128, activation='relu')(x)
    predictions = Dense(55, activation='linear')(x)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=[inputs,input_head], outputs=predictions)

    return model


def cnn_1D(shape=(55,300), layer_1=256, layer_2=256, layer_3=128, dropout=0.2,
       layer_4=16, layer_5=128, layer_6=64, activation_1='elu', activation_2='relu',
       optimizer='nadam', loss='mse'):
    
    # This returns a tensor
    inputs = Input(shape=shape)
    input_head = Input(shape=(6,))

    x = Conv1D(layer_1)(inputs)
    x = AveragePooling1D()(x)
    #x = GlobalAveragePooling1D()(x)
    x = Activation(activation_1)(x) #  tfa.activations.mish


    
    x = Conv1D(layer_2)(x)
    x = AveragePooling1D()(x)
    #x = GlobalAveragePooling1D()(x)
    x = Activation(activation_1)(x)

    x = Conv1D(layer_3)(x)
    x = AveragePooling1D()(x)
    #x = GlobalAveragePooling1D()(x)
    x = BatchNormalization()(x)
    x = Activation(activation_1)(x)

    x = Dropout(dropout)(x)
    x = Flatten()(x)

    # a layer instance is callable on a tensor, and returns a tensor
    x2 = Conv1D(layer_4)(input_head) 
    #x2 = BatchNormalization()(x2)
    x2 = Activation(activation_1)(x2)

    x = concatenate([x,x2])
    x = Dense(layer_5)(x)
    x = BatchNormalization()(x)
    x = Activation(activation_1)(x)
    x = Dropout(dropout)(x)

    x = Dense(layer_6, activation=activation_2)(x)
    #x = Dense(128, activation='relu')(x)
    predictions = Dense(55, activation='linear')(x)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=[inputs,input_head], outputs=predictions)

    
    return model


def gru(shape=(55,300), layer_1=64, layer_2=32, layer_3=16, dropout=0.2,
       layer_4=16, layer_5=128, layer_6=64, activation_1='elu', activation_2='relu'):
    
    # This returns a tensor
    inputs = Input(shape=shape)
    input_head = Input(shape=(6,))

    x = GRU(layer_1, return_sequences=True)(inputs)
    #x = AveragePooling1D()(x)
    #x = GlobalAveragePooling1D()(x)
    #x = MaxPooling1D(pool_size=4)(x)
    x = Activation(activation_1)(x) #  tfa.activations.mish

    x = GRU(layer_2, return_sequences=True)(x)
    #x = AveragePooling1D()(x)
    #x = GlobalAveragePooling1D()(x)
    #x = MaxPooling1D(pool_size=4)(x)
    x = Activation(activation_1)(x)
    
    x = GRU(layer_3, return_sequences=True)(x)
    #x = AveragePooling1D()(x)
    #x = GlobalMaxPooling1D()(x)
    #x = MaxPooling1D()(x)
    x = BatchNormalization()(x)
    x = Activation(activation_1)(x)

    x = Dropout(dropout)(x)
    x = Flatten()(x)

    # a layer instance is callable on a tensor, and returns a tensor
    x2 = Dense(layer_4)(input_head) 
    x2 = BatchNormalization()(x2)
    x2 = Activation(activation_1)(x2)

    x = concatenate([x,x2])
    x = Dense(layer_5)(x)
    x = BatchNormalization()(x)
    x = Activation(activation_1)(x)
    x = Dropout(dropout)(x)

    x = Dense(layer_6, activation=activation_2)(x)
    #x = Dense(128, activation='relu')(x)
    predictions = Dense(55, activation='linear')(x)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=[inputs,input_head], outputs=predictions)

    return model

def lstm(shape=(55,300), layer_1=256, layer_2=128, layer_3=64, dropout=0.2,
       layer_4=16, layer_5=128, layer_6=64, activation_1='elu', activation_2='relu',
       optimizer='nadam', loss='mse'):
    
    # This returns a tensor
    inputs = Input(shape=shape)
    input_head = Input(shape=(6,))

    x = LSTM(layer_1, return_sequences=True)(inputs)
    x = AveragePooling1D()(x)
    #x = GlobalAveragePooling1D()(x)
    #x = MaxPooling1D(pool_size=4)(x)
    x = Activation(activation_1)(x) #  tfa.activations.mish

    x = LSTM(layer_2, return_sequences=True)(x)
    x = AveragePooling1D()(x)
    #x = GlobalAveragePooling1D()(x)
    #x = MaxPooling1D(pool_size=4)(x)
    x = Activation(activation_1)(x)
    
    x = LSTM(layer_3, return_sequences=True)(x)
    #x = AveragePooling1D()(x)
    x = GlobalMaxPooling1D()(x)
    #x = MaxPooling1D()(x)
    x = BatchNormalization()(x)
    x = Activation(activation_1)(x)

    x = Dropout(dropout)(x)
    x = Flatten()(x)

    # a layer instance is callable on a tensor, and returns a tensor
    x2 = Dense(layer_4)(input_head) 
    x2 = BatchNormalization()(x2)
    x2 = Activation(activation_1)(x2)

    x = concatenate([x,x2])
    x = Dense(layer_5)(x)
    x = BatchNormalization()(x)
    x = Activation(activation_1)(x)
    x = Dropout(dropout)(x)

    x = Dense(layer_6, activation=activation_2)(x)
    #x = Dense(128, activation='relu')(x)
    predictions = Dense(55, activation='linear')(x)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=[inputs,input_head], outputs=predictions)

    #model.compile(optimizer=optimizer,
                  #loss=loss,
                  #)
    return model



def bi_lstm(shape=(55,300), layer_1=32, layer_2=16, layer_3=8, dropout=0.2,
       layer_4=16, layer_5=128, layer_6=64, activation_1='elu', activation_2='relu',
       optimizer='nadam', loss='mse'):
    
    # This returns a tensor
    inputs = Input(shape=shape)
    input_head = Input(shape=(6,))

    x = Bidirectional(LSTM(layer_1, return_sequences=True))(inputs)
    #x = AveragePooling1D()(x)
    #x = GlobalAveragePooling1D()(x)
    #x = MaxPooling1D(pool_size=4)(x)
    x = Activation(activation_1)(x) #  tfa.activations.mish

    x = Bidirectional(LSTM(layer_2, return_sequences=True))(x)
    #x = AveragePooling1D()(x)
    #x = GlobalAveragePooling1D()(x)
    #x = MaxPooling1D(pool_size=4)(x)
    x = Activation(activation_1)(x)
    
    x = Bidirectional(LSTM(layer_3, return_sequences=True))(x)
    #x = AveragePooling1D()(x)
    #x = GlobalMaxPooling1D()(x)
    #x = MaxPooling1D()(x)
    x = BatchNormalization()(x)
    x = Activation(activation_1)(x)

    x = Dropout(dropout)(x)
    x = Flatten()(x)

    # a layer instance is callable on a tensor, and returns a tensor
    x2 = Dense(layer_4)(input_head) 
    x2 = BatchNormalization()(x2)
    x2 = Activation(activation_1)(x2)

    x = concatenate([x,x2])
    x = Dense(layer_5)(x)
    x = BatchNormalization()(x)
    x = Activation(activation_1)(x)
    x = Dropout(dropout)(x)

    x = Dense(layer_6, activation=activation_2)(x)
    #x = Dense(128, activation='relu')(x)
    predictions = Dense(55, activation='linear')(x)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=[inputs,input_head], outputs=predictions)

  
    return model



def attention(shape=(55,300), layer_1=64, layer_2=32, layer_3=16, dropout=0.2,
       layer_4=16, layer_5=128, layer_6=64, activation_1='elu', activation_2='relu'):
    
    # This returns a tensor
    inputs = Input(shape=shape)
    input_head = Input(shape=(6,))

    x, state = GRU(layer_1, return_sequences=True, return_state=True)(inputs)
    #x = AveragePooling1D()(x)
    #x = GlobalAveragePooling1D()(x)
    #x = MaxPooling1D(pool_size=4)(x)
    x = Activation(activation_1)(x) #  tfa.activations.mish

    x = Attention(layer_2)([x, state])
    #x = AveragePooling1D()(x)
    #x = GlobalAveragePooling1D()(x)
    #x = MaxPooling1D(pool_size=4)(x)
    x = Activation(activation_1)(x)
    
    x = GRU(layer_3, return_sequences=True)(x)
    #x = AveragePooling1D()(x)
    #x = GlobalMaxPooling1D()(x)
    #x = MaxPooling1D()(x)
    #x = BatchNormalization()(x)
    #x = Activation(activation_1)(x)

    x = Dropout(dropout)(x)
    x = Flatten()(x)

    # a layer instance is callable on a tensor, and returns a tensor
    x2 = Dense(layer_4)(input_head) 
    x2 = BatchNormalization()(x2)
    x2 = Activation(activation_1)(x2)

    x = concatenate([x,x2])
    x = Dense(layer_5)(x)
    x = BatchNormalization()(x)
    x = Activation(activation_1)(x)
    x = Dropout(dropout)(x)

    x = Dense(layer_6, activation=activation_2)(x)
    #x = Dense(128, activation='relu')(x)
    predictions = Dense(55, activation='linear')(x)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=[inputs,input_head], outputs=predictions)

    return model



def attention2(shape=(55,300), layer_1=64, layer_2=32, layer_3=16, dropout=0.2,
       layer_4=16, layer_5=128, layer_6=64, activation_1='elu', activation_2='relu'):
    
    # This returns a tensor
    inputs = Input(shape=shape)
    input_head = Input(shape=(6,))

    x, state = GRU(layer_1, return_sequences=True, return_state=True)(inputs)
    #x = AveragePooling1D()(x)
    #x = GlobalAveragePooling1D()(x)
    #x = MaxPooling1D(pool_size=4)(x)
    x = Activation(activation_1)(x) #  tfa.activations.mish

    x = Attention(layer_2)([x, state])
    #x = AveragePooling1D()(x)
    #x = GlobalAveragePooling1D()(x)
    #x = MaxPooling1D(pool_size=4)(x)
    x = Activation(activation_1)(x)
    
    x, state = GRU(layer_3,  return_state=True, return_sequences=True)(x)
    x = Activation(activation_1)(x)
    
    x = Attention(layer_3)([x, state])
    
    #x = AveragePooling1D()(x)
    #x = GlobalMaxPooling1D()(x)
    #x = MaxPooling1D()(x)
    #x = BatchNormalization()(x)
    x = Activation(activation_1)(x)
    x= GRU(layer_3, return_sequences=True)(x)
    x = Activation(activation_1)(x)
    
    x = Dropout(dropout)(x)
    x = Flatten()(x)

    # a layer instance is callable on a tensor, and returns a tensor
    x2 = Dense(layer_4)(input_head) 
    x2 = BatchNormalization()(x2)
    x2 = Activation(activation_1)(x2)

    x = concatenate([x,x2])
    x = Dense(layer_5)(x)
    x = BatchNormalization()(x)
    x = Activation(activation_1)(x)
    x = Dropout(dropout)(x)

    x = Dense(layer_6, activation=activation_2)(x)
    #x = Dense(128, activation='relu')(x)
    predictions = Dense(55, activation='linear')(x)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=[inputs,input_head], outputs=predictions)

    return model