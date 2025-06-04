from pathlib import Path

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Flatten, GRU, TimeDistributed, RepeatVector, Layer, Input
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Layer
# from numpy.lib.financial import rate

# from keras import optimizers
# from keras_tuner import RandomSearch, BayesianOptimization 

class Custom_Loss_Prices(tf.keras.losses.Loss):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.prices = 7*[1]+5*[5]+12*[1] # 24 hourly "prices" [.1,.1,.1,.1... .3,.3,.3,.3,... 1,1,1] / 20
        self.prices = [self.prices[i//4] for i in range(4*len(self.prices))] # 96 hourly prices
        self.prices_tf = tf.constant(self.prices,dtype=tf.float32)
    def call(self, y_true, y_pred):        
        elements = tf.multiply(x=self.prices_tf, y=tf.abs(y_true - y_pred))
        #elements = y_true - y_pred
        return tf.reduce_mean(tf.square(elements)) 


class attention(Layer):
    def _init_(self, **kwargs):
        super(attention, self)._init_(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1),
                                 initializer='zeros', trainable=True)
        super(attention, self).build(input_shape)

    def call(self, x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x, self.W) + self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context


def create_model_attention(model_opt, train_X, train_y):
    input_train = Input(shape=(train_X.shape[1], train_X.shape[2]))
    output_train = Input(shape=(train_y.shape[1], train_y.shape[2]))
    ENC_layer, encoder_last_h, encoder_last_c = LSTM(model_opt['LSTM_num_hidden_units'], return_sequences=True, return_state=True)(input_train)
    attention_layer = attention()(ENC_layer)
    decoder = RepeatVector(output_train.shape[1])(attention_layer)
    decoder = LSTM(model_opt['LSTM_num_hidden_units'], return_state=False,
                   return_sequences=True)(decoder, initial_state=[encoder_last_h, encoder_last_c])
    outputs = TimeDistributed(Dense(output_train.shape[2]))(decoder)     #Dense(dense_units, trainable=True, activation=activation)(decoder)
    model = Model(inputs=input_train, outputs=outputs)
    model.compile(loss=model_opt["metrics"],
                  #loss=Custom_Loss_Prices(),
                  optimizer=model_opt["optimizer"])

    # Create early stopping function
    erlstp_callback = callbacks.EarlyStopping(monitor="val_loss",  # loss o val_loss
                                              patience=model_opt["patience"],
                                              mode="min",
                                              restore_best_weights=True,
                                              verbose=1)
    # Create a callback that saves the model's weights
    ckpt_callback = callbacks.ModelCheckpoint(str(model_opt["model_path"] / Path('model.keras')),
                                              save_best_only=True,
                                              save_weights_only=False,
                                              monitor='loss',
                                              mode='min')
    # Callback stop on NaN
    nan_callback = callbacks.TerminateOnNaN()

    cb_list = [erlstp_callback, nan_callback, ckpt_callback]
    model.summary()
    # fit network
    # history = model.fit(train_X, train_y, epochs=model_opt['epochs'], callbacks=cb_list,
    #                     validation_split=model_opt['validation_split'])
    # history = 1

    history = model.fit(train_X, train_y, epochs=model_opt['epochs'], callbacks=cb_list, validation_split=model_opt['validation_split'])
    return model, history


def create_model(model_opt, X, y):

    # tf.debugging.set_log_device_placement(True)
    with tf.device("/gpu:0"):
    #not_use_gpu = True
    #if not_use_gpu:
        model = Sequential()
        model.add(Dense(model_opt["Dense_input_dim"], input_shape=model_opt["input_dim"]))
        # model.add(Dropout(rate=model_opt["Dropout_rate"]))
        # model.add(LSTM(model_opt["LSTM_num_hidden_units"][0], return_sequences=True))
        # model.add(Dropout(rate=model_opt["Dropout_rate"]))
        # model.add(LSTM(model_opt["LSTM_num_hidden_units"][1], return_sequences=True))
        # # model.add(LSTM(model_opt["LSTM_num_hidden_units"][2], return_sequences=True))
        # model.add(LSTM(model_opt["LSTM_num_hidden_units"][3]))
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Dense(model_opt["dense_out"]))
        # model.add(Dense(1, activation=model_opt["neurons_activation"]))
        # tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        # compile the keras model
        model.compile(loss=model_opt["metrics"],
                      optimizer=model_opt["optimizer"])

        # Create early stopping function
        erlstp_callback = callbacks.EarlyStopping(monitor="val_loss",  # loss o val_loss
                                                  patience=model_opt["patience"],
                                                  mode="min",
                                                  restore_best_weights=True,
                                                  verbose=1)
        # Create a callback that saves the model's weights
        ckpt_callback = callbacks.ModelCheckpoint(model_opt["model_path"] / Path('model.keras'),
                                                  save_best_only=True,
                                                  save_weights_only=False,
                                                  monitor='loss',
                                                  mode='min')
        # Callback stop on NaN
        nan_callback = callbacks.TerminateOnNaN()

        cb_list = [erlstp_callback,  nan_callback, ckpt_callback]
        model.summary()
        # fit network
        history = model.fit(X, y, epochs=model_opt['epochs'], callbacks=cb_list, validation_split=model_opt['validation_split'])
        # history = 1
        # model.fit(X, y, epochs=model_opt['epochs'], callbacks=cb_list, validation_split=model_opt['validation_split'])

    return model, history



if __name__ == '__main__':
    output_dim = 2
    features = 1
    t_s = 15
    model_opt = {'num_hidden_units_1': output_dim,
                 'input_dim': (3, features),
                 'neurons_activation': 'relu',
                 'metrics': 'mse',
                 'optimizer': 'adam',
                 'patience': 10,
                 'model_path': '/'}
    model = create_model(model_opt)
    model.get_weights()
    # X = np.array([[0,0,1,2],
    # [0,0,1,2],
    # [0,0,1,2]])
    # X = np.array([[0,1],
    # [1,0],
    # [0,2]])
    # data = X.reshape((1, t_s, features))
    print('Computed variables: 4 x out_dim x (features + out_dim + 1) = {:d}'.format(
        4 * output_dim * (features + output_dim + 1)))
    # model.predict(data)

    a = 1
