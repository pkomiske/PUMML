##========================================================================##
##  File: pumml_arch.py                                                   ##
##                                                                        ##
##  Copyright (c) 2016-2018 Patrick Komiske, Eric Metodiev                ##
##                                                                        ##
##  This program is free software; you can redistribute it and/or modify  ##
##  it under the terms of the GNU General Public License as published by  ##
##  the Free Software Foundation; either version 3 of the License, or     ##
##  (at your option) any later version.                                   ##
##                                                                        ##
##  This program is distributed in the hope that it will be useful,       ##
##  but WITHOUT ANY WARRANTY; without even the implied warranty of        ##
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         ##
##  GNU General Public License for more details.                          ##
##                                                                        ##
##  You should have received a copy of the GNU General Public License     ##
##  along with this program; If not, see <http://www.gnu.org/licenses/>.  ##
##========================================================================##


from keras import backend as K
from keras.layers import Conv2D, LocallyConnected2D, ZeroPadding2D
from keras.models import Sequential
from keras.optimizers import Adam


def parg(arg):

    """Function for making single elements into a one-element list for iteration."""

    if type(arg) != list:
        return [arg]
    else:
        return arg 


# constant appearing in PUMML loss function 
pbar = 10.0

def pumml_loss(y_true, y_pred):

    """Custom per-pixel logarithmic squared loss function."""

    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + pbar)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + pbar)
    return K.mean(K.square(first_log - second_log), axis=-1)


def pileup_model(hps):

    """Function that takes a hyperparameter dictionary and returns a 
    compiled Keras model.
    """

    img_size     = hps['img_size']
    nb_channels  = hps['nb_channels']
    
    filter_size  = parg(hps['filter_size'])
    nb_filters   = parg(hps['nb_filters'])
    stride       = parg(hps['stride'])
    layers       = parg(hps['layers'])
    zero_pad     = parg(hps.get('zero_pad', stride[0]))
    lr           = hps.get('lr', .001)
    summary      = hps.get('summary', True)

    proj_layer   = hps.get('proj_layer', Conv2D)
    loss         = hps.get('loss', pumml_loss)
 
    model = Sequential()
    
    for i in range(len(layers)):
        opts = {'padding': (zero_pad[i], zero_pad[i])}
        if i == 0:
            opts['input_shape'] = (nb_channels, img_size, img_size)
        model.add(ZeroPadding2D(**opts))
        
        model.add(layers[i](nb_filters[i], filter_size[i], activation='relu', strides=stride[i], kernel_initializer='he_uniform', padding='valid'))

    if nb_filters[len(layers)-1] > 1:
        model.add(proj_layer(1, 1, kernel_initializer='he_uniform', padding='valid', activation='relu'))

    model.compile(loss=loss, optimizer=Adam(lr=lr))

    if summary:
        model.summary()

    return model

"""hps_examples = {

    'hps_1layer_conv': {
        'filter_size': 5,
        'img_size': 45,
        'nb_filters': 1,
        'stride': 5,
        'nb_channels': 3,
        'layers': [Convolution2D],
    },

    'hps_2layer_conv_large': {
        'filter_size': [8, 3],
        'img_size': 45,
        'nb_filters': [1, 1],
        'stride': [2,2],
        'nb_channels': 3,
        'layers': [Convolution2D, Convolution2D],
    },

    'hps_2layer_conv_med': {
        'filter_size': [4, 5],
        'img_size': 45,
        'nb_filters': [1, 1],
        'stride': [2,2],
        'nb_channels': 3,
        'layers': [Convolution2D, Convolution2D],
    },

    'hps_1layer_conv_multichan': {
        'filter_size': 5,
        'img_size': 45,
        'nb_filters': 2,
        'stride': 5,
        'nb_channels': 3,
        'layers': [Convolution2D]
    },

    'hps_2layer_conv_multichan': {
        'filter_size': [8, 3],
        'img_size': 45,
        'nb_filters': [2, 2],
        'stride': [2,2],
        'nb_channels': 3,
        'layers': [Convolution2D, Convolution2D]
    },

}"""
