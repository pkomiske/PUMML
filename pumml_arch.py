from keras.models import Sequential
from keras.layers import Conv2D, LocallyConnected2D, ZeroPadding2D
from keras.optimizers import Adam

def parg(arg):

    """ Useful function for making single elements into a one-element list,
    i.e. for easy iteration. """

    if type(arg) != list:
        return [arg]
    else:
        return arg 


# function taking a hyperparameter dictionary (some examples are below)
def pileup_model(hps):

    img_size     =  hps['img_size']
    nb_channels  =  hps['nb_channels']
    
    filter_size  =  parg(hps['filter_size'])
    nb_filters   =  parg(hps['nb_filters'])
    stride       =  parg(hps['stride'])
    layers       =  parg(hps['layers'])
    zero_pad     =  parg(hps.setdefault('zero_pad', stride[0]))
    lr           =  hps.setdefault('lr', .001)
    summary      =  hps.setdefault('summary', True)

    proj_layer   =  hps.setdefault('proj_layer', Conv2D)
 
    model = Sequential()
    
    for i in range(len(layers)):
        if i == 0:
            model.add(ZeroPadding2D(padding=(zero_pad[i], zero_pad[i]), input_shape=(nb_channels, img_size, img_size)))
        else:
            model.add(ZeroPadding2D(padding=(zero_pad[i], zero_pad[i])))
        model.add(layers[i](nb_filters[i], filter_size[i], activation='relu', strides=stride[i], kernel_initializer='he_uniform', padding='valid'))

    if nb_filters[len(layers)-1] > 1:
        model.add(proj_layer(1, 1, kernel_initializer='he_uniform', padding='valid', activation='relu'))

    model.compile(loss=hps['loss'], optimizer=Adam(lr=lr))
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
