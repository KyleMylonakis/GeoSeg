from keras.layers import  Conv2D, Concatenate, Dropout, BatchNormalization
from keras.layers import Conv2DTranspose, Add
from keras.layers import Activation

def conv_layer(inputs,
            activation = 'relu',
            filters = 4,
            kernel_size = 3,
            dropout = 0.5,
            bottleneck = True,
            name = 'conv'):

    out = inputs 

    if bottleneck:
        out = Conv2D(filters=filters*4,
                    kernel_size=(1,3),
                    strides = (1,1),
                    activation = activation,
                    padding='same',
                    name = name+'/bottle_neck')(out)
        out = Dropout(dropout)(out)
        
    out = Conv2D(filters=filters,
                    kernel_size=(kernel_size,3),
                    strides = (1,1),
                    activation = activation,
                    padding='same',
                    name = name+'/conv')(out)
    out = Dropout(dropout)(out)
    
    return out

def bn_conv_layer(inputs,
            activation = 'relu',
            filters = 4,
            kernel_size = 3,
            dropout = 0.5,
            bottleneck = True,
            name = 'conv'):

    out = inputs 

    out = BatchNormalization()(out)
    out = Activation(relu)(out)

    if bottleneck:
        out = Conv2D(filters=num_filters*4,
                    kernel_size=(1,3),
                    strides = (1,1),
                    activation = None,
                    padding='same',
                    name = name+'/bottle_neck')(out)
        
        out = Dropout(dropout)(out)
        out = BatchNormalization()(out)
        out = Activation(activation)(out)

    out = Conv2D(filters=num_filters,
                    kernel_size=(kernel_size,3),
                    strides = (1,1),
                    activation = None,
                    padding='same',
                    name = name+'/conv')(out)
    out = Dropout(dropout)(out)
    
    return out

def residual_layer(inputs,
            activation = 'relu',
            filters = 4,
            kernel_size = 3,
            dropout = 0.5,
            bottleneck = True,
            name = 'residual'):
    
    out = conv_layer(inputs,
            activation = 'relu',
            filters = filters,
            kernel_size = 3,
            dropout = 0.5,
            bottleneck = True,
            name = name+'conv')
    
    out = Add()([out,inputs])

    return out 

def bn_residual_layer(inputs,
            activation = 'relu',
            filters = 4,
            kernel_size = 3,
            dropout = 0.5,
            bottleneck = True,
            name = 'conv'):
    
    out = bn_conv_layer(inputs,
            activation = 'relu',
            filters = filters,
            kernel_size = 3,
            dropout = 0.5,
            bottleneck = True,
            name = 'conv')
    
    out = Add()([out,inputs])

    return out


def transition_layer(inputs,filters,
                    up_or_down = 'down',
                    compression = 2,
                    kernel_size = 2,
                    activation = 'relu',
                    name = 'transition'):
    name = name+'/transistion'+'_'+up_or_down
    if up_or_down == 'down':
        filter_factor = 1/float(compression)
    elif up_or_down == 'up':
        filter_factor = compression
    else:
        msg = 'Expected up_or_down to be either "up" or "down"but instead got {}'
        raise ValueError(msg.format(up_or_down))
    
    #filters = int(filters*filter_factor)
    #print('transition filters:', filters)
    out = Conv2D(filters=filters,
                kernel_size=(1,3),
                strides = (1,1),
                activation = activation,
                padding='same',
                name = name+'/conv')(inputs)

    if up_or_down == 'down':
        out = Conv2D(filters=filters,
                kernel_size=(kernel_size,3),
                strides = (compression,1),
                activation = activation,
                padding='same',
                name = name+'/conv_dwn')(out)
    elif up_or_down == 'up':
        out = out = Conv2DTranspose(filters=filters,
                kernel_size=(kernel_size,3),
                strides = (compression,1),
                activation =activation,
                padding='same',
                name = name+'/conv_dwn')(out)
    return out 

