#
# Various network layers used in block construction.
# The structure is intended for use in Dense nets as in 
# but can be used in any network that supports block structure

# References:
#   1) Huang, Lui (2016), Densely Connected Convolutional Networks: https://arxiv.org/pdf/1608.06993.pdf
#   2) Ioffe, Sergey; Szegedy, Christian (2015): https://arxiv.org/abs/1502.03167
#   3) Dmytro Mishkin and Nikolay Sergievskiy and Jiri Matas (2017): http://www.sciencedirect.com/science/article/pii/S1077314217300814
from keras.layers import  Conv2D, Concatenate, Dropout, BatchNormalization
from keras.layers import Conv2DTranspose, Add
from keras.layers import Activation

def conv_layer(inputs,
            activation = 'relu',
            filters = 4,
            kernel_size = 3,
            dropout = 0.5,
            bottleneck = True,
            bottleneck_factor = 4,
            name = 'conv'):
    """
    Performs a 2D convolution on seismic data. Assumes 
    inputs has shape (N,3,r) corresponding an N-discretized
    seismograph signal for a 3d signal comming from r recievers.
    Has the option of adding a 1x1 convolutional bottleneck for 
    feature reduction and computational efficiency as in [1].
    
    Parameters:
    -----------
        inputs: A tensor of size (N,3,r).
        activation: Name of a keras activation function. (str)
        filters: Number of output filters. Can be lowered if bottlenecking.(int)
        kernel_size: The size of kernel to use for the convolution. The
                    2D convolution kernel will have kernel size: (kernel_size,3)
        dropout: The percent probability for each individual layer's dropout
                    during training. Must be in [0,1) (float).
        bottleneck: Whether to apply a 1x1 bottleneck layer before applying
                        the convolution layer. (bool)
        bottleneck_factor: Factor to multiply filters by in bottleneck layer.
                    If bottleneck is True the bottleneck layer will have 
                    filters*bottleneck_factor number of filters to feed
                    into the convolution layer. (int)
        name: A name for the operation. (str)
    Returns:
    --------
        A tensor of shape (N,3,filters)
    """
    out = inputs 
    # Check bottleneck
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
            bottleneck_factor=4,
            name = 'conv'):
    """
    A convolution with batch normalization done before
    all activations as suggested in [2] though in [3] performance
    gains were noticed when BN was done after activation.
    Parameters:
    -----------
        inputs: A tensor of size (N,3,r).
        activation: Name of a keras activation function. (str)
        filters: Number of output filters. Can be lowered if bottlenecking.(int)
        kernel_size: The size of kernel to use for the convolution. The
                    2D convolution kernel will have kernel size: (kernel_size,3)
        dropout: The percent probability for each individual layer's dropout
                    during training. Must be in [0,1) (float).
        bottleneck: Whether to apply a 1x1 bottleneck layer before applying
                        the convolution layer. (bool)
        bottleneck_factor: Factor to multiply filters by in bottleneck layer.
                    If bottleneck is True the bottleneck layer will have 
                    filters*bottleneck_factor number of filters to feed
                    into the convolution layer. (int)
        name: A name for the operation. (str)
    Returns:
    --------
        A tensor of shape (N,3,filters)
    """
    out = inputs 

    # Placing batch norm BEFORE activation
    # some results get improvements when placing
    # it after. 
    out = BatchNormalization()(out)
    out = Activation(relu)(out)

    if bottleneck:
        out = Conv2D(filters=num_filters*bottleneck_factor,
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
    """
    A residual version of the usual convolution blocks.
    
    Parameters:
    -----------
        inputs: A tensor of size (N,3,r).
        activation: Name of a keras activation function. (str)
        filters: Number of output filters. Can be lowered if bottlenecking.(int)
        kernel_size: The size of kernel to use for the convolution. The
                    2D convolution kernel will have kernel size: (kernel_size,3)
        dropout: The percent probability for each individual layer's dropout
                    during training. Must be in [0,1) (float).
        bottleneck: Whether to apply a 1x1 bottleneck layer before applying
                        the convolution layer. (bool)
        bottleneck_factor: Factor to multiply filters by in bottleneck layer.
                    If bottleneck is True the bottleneck layer will have 
                    filters*bottleneck_factor number of filters to feed
                    into the convolution layer. (int)
        name: A name for the operation. (str)
    Returns:
    --------
        A tensor of shape (N,3,filters)
    """
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
    """
    A residual layer with batch normalizations before
    each activation. 
    Parameters:
    -----------
        inputs: A tensor of size (N,3,r).
        activation: Name of a keras activation function. (str)
        filters: Number of output filters. Can be lowered if bottlenecking.(int)
        kernel_size: The size of kernel to use for the convolution. The
                    2D convolution kernel will have kernel size: (kernel_size,3)
        dropout: The percent probability for each individual layer's dropout
                    during training. Must be in [0,1) (float).
        bottleneck: Whether to apply a 1x1 bottleneck layer before applying
                        the convolution layer. (bool)
        bottleneck_factor: Factor to multiply filters by in bottleneck layer.
                    If bottleneck is True the bottleneck layer will have 
                    filters*bottleneck_factor number of filters to feed
                    into the convolution layer. (int)
        name: A name for the operation. (str)
    Returns:
    --------
        A tensor of shape (N,3,filters)
    """
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
                    compression = 2,
                    up_or_down = 'down',
                    kernel_size = 2,
                    activation = 'relu',
                    name = 'transition'):
    """
    A layer to either up or down sample the first dimension
    of the data. Down and Up are used in encoding and decoding
    branches, respectively, of architectures like UNets and 
    AutoEncodes.
    
    Parameters:
    -----------
        inputs: The input tensor of shape (N,3,f). 
        filters: Number of output filters for the layer. (int)
        compression: Factor to either up or down sample the feature 
            channels by. (int)
        up_or_down: One of {'up','down'}. If 'down' then the first
            dimension is decreased by a factor of 2 using a (2,1) strided
            2Dconvolution. If 'up' it is increased by a factor of 2
            using a (2,1) strided Convolution Transpose. (str)
        kernel_size: The size of first dimension of convolution kernel.
            Convolution will have size (kernel_size,3). (int)
        activation: Name of a keras activation function. (str) 
        name: A name for the operation. (str)      
    Returns:
    --------
        If 'down' returns a tensor of shape (N/2,3,filters), if
        'up' returns a tensor of shape (2*N,3,filters).
    """
    name = name+'/transistion'+'_'+up_or_down
    # up samples decrease features
    # down samples increase features.
    # TODO: Allow for up and down convolutions to have
    #       factors other than 2.
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

