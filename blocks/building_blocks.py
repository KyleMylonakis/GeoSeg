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
            kernel_size = (3,3),
            dropout = 0.5,
            bottleneck = True,
            bottleneck_factor = 4,
            name = 'conv'):
    """
    Performs a 2D convolution on seismic data. 
    Has the option of adding a 1x1 convolutional bottleneck for 
    feature reduction and computational efficiency as in [1].
    
    Parameters:
    -----------
        inputs: A tensor.
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
        A tensor of same shape as input.
    """
    out = inputs 
    # Check bottleneck
    if bottleneck:
        out = Conv2D(filters=filters*bottleneck_factor,
                    kernel_size=(1,1),
                    strides = (1,1),
                    activation = activation,
                    padding='same',
                    name = name+'/bottle_neck')(out)
        out = Dropout(dropout, name = name+'/bottle_neck/dropout')(out)
        
    out = Conv2D(filters=filters,
                    kernel_size=kernel_size,
                    strides = (1,1),
                    activation = activation,
                    padding='same',
                    name = name+'/conv')(out)
    out = Dropout(dropout, name = name +'/conv/dropout')(out)
    
    return out

def bn_conv_layer(inputs,
            activation = 'relu',
            filters = 4,
            kernel_size = (3,3),
            dropout = 0.5,
            bottleneck = True,
            bottleneck_factor=4,
            name = 'conv'):
    """
    A convolution with batch normalization (BN) done before
    all activations as suggested in [2] though in [3] performance
    gains were noticed when BN was done after activation.
    Parameters:
    -----------
        inputs: A tensor.
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
        A tensor of same shape as input.
    """
    out = inputs 

    # Placing batch norm BEFORE activation
    # some results get improvements when placing
    # it after. 
    out = BatchNormalization()(out)
    out = Activation(activation)(out)

    if bottleneck:
        out = Conv2D(filters=filters*bottleneck_factor,
                    kernel_size=(1,1),
                    strides = (1,1),
                    activation = None,
                    padding='same',
                    name = name+'/bottle_neck')(out)
        
        out = Dropout(dropout)(out)
        out = BatchNormalization()(out)
        out = Activation(activation)(out)

    out = Conv2D(filters=filters,
                    kernel_size=kernel_size,
                    strides = (1,1),
                    activation = None,
                    padding='same',
                    name = name+'/conv')(out)
    out = Dropout(dropout)(out)
    
    return out

def residual_layer(inputs,
            activation = 'relu',
            filters = 4,
            kernel_size = (3,3),
            dropout = 0.5,
            bottleneck = True,
            bottleneck_factor = 4,
            name = 'residual'):
    """
    A residual version of the usual convolution blocks.
    
    Parameters:
    -----------
        inputs: A rank 3 tensor.
        activation: Name of a keras activation function. (str)
        filters: Number of output filters. Can be lowered if bottlenecking.(int)
        kernel_size: The size of kernel to use for the convolution.
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
        A tensor of same shape as input.
    """
    out = conv_layer(inputs,
            activation = activation,
            filters = filters,
            kernel_size = kernel_size,
            dropout = dropout,
            bottleneck = bottleneck,
            bottleneck_factor=bottleneck_factor,
            name = name+'conv')
    
    out = Add()([out,inputs])

    return out 

def bn_residual_layer(inputs,
            activation = 'relu',
            filters = 4,
            kernel_size= (3,1),
            dropout = 0.5,
            bottleneck = True,
            bottleneck_factor = 4,
            name = 'residual'):
    """
    A residual layer with batch normalizations before
    each activation. (See bn_conv_layer for BN references)
    Parameters:
    -----------
        inputs: A rank 3 tensor.
        activation: Name of a keras activation function. (str)
        filters: Number of output filters. Can be lowered if bottlenecking.(int)
        kernel_size: The size of kernel to use for the convolution.
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
        A tensor of same shape as input
    """
    out = bn_conv_layer(inputs,
            activation = activation,
            filters = filters,
            kernel_size = kernel_size,
            dropout = dropout,
            bottleneck = bottleneck,
            bottleneck_factor=bottleneck_factor,
            name = name + 'conv')
    
    out = Add()([out,inputs])

    return out

# A one dimensional transition layer.
# ONLY DOWNSAMPLES TEMPORAL AXIS
def transition_layer(inputs,filters,
                    compression = 2,
                    up_or_down = 'down',
                    kernel_size= (2,1),
                    activation = 'relu',
                    padding = 'same',
                    name = 'transition'):
    """
    A layer to either up or down sample the FIRST DIMENSION
    of the data. Down and Up are used in encoding and decoding
    branches, respectively, of architectures like UNets and 
    SegNets.
    NOTE: This is a ONE DIMENSIONAL transition.
    
    Parameters:
    -----------
        inputs: The input rank 3 tensor of shape (N,r,f). 
        filters: Number of output filters for the layer. (int)
        compression: Factor to either up or down sample the feature 
            channels by. (int)
        up_or_down: One of {'up','down'}. If 'down' then the first
            dimension is decreased by a factor of compression using a (compression,1) strided
            2Dconvolution. If 'up' it is increased by a factor of compression using a 
            (compression,1) strided Convolution Transpose. (str)
        kernel_size: The size of first dimension of convolution kernel.
            Convolution will have size (kernel_size,3). (int)
        activation: Name of a keras activation function. (str) 
        name: A name for the operation. (str)      
    Returns:
    --------
        If 'down' returns a tensor of shape (N/compression,r,filters), if
        'up' returns a tensor of shape (compression*N,r,filters).
    """
    name = name+'/transistion'+'_'+up_or_down
    # up samples decrease features
    # down samples increase features.
    if up_or_down == 'down':
        filter_factor = 1/float(compression)
    elif up_or_down == 'up':
        filter_factor = compression
    else:
        msg = 'Expected up_or_down to be either "up" or "down"but instead got {}'
        raise ValueError(msg.format(up_or_down))
    
    out = Conv2D(filters=filters,
                kernel_size=(1,3), 
                strides = (1,1),
                activation = activation,
                padding=padding,
                name = name+'/conv')(inputs)

    if up_or_down == 'down':
        out = Conv2D(filters=filters,
                kernel_size=kernel_size,
                strides = (compression,1),
                activation = activation,
                padding=padding,
                name = name+'/conv_dwn')(out)
    
    elif up_or_down == 'up':
        out = out = Conv2DTranspose(filters=filters,
                kernel_size=kernel_size,
                strides = (compression,1),
                activation =activation,
                padding=padding,
                name = name+'/conv_dwn')(out)
    return out 

def transition_layer_2D(inputs,filters,
                    compression = 2,
                    up_or_down = 'down',
                    kernel_size= (2,2),
                    activation = 'relu',
                    padding = 'same',
                    name = 'transition'):
    """
    A layer to either up or down sample the input. Down and Up 
    are used in encoding and decoding branches, respectively, 
    of architectures like UNets and SegNets.
    
    Parameters:
    -----------
        inputs: The input rank 3 tensor of shape (N,r,f). 
        filters: Number of output filters for the layer. (int)
        compression: Factor to either up or down sample the feature 
            channels by. (int)
        up_or_down: One of {'up','down'}. If 'down' then the first
            dimension is decreased by a factor of compression using a (compression,1) strided
            2Dconvolution. If 'up' it is increased by a factor of compression using a 
            (compression,1) strided Convolution Transpose. (str)
        kernel_size: The size of first dimension of convolution kernel.
            Convolution will have size (kernel_size,3). (int)
        activation: Name of a keras activation function. (str) 
        name: A name for the operation. (str)      
    Returns:
    --------
        If 'down' returns a tensor of shape (N/compression, r/compression, filters), if
        'up' returns a tensor of shape (compression*N, compression*r, filters).
    """
    name = name+'/transistion'+'_'+up_or_down
    # up samples decrease features
    # down samples increase features.
    if up_or_down == 'down':
        filter_factor = 1/float(compression)
    elif up_or_down == 'up':
        filter_factor = compression
    else:
        msg = 'Expected up_or_down to be either "up" or "down"but instead got {}'
        raise ValueError(msg.format(up_or_down))
    
    #filters = int(filters*filter_factor)
    out = Conv2D(filters=filters,
                kernel_size=(1,3),
                strides = (1,1),
                activation = activation,
                padding=padding,
                name = name+'/conv')(inputs)

    if up_or_down == 'down':
        out = Conv2D(filters=filters,
                kernel_size=kernel_size,
                strides = (compression,compression),
                activation = activation,
                padding=padding,
                name = name+'/conv_dwn')(out)
    elif up_or_down == 'up':
        out = out = Conv2DTranspose(filters=filters,
                kernel_size=kernel_size,
                strides = (compression,compression),
                activation =activation,
                padding=padding,
                name = name+'/conv_dwn')(out)
    return out 

def binary_output_layer_1d(inputs,
                    num_receivers = 3, 
                    activation = 'softmax',
                    filters = 4,
                    dropout = 0.5,
                    name = 'softmax',
                    num_classes = 2):
    """
    *(Deprecated): To be replaced by multiclass_output_layer_1d. Added
        num_classes as dummy variable to keep it working with current
        code base but this may not work forever and we will stop supporting
        this function soon. *

    tensors = (t-direction, x-direction, filters)
    A final layer for 1d low velocity detection. Performs
    a downsampling in the x-direction with a (num_receivers,1)
    kernel strided at (1,num_receivers) then outputs class 
    probabilities at each pixels.
    
    (N,num_receivers,f) -> (N,1,num_receivers) -> (N,1,2)
                                    ^- This is to preserve total image parameters.
    First mapping is hardcoded relu. Act
    Parameters:
    -----------
        inputs: Tensor of size (N,num_receivers,f)
        activation: Activation to use in second layer. 
        num_receivers: size of x-direction.
        filters: (Deprecated) Number of filters to put before softmaxing.
        dropout: Dropout probability for first layer.
        name: Name to use for block.
    Returns:
    --------
        A (N,1,2) tensor where out[n,:,j] is the probability that pixes n is
        of class j. Needs to be reshaped before use with loss. 
    """
    # (N,r,s) -> (N,1,filters)
    # defaults to relu in the middle
    out = Conv2D(filters = num_receivers,
                kernel_size = (num_receivers,1),
                padding = "same",
                activation= 'relu',
                strides = (1,num_receivers))(inputs)

    out = Dropout(dropout)(out)

    # (N,1,filters) -> (N,1,2)
    out = Conv2D(filters = 2,
                kernel_size = (1,1),
                activation = activation,
                padding = "same")(out)
    return out

def multiclass_output_layer_1d(inputs,
                    num_classes = 2,
                    num_receivers = 3, 
                    activation = 'softmax',
                    filters = 4,
                    dropout = 0.5,
                    name = 'softmax'):
    """
    tensors = (t-direction, x-direction, filters)
    A final layer for 1d multi elocity detection. Performs
    a downsampling in the x-direction with a (num_receivers,1)
    kernel strided at (1,num_receivers) then outputs classes
    probabilities at each pixel.
    
    (N,num_receivers,f) -> (N,1,num_receivers) -> (N,1,num_classes)

    First mapping is hardcoded relu. 
    Parameters:
    -----------
        inputs: Tensor of size (N,num_receivers,f)
        activation: Activation to use in second layer. 
        num_receivers: size of x-direction.
        filters:(Deprecated) Number of filters to put before softmaxing.
        dropout: Dropout probability for first layer.
        name: Name to use for block.
    Returns:
    --------
        A (N,1,num_classes) tensor where out[n,:,j] is the probability that pixes n is
        of class j. Needs to be reshaped before use with loss. 
    """
    # (N,r,s) -> (N,1,r)
    # defaults to relu in the middle
    out = Conv2D(filters = num_receivers,
                kernel_size = (num_receivers,1),
                padding = "same",
                activation= 'relu',
                strides = (1,num_receivers))(inputs)

    out = Dropout(dropout)(out)

    # (N,1,r) -> (N,1,num_classes)
    out = Conv2D(filters = num_classes,
                kernel_size = (1,1),
                activation = activation,
                padding = "same")(out)
    return out

def multiclass_output_layer_2d(inputs,
                    num_classes = 2,
                    activation = 'softmax',
                    dropout = 0.5,
                    name = 'softmax'):
    """
    A little wrapper for a 2D so
    tensors = (t-direction, x-direction, filters)
    A final layer for 2d multi velocity detection. Outputs classes
    probabilities at each pixel.
    
    (Nt,Nx,f) -> (Nt,Nx,num_classes)

    Parameters:
    -----------
        inputs: Tensor of size (Nt,Nx,f)
        activation: Activation to use in second layer. 
        dropout: Dropout probability for first layer.
        name: Name to use for block.
    Returns:
    --------
        A (Nt,Nx,num_classes) tensor where out[n,m,j] is the probability that pixes (n,m) is
        of class j. Needs to be reshaped before use with loss. 
    """
    print(name)
    out = Conv2D(filters = num_classes,
                kernel_size = (1,1),
                activation = activation,
                padding = "same",
                name = name)(inputs)
    return out
