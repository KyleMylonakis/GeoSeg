#   
#   Blocks to use for input transfer branches.
#   ALL blocks should take as input: 
#   
#   kernel_size, filters, activation, batch_norm, padding, and dropout
#   
#   Blocks can take additional parameters but default 
#   TransferBranches will use these parameters.


from keras.layers import Conv2D, Conv2DTranspose, Dropout, BatchNormalization, Activation
from blocks.building_blocks import conv_layer, residual_layer, bn_conv_layer, bn_residual_layer

def batch_norm_layer(inputs,activation,batch_norm):
    """
    Applies batch normalization  to inputs with given activation if 
    batch_norm is true.

    Parameters:
    -----------
        inputs: A tensor.
        activation: A name of a keras activation function.
        batch_norm: bool for whether to apply batch norm.
    Returns:
    --------
        If batch norm is True returns (Batch normalized input, None) if 
        batch norm is False returns (inputs, activation).
    """
    if batch_norm:
        out = BatchNormalization()(inputs)
        out = Activation(activation)(out)
        bn_act = None
    else:
        out = inputs
        bn_act = activation
    return out, bn_act

# A basic upsample of the data in receiver axis
def basic_up_sample(inputs,
                filters = 4, 
                up_factor = 2,
                kernel_size= (3,3),
                activation = 'relu',
                padding = 'same',
                dropout = 0.5,
                batch_norm = True,
                name = 'basic_up_sample'):
    """
    Up samples the receiver axis (second axis) of the input
    data. 
        (N,r,3) -> (N, up_factor r, filters)

    Parameters:
    -----------
        inputs: A tensor.
        up_factor: The factor to up sample the second axis by. (int)
        activation: Name of a keras activation function. (str)
        filters: Number of output filters. Can be lowered if bottlenecking.(int)
        batch_norm: Whether or not to apply batch nomralization. (bool)
        kernel_size: The size of kernel to use for the convolution.
        dropout: The percent probability for each individual layer's dropout

    
    Returns:
    --------
        A tensor of shape (N, up_factor r, filters).
    """

    out, new_activation = batch_norm_layer(inputs,activation,batch_norm)
    
    out = Conv2DTranspose(filters = filters,
                            kernel_size=kernel_size,
                            strides = (1,up_factor),
                            activation = new_activation,
                            padding = padding,
                            name = name +'/conv_up')(out)
    out = Dropout(dropout)(out)

    return out 

def basic_down_up_sample(inputs,
                filters = 4, 
                up_factor = 2,
                down_factor = 2,
                kernel_size= (3,3),
                batch_norm = True,
                activation = 'relu',
                padding = 'same',
                dropout = 0.5,
                name = 'basic_up_sample'):
    """
    Down samples the first and up samples the second axis 
    of the input data. 
        (N,r,3) -> (N /down_factor, up_factor r, filters)
    

    Parameters:
    -----------
        inputs: A tensor.
        up_factor: The factor to up sample the second axis by. (int)
        down_factor: The factor to down sample the first axis by. (int)
        activation: Name of a keras activation function. (str)
        batch_norm: Whether or not to apply batch nomralization. (bool)
        filters: Number of output filters. Can be lowered if bottlenecking.(int)
        kernel_size: The size of kernel to use for the convolution.
        dropout: The percent probability for each individual layer's dropout
    
    Returns:
    --------
        A tensor of shape (N /down_factor, up_factor r, filters).
    """
    out, new_activation = batch_norm_layer(inputs,activation,batch_norm)

    out = Conv2D(filters = filters,
                            kernel_size=kernel_size,
                            strides = (down_factor,1),
                            activation = new_activation,
                            padding = padding,
                            name = name +'/conv_down')(out)

    out = Dropout(dropout)(out)

    out, new_activation = batch_norm_layer(out, activation, batch_norm)
    
    out = Conv2DTranspose(filters = filters,
                            kernel_size=kernel_size,
                            strides = (1,up_factor),
                            activation = new_activation,
                            padding = padding,
                            name = name +'/conv_up')(out)
    out = Dropout(dropout)(out)

    return out 

def residual_up_sample(inputs,
                filters = 4, 
                up_factor = 2,
                kernel_size= (3,3),
                batch_norm = True,
                activation = 'relu',
                padding = 'same',
                dropout = 0.5,
                name = 'residual_up_sample'):
    
    """
    Up samples the reciever axis (second axis) of the input
    data. 
        (N,r,3) -> (N, up_factor r, filters) 
                        L-> (N, up_factor r, filters) --+--> out
                        L->-----------------------------^    

    Parameters:
    -----------
        inputs: A tensor.
        activation: Name of a keras activation function. (str)
        filters: Number of output filters. Can be lowered if bottlenecking.(int)
        batch_norm: Whether or not to apply batch nomrmalization. (bool)
        kernel_size: The size of kernel to use for the convolution. 
        dropout: The percent probability for each individual layer's dropout
    """

    out, new_activation = batch_norm_layer(inputs, activation,batch_norm)
    
    out = Conv2DTranspose(filters = filters,
                            kernel_size=kernel_size,
                            strides = (1,up_factor),
                            activation = new_activation,
                            padding = padding,
                            name = name +'/conv_up')(out)
    out = Dropout(dropout)(out)

    # No bottleneck for now.
    #TODO: Decide if we want bottlenecking.
    if batch_norm:
        out = bn_residual_layer(inputs = out, filters=filters,
                            bottleneck = False,
                            activation = activation,
                            kernel_size=kernel_size,
                            dropout=dropout,
                            name=name+'/residual'  )
    
    else:
        out = residual_layer(inputs = out, filters=filters,
                            bottleneck = False,
                            activation = activation,
                            kernel_size=kernel_size,
                            dropout=dropout,
                            name=name+'/residual' )
    return out 


def residual_down_up_sample(inputs,
                filters = 4, 
                up_factor = 2,
                down_factor = 2,
                kernel_size= (3,3),
                batch_norm = True,
                activation = 'relu',
                padding = 'same',
                dropout = 0.5,
                name = 'basic_up_sample'):
    """
    Down samples the first and up samples the second axis 
    of the input data. 
        (N,r,3) -> (N /down_factor, up_factor r, filters)
    

    Parameters:
    -----------
        inputs: A tensor.
        up_factor: The factor to up sample the second axis by. (int)
        down_factor: The factor to down sample the first axis by. (int)
        activation: Name of a keras activation function. (str)
        batch_norm: Whether or not to apply batch normalization. (bool)
        filters: Number of output filters. Can be lowered if bottlenecking.(int)
        kernel_size: The size of kernel to use for the convolution.
        dropout: The percent probability for each individual layer's dropout
    
    Returns:
    --------
        A tensor of shape (N /down_factor, up_factor r, filters).
    """
    out, new_activation = batch_norm_layer(inputs,activation,batch_norm)

    out = Conv2D(filters = filters,
                    kernel_size=kernel_size,
                    strides = (down_factor,1),
                    activation = new_activation,
                    padding = padding,
                    name = name +'/conv_down')(out)

    out = Dropout(dropout)(out)

    out, new_activation = batch_norm_layer(out, activation, batch_norm)
    
    out = Conv2DTranspose(filters = filters,
                            kernel_size=kernel_size,
                            strides = (1,up_factor),
                            activation = new_activation,
                            padding = padding,
                            name = name +'/conv_up')(out)
    out = Dropout(dropout)(out)
    # No bottleneck for now.
    #TODO: Decide if we want bottlenecking.
    if batch_norm:
        out = bn_residual_layer(inputs = out, filters=filters,
                            bottleneck = False,
                            activation = activation,
                            kernel_size=kernel_size,
                            dropout=dropout,
                            name=name+'/residual'  )
    
    else:
        out = residual_layer(inputs = out, filters=filters,
                            bottleneck = False,
                            activation = activation,
                            kernel_size=kernel_size,
                            dropout=dropout,
                            name=name+'/residual' )
    return out 

def up_output_shape(input_shape, num_layers, factor = 2):
    result = []*len(input_shape)
    result[:] = input_shape[:]
    result[1] = input_shape[1]*(2**num_layers)
    return result

def down_up_output_shape(input_shape, num_layers, factor = 2):
    result = []*len(input_shape)
    result[:] = input_shape[:]
    result[0] = input_shape[0]//(2**num_layers)
    result[1] = input_shape[1]*(2**num_layers)
    return result

# Naming the final blocks
TRANSFER_BLOCKS = {
    'basic-up': [basic_up_sample, up_output_shape],
    'basic-down-up': [basic_down_up_sample, down_up_output_shape],
    'res-up': [residual_up_sample, up_output_shape],
    'res-down-up': [residual_down_up_sample,down_up_output_shape]
    }
