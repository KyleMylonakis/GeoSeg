#
#   DenseBlock class.
#   Subclass of  ConvBlock
# 1) Huang, Lui (2016), 
#       Densely Connected Convolutional Networks: https://arxiv.org/pdf/1608.06993.pdf
# 2) Dmytro Mishkin and Nikolay Sergievskiy and Jiri Matas (2017)
#       Systematic evaluation of convolution neural network advances on the Imagenet: http://www.sciencedirect.com/science/article/pii/S1077314217300814

from keras.layers import  Conv2D, Concatenate, Dropout, BatchNormalization
from keras.layers import Conv2DTranspose
from keras.layers import Activation

from blocks.Blocks import Blocks
from blocks.ConvBlock import ConvBlock
import blocks.building_blocks as building_blocks 

class DenseBlock(ConvBlock):
    def __init__(self, 
                config = None,
                filters = 4,
                block_layers = 4,
                base_kernel = 3,
                trans_kernel = 2,
                batch_norm = False,
                base_activation = 'relu',
                trans_activation = 'relu',
                dropout = 0.5,
                compression = 2,
                bottleneck = True,
                bottleneck_factor = 4,
                name = 'DenseBlock'):
        """
        A Block object based on the design of DensNet in [1]. 
        The basic idea is each dense block has block_layers nubmer
        of layers where the kth layer recieves, via concatenation,
        inputs from all previous k-1 layers. For precise details
        see the reference [1].

        Parameters:
        -----------
            config: A configuration for the block. Containing all of the 
                remaining kwargs.(dict)
            filters: Number of filters for the base_block. (int)
            block_layers: Number of layers each dense block has. (int)
            base_kernel: Kernel size for base block.(int)
            trans_kernel: Kernel size for up and down sample blocks. (int)
            batch_norm: Whether base block has BN. (bool)
            base_activation: A keras activation function for the base block. (str)
            trans_activation: A keras activation function for the up and down sample blocks. (str)
            dropout: Dropout probability to use for all layers. (float)
            compression: Compression factor for transition blocks. For down_sample it increases
                channels by factor compresssion and visa versa for up sample. Currently should be set
                to 2. (int)
            bottleneck: Whether base block has a 1x1 bottleneck before convolutions. (bool)
            bottleneck_factor: The factor to expand feature maps by for a bottleneck. Will put a 
                1x1 convolution with filters*bottleneck_factor filters before the actual conv layer. (int)
            name: A name for the block. 
        
        Returns:
        --------
            A Block object. 
        """
        super().__init__( 
                config = config,
                filters = filters,
                base_kernel = base_kernel,
                trans_kernel = trans_kernel,
                batch_norm = batch_norm,
                base_activation = base_activation,
                trans_activation = trans_activation,
                dropout = dropout,
                compression = compression,
                bottleneck = bottleneck,
                bottleneck_factor=bottleneck_factor,
                name = name)
    
        # Add in DenseBLock parameters
        self.config['base']['num_layers'] = block_layers

        # Override the ConvBlock base_fn
        if self.batch_norm:
                self.base_fn = _bn_dense_block_base
        else:
            self.base_fn = _dense_block_base

    def base_block(self,tag, filters):
            bs_config = self.base_config
            blk_name = '/'.join(['base',self.config['name']+tag])
            def result_fn(inputs):
                out = self.base_fn(inputs,
                                    growth_rate=filters, 
                                    name = blk_name,
                                    **bs_config)
                return out
            return result_fn

def _dense_block_base(inputs,
                num_layers = 4,
                growth_rate = 4,
                kernel_size = 3,
                dropout = 0.5,
                bottleneck = True,
                bottleneck_factor = 4,
                activation = 'relu',
                name = 'dense_block'):
    """
    A function for a num_layers deep dense block for a 
    given growth rate. The growth rate is the number of filters
    created at each level. If the input has k0 filters then at the 
    lth layer there are k0 + k x (l-1) input filters. Because
    of this growth in input filters DenseNets can be thinner
    than traditional CNNs [1].

        Parameters:
    -----------
        inputs: A tensor of size (N,3,r).
        num_layers: The number of layers to have in the block. (int)
        growth_rate: The number of filters each layer should output.
                Can be lowered if bottlenecking.(int)
        activation: Name of a keras activation function. (str)
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
    input_layers = [inputs]
    for i in range(num_layers):
        name = name + '/denselayer'+str(i)
        if i >0:
            ins = Concatenate()(input_layers)
        else:
            ins = inputs

        out =building_blocks.conv_layer(ins,
                            activation=activation,
                            filters=growth_rate, 
                            kernel_size=kernel_size, 
                            dropout=dropout,
                            bottleneck = bottleneck,
                            bottleneck_factor=bottleneck_factor,
                            name = name)
        input_layers.append(out)
    
    return out 

def _bn_dense_block_base(inputs,
                num_layers = 4,
                growth_rate = 4,
                kernel_size = 3,
                dropout = 0.5,
                bottleneck = True,
                bottleneck_factor = 4,
                activation = 'relu',
                name = 'dense_block'):
    """
    A batch normalized version of _dense_block_base. 
    Normalization occurse before activation though some
    results suggest moving it after can improve training
    performance [2]. See _dense_block_base for description
    of the dense layers. 

        Parameters:
    -----------
        inputs: A tensor of size (N,3,r).
        num_layers: The number of layers to have in the block. (int)
        growth_rate: The number of filters each layer should output.
                Can be lowered if bottlenecking.(int)
        activation: Name of a keras activation function. (str)
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
    input_layers = [inputs]
    for i in range(num_layers):
        name = name + '/denselayer'+str(i)
        if i >0:
            ins = Concatenate()(input_layers)
        else:
            ins = inputs

        out = building_blocks.bn_conv_layer(ins,
                            activation=activation,
                            filters=growth_rate, 
                            kernel_size=kernel_size, 
                            dropout=dropout,
                            bottleneck = bottleneck,
                            bottleneck_factor= bottleneck_factor,
                            name = name)
        input_layers.append(out)
    
    return out 