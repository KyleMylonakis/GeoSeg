from keras.layers import  Conv2D, Concatenate, Dropout, BatchNormalization
from keras.layers import Conv2DTranspose
from keras.layers import Activation

from blocks.Blocks import Blocks
from blocks.ConvBlock import ConvBlock
import blocks.building_blocks as building_blocks 

# TODO: Doc the block
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
        Parameters:
        -----------
            config: A configuration for the block. Containing all of the 
                remaining kwargs.(dict)
            filters: Number of filters for the base_block. (int)
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
    
    input_layers = [inputs]
    for i in range(num_layers):
        name = name + '/denselayer'+str(i)
        if i >0:
            ins = Concatenate()(input_layers)
        else:
            ins = inputs

        out_rate = growth_rate

        out =building_blocks.conv_layer(ins,
                            activation=activation,
                            filters=out_rate, 
                            kernel_size=kernel_size, 
                            dropout=dropout,
                            bottleneck = bottleneck,
                            bottleneck_factor=bottleneck_factor,
                            name = name+str(i))
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
    
    input_layers = [inputs]
    for i in range(num_layers):
        name = name + '/denselayer'+str(i)
        if i >0:
            ins = Concatenate()(input_layers)
        else:
            ins = inputs

        out_rate = growth_rate

        out = building_blocks.bn_conv_layer(ins,
                            activation=activation,
                            num_filters=out_rate, 
                            kernel_size=kernel_size, 
                            dropout=dropout,
                            bottleneck = bottleneck,
                            bottleneck_factor= bottleneck_factor,
                            name = name+str(i))
        input_layers.append(out)
    
    return out 