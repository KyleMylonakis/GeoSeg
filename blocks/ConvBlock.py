#
#   Convolution Block class and Residual Unit Block Class
#      
from keras.layers import  Conv2D, ReLU, Dropout, BatchNormalization
from keras.layers import Conv2DTranspose
from keras.layers import Activation
from keras.activations import relu

from blocks.Blocks import Blocks
import blocks.building_blocks as building_blocks

class ConvBlock(Blocks):
    def __init__(self, 
                config = None,
                filters = 4,
                base_kernel = 3,
                trans_kernel = 2,
                batch_norm = False,
                base_activation = 'relu',
                trans_activation = 'relu',
                dropout = 0.5,
                compression = 2,
                bottleneck = True,
                bottleneck_factor = 4,
                name = 'ConvBlock'):
        """
        A Block object which represents a basic convolutional block with:
            base_block: building_blocks.conv_layer  (or bn_conv_layer)
            up_sample: building_blocks.transition_layer("up")
            down_sample: building_blocks.transition_layer("down)
        The down and up sample blocks reduce the resolution. To maintain 
        a similar number of parameters the compression term expands or 
        contracts the number of feature maps to compenesate. 
        
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
        # Set a default config
        # in case configuration is missing parameters.
        default_config = {
                    'name':name,
                    'base':{},
                    'transition':{},
                    'batch_norm': batch_norm
                    }
        base_config = {}
        transition_config = {}
        
        base_config ['kernel_size'] = base_kernel
        base_config['dropout'] = dropout
        base_config['bottleneck'] = bottleneck
        base_config['bottleneck_factor'] = bottleneck_factor
        base_config['activation'] = base_activation

        transition_config['kernel_size'] = trans_kernel
        transition_config['compression'] = compression
        transition_config['activation'] = trans_activation

        default_config['base'] = base_config
        default_config['transition'] = transition_config

        if config is None:
            block_config = default_config
        else:
            # Get the config and add in any
            # missing keys
            block_config = config

            missing_keys = [k for k in default_config if k not in block_config.keys()]
            for k in missing_keys:
                block_config[k] = default_config[k]
        
        super().__init__(name, block_config)
        
        self.batch_norm = block_config['batch_norm']
        self.name = block_config['name']
        self.base_config = block_config['base']
        self.trans_config = block_config['transition']

        # Get conv_layer or bn_conv_layer
        # for the base block.
        if self.batch_norm:
            self.base_fn = building_blocks.bn_conv_layer
        else:
            self.base_fn = building_blocks.conv_layer
        
    def base_block(self,tag,filters):
        bs_config = self.base_config
        blk_name = '/'.join(['base',self.config['name']+tag])
        # Wrap the base function into a function
        # to match keras layer style.
        def result_fn(inputs):
            out = self.base_fn(inputs,
                                filters=filters, 
                                name = blk_name,
                                **bs_config) # Unwrap the extra parameters
            return out
        return result_fn
    
    def up_sample(self,tag,filters):
        up_config = self.trans_config
        blk_name = '/'.join([self.config['name'],tag])
        # Wrap the base function into a function
        # to match keras layer style.
        return lambda x: building_blocks.transition_layer(inputs = x,
                                    filters=filters,
                                    name = blk_name,
                                    up_or_down='up', 
                                    **up_config) # Unwrap the extra parameters
    
    def down_sample(self,tag,filters):
        down_config = self.trans_config
        blk_name = '/'.join([self.config['name'],tag])
        # Wrap the base function into a function
        # to match keras layer style.
        return lambda x: building_blocks.transition_layer(inputs = x,
                                    filters=filters,
                                    name = blk_name,
                                    up_or_down='down', 
                                    **down_config) # Unwrap the extra parameters

class ResBlock(ConvBlock):
    def __init__(self, 
                config = None,
                filters = 4,
                base_kernel = 3,
                trans_kernel = 2,
                batch_norm = False,
                base_activation = 'relu',
                trans_activation = 'relu',
                dropout = 0.5,
                compression = 2,
                bottleneck = True,
                bottleneck_factor = 4,
                name = 'ResBlock'):
        """
        A Block object which represents a basic residual block with:
            base_block: building_blocks.residual_layer  (or bn_residual_layer)
            up_sample: building_blocks.transition_layer("up")
            down_sample: building_blocks.transition_layer("down)
        The down and up sample blocks reduce the resolution. To maintain 
        a similar number of parameters the compression term expands or 
        contracts the number of feature maps to compenesate. 
        
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
        
        # Override the ConvBlock base_fn
        if self.config['batch_norm']:
            self.base_fn = building_blocks.bn_residual_layer
        else:
            self.base_fn = building_blocks.residual_layer
    
    # Override the ConvBlock base_block
    def base_block(self,tag,filters):
        bs_config = self.base_config
        blk_name = '/'.join(['base',self.config['name']+tag])
        def result_fn(inputs):
            out = self.base_fn(inputs,
                                filters=filters, 
                                name = blk_name,
                                **bs_config)
            return out
        return result_fn
    