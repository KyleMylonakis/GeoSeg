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
                block_layers = 4,
                base_kernel = 3,
                trans_kernel = 2,
                batch_norm = False,
                base_activation = 'relu',
                trans_activation = 'relu',
                dropout = 0.5,
                compression = 2,
                bottleneck = True,
                name = 'dense_block',
                config_path ='dense_block'):

        if config is None:
            block_config = {
                    'name':name,
                    'base':{},
                    'transition':{},
                    'batch_norm': batch_norm
                    }
            base_config = {}
            transition_config = {}
            
            base_config ['kernel_size'] = base_kernel
            base_config['dropout'] = dropout
            base_config['num_layers'] = block_layers
            base_config['bottleneck'] = bottleneck
            base_config['activation'] = base_activation

            transition_config['kernel_size'] = trans_kernel
            transition_config['compression'] = compression
            transition_config['activation'] = trans_activation

            block_config['base'] = base_config
            block_config['transition'] = transition_config
            
        else:
            block_config = config
        
        super().__init__(name, block_config)
        
        
        self.batch_norm = block_config['batch_norm']
        self.name = block_config['name']
        self.base_config = block_config['base']
        self.trans_config = block_config['transition']

        if self.batch_norm:
            self.base_fn = building_blocks.bn_conv_layer
        else:
            self.base_fn = building_blocks.conv_layer
        
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
    
    def up_sample(self,tag,filters):
        up_config = self.trans_config
        blk_name = '/'.join([self.config['name'],tag])
        return lambda x: building_blocks.transition_layer(inputs = x,
                                    filters=filters,
                                    name = blk_name,
                                    up_or_down='up', 
                                    **up_config)
    
    def down_sample(self,tag,filters):
        down_config = self.trans_config
        blk_name = '/'.join([self.config['name'],tag])
        return lambda x: building_blocks.transition_layer(inputs = x,
                                    filters=filters,
                                    name = blk_name,
                                    up_or_down='down', 
                                    **down_config)

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
                name = 'residual_block',
                config_path ='residual_block'):
    
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
                name = name,
                config_path =config_path)
        
        if self.config['batch_norm']:
            self.base_fn = building_blocks.bn_residual_layer
        else:
            self.base_fn = building_blocks.residual_layer
    
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
    
    def up_sample(self,tag,filters):
        up_config = self.trans_config
        blk_name = '/'.join([self.config['name'],tag])
        return lambda x: building_blocks.transition_layer(inputs = x,
                                    filters=filters,
                                    name = blk_name,
                                    up_or_down='up', 
                                    **up_config)
    