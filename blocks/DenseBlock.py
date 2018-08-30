from keras.layers import  Conv2D, Concatenate, Dropout, BatchNormalization
from keras.layers import Conv2DTranspose
from keras.layers import Activation

from blocks.Blocks import Blocks
import blocks.building_blocks as building_blocks 

class DenseBlock(Blocks):
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

def _dense_block_base(inputs,
                num_layers = 4,
                growth_rate = 4,
                kernel_size = 3,
                dropout = 0.5,
                bottleneck = True,
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
                            name = name+str(i))
        input_layers.append(out)
    
    return out 

def _bn_dense_block_base(inputs,
                num_layers = 4,
                growth_rate = 4,
                kernel_size = 3,
                dropout = 0.5,
                bottleneck = True,
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
                            name = name+str(i))
        input_layers.append(out)
    
    return out 