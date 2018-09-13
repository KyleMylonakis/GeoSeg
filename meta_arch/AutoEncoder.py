#
#   AutoEncoder class to use as a keras layer. 
#   Takes resolution neutral functions and builds
#   an auto encoder out of them.
#

import json
from meta_arch import utils
from meta_arch.MetaModel import MetaModel
from meta_arch.ConvNet import CNN
from keras.layers import Conv2D
class AutoEncoder(CNN):
    def __init__(self,
                meta_config=None,
                num_layers = 2, 
                num_classes = 2,
                compression = 2,
                init_filters = 4,
                name = 'UNet',
                block = None,
                first_layer = False,
                first_kernel_size = (3,1),
                first_activation = 'relu',
                first_padding = 'same',
                final_kernel_size = (1,1),
                final_strides = (1,3),
                final_padding = 'same',
                final_activation = 'softmax',
                final_name = 'binary-1d'):
        
        super().__init__(meta_config = meta_config,
                        num_layers=num_layers,
                        num_classes=num_classes,
                        compression=compression,
                        name = name,
                        block = block,
                        first_layer = first_layer,
                        first_kernel_size = first_kernel_size,
                        first_activation = first_activation,
                        first_padding = first_padding,
                        final_kernel_size = final_kernel_size,
                        final_strides = final_strides,
                        final_padding = final_padding,
                        final_activation = final_activation,
                        final_name = final_name)
    # Override CNN's main_model_fn method.
    def main_model_fn(self):
        return lambda x: self.__auto_encoder_fn(x)

    def __auto_encoder_fn(self, inputs):
        
        compression = self.compression
        num_layers = self.num_layers
        block = self.block
        #utils._dump_config(block,'block_test')
        out = inputs
        num_filters = block.config['filters']        
        
        for i in range(num_layers):            
            out = block.base_block(tag ='down_'+ str(i),
                            filters = num_filters)(out)            
            
            num_filters = num_filters*compression
            out = block.down_sample(tag = str(i), 
                            filters = num_filters)(out)            

        for i in range(num_layers):
            out = block.base_block(tag = 'up_'+str(i),
                            filters = num_filters)(out)
            num_filters = int(num_filters // compression)
            out = block.up_sample(tag = str(i), 
                                filters = num_filters)(out)        
        return out 
