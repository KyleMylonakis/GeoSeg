#
#   AutoEncoder class to use as a keras layer. 
#   Takes resolution neutral functions and builds
#   an auto encoder out of them.
#

import json
from meta_arch import utils
from meta_arch.MetaModel import MetaModel
from keras.layers import Conv2D
class AutoEncoder(MetaModel):
    def __init__(self,
                meta_config=None,
                num_layers = 2, 
                num_classes = 2,
                compression = 2,
                init_filters = 4,
                name = 'AE',
                block = None,
                first_layer = False,
                first_kernel_size = (3,1),
                first_activation = 'relu',
                first_padding = 'padding',
                final_kernel_size = (1,1),
                final_strides = (1,3),
                final_padding = 'same',
                final_activation = 'softmax',
                final_name = 'softmax'):

        if meta_config is None:
            meta_config = {
                'name': 'UNet',
                'compression': compression,
                'init_filters': init_filters,
                'num_classes': num_classes,
                'num_layers': num_layers,
                'final_layer': 
                {
                    'filters': num_classes,
                    'kernel_size':final_kernel_size,
                    'strides': final_strides,
                    'padding': final_padding,
                    'activation': final_activation,
                    'name': final_name
                }
            }
            if first_layer is not None:
                first_layer_config = {
                            'kernel_size': first_kernel_size,
                            'activation': first_activation,
                            'filters':init_filters
                                    }
                meta_config['first_layer'] = first_layer_config
            
        model_config = {'model':{'meta_arch':meta_config}}
        model_config['model']['block'] = block.config
        self.meta_config = meta_config
        self.num_layers = meta_config['num_layers']
        self.compression = meta_config['compression']
        self.num_classes = meta_config['num_classes']
        self.block = block
        
        if 'first_layer' in self.meta_config.keys():
            self.first_layer = True
        else:
            self.first_layer = False
            
        super().__init__(model_config)

    def first_layer_fn(self):
        if self.first_layer:
            fl_config = self.meta_config['first_layer']
            fl_config['filters'] = self.block.config['filters']
            #if not 'filters' in fl_config.keys():
            #    fl_config['filters'] = self.meta_config['init_filters']
            return lambda x: Conv2D(**fl_config)(x)
        else:
            return None

    def main_model_fn(self):
        return lambda x: self.__auto_encoder_fn(x)

    def final_layer_fn(self):
        conv_config = self.meta_config['final_layer']
        if not 'filters' in conv_config.keys():
            conv_config['filters'] = self.num_classes        
        return lambda x: Conv2D(**conv_config)(x)
    
    def __auto_encoder_fn(self, inputs):
        
        
        compression = self.compression
        num_layers = self.num_layers
        block = self.block
        utils._dump_config(block,'block_test')
        #print('here')
        out = inputs
        #num_filters = self.init_filters
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
