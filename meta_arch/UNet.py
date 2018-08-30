from abc import ABC, abstractmethod
from keras.layers import Concatenate, Conv2D

import json
from meta_arch.MetaModel import MetaModel

class UNet(MetaModel):
    def __init__(self,
                meta_config=None,
                num_rungs = 2, 
                num_classes = 2,
                compression = 2,
                init_filters = 4,
                name = 'UNet',
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
                'num_rungs': num_rungs,
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
                            #'filters':init_filters
                                    }
                meta_config['first_layer'] = first_layer_config
            
        self.meta_config = meta_config
        model_config = {'model':{'meta_arch':meta_config}}
        model_config['model']['block'] = block.config
        self.num_rungs = meta_config['num_rungs']
        self.compression = meta_config['compression']
        self.num_classes = meta_config['num_classes']
        self.block = block
        
        if 'first_layer' in self.meta_config.keys():
            self.first_layer = True
        else:
            self.first_layer = False
            
        super().__init__(init_filters,model_config,name)
    
    def first_layer_fn(self):
        if self.first_layer:
            fl_config = self.meta_config['first_layer']
            #blk_filters = self.block.config['filters']
            #assert not 'filters' in fl_config.keys(), 'Cannot specify first layer filters they come from block'
            fl_config['filters'] = self.meta_config['init_filters']
            return lambda x: Conv2D(**fl_config)(x)
        else:
            return None
    def main_model_fn(self):
        return lambda x:self.__unet_model_fn(x)
    
    
    def final_layer_fn(self):
        conv_config = self.meta_config['final_layer']
        if not 'filters' in conv_config.keys():
            conv_config['filters'] = self.num_classes        
        return lambda x: Conv2D(**conv_config)(x)
    
    def __unet_model_fn(self,inputs):
        init_filters = self.init_filters
        compression = self.compression
        num_rungs = self.num_rungs 
        block = self.block 

        out = inputs
        end_points = [0]*(num_rungs)

        num_filters = init_filters
        for i in range(num_rungs):
            
            out = block.base_block(tag = str(i),
                            filters = num_filters)(out)
            
            num_filters = num_filters*compression

            out = block.down_sample(tag = str(i), 
                            filters = num_filters)(out)
            end_points[i] = out
            
        num_filters = num_filters*compression
        out = block.down_sample(tag = 'bridge', 
                        filters = num_filters)(out)
        
        end_points = end_points[::-1]

        for i in range(num_rungs):
            fine_in = end_points[i]
            num_filters = int(num_filters // compression)
            coarse_in = block.up_sample(tag = str(i), 
                                filters = num_filters)(out)
            out = Concatenate()([coarse_in,fine_in])
        
        fine_in = inputs
        num_filters = int(num_filters // compression)
        coarse_in = block.up_sample(tag = str(num_rungs+1), 
                            filters = num_filters)(out)
        out = Concatenate()([coarse_in,fine_in])
        return out 

    