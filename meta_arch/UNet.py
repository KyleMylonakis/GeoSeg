from abc import ABC, abstractmethod
from keras.layers import Concatenate, Conv2D

import json
from meta_arch.MetaModel import MetaModel
from meta_arch.ConvNet import CNN

class UNet(CNN):
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
                first_padding = 'padding',
                final_kernel_size = (1,1),
                final_strides = (1,3),
                final_padding = 'same',
                final_activation = 'softmax',
                final_name = 'softmax'):
        
        super().__init__(meta_config = meta_config,
                        num_layers=num_layers,
                        num_classes=num_classes,
                        compression=compression,
                        init_filters = 4,
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
        return lambda x:self.__unet_model_fn(x)
    
    def __unet_model_fn(self,inputs):
        #init_filters = self.init_filters
        compression = self.compression
        num_layers = self.num_layers 
        block = self.block 

        out = inputs
        end_points = [0]*(num_layers)

        num_filters = block.config['filters']
        for i in range(num_layers):
            
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

        for i in range(num_layers):
            fine_in = end_points[i]
            num_filters = int(num_filters // compression)
            coarse_in = block.up_sample(tag = str(i), 
                                filters = num_filters)(out)
            out = Concatenate()([coarse_in,fine_in])
        
        fine_in = inputs
        num_filters = int(num_filters // compression)
        coarse_in = block.up_sample(tag = str(num_layers+1), 
                            filters = num_filters)(out)
        out = Concatenate()([coarse_in,fine_in])
        return out 

    