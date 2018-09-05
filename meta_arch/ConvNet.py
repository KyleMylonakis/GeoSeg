from meta_arch.MetaModel import MetaModel
from keras.layers import Conv2D
import json

class CNN(MetaModel):
    def __init__(self,
                meta_config=None,
                num_layers = 2, 
                num_classes = 2,
                compression = 2,
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
                final_name = 'softmax'):

        default_meta_config = {
                'name': name,
                'compression': compression,
                'num_classes': num_classes,
                'num_layers': num_layers
            } 
        
        default_meta_config['final_layer'] =   {
                    'filters': num_classes,
                    'kernel_size':final_kernel_size,
                    'strides': final_strides,
                    'padding': final_padding,
                    'activation': final_activation,
                    'name': final_name
                }
        default_first_layer_config = {
                                'kernel_size': first_kernel_size,
                                'activation': first_activation,
                                'padding': first_padding,
                                }
        
        if meta_config is None:
            
            meta_config = default_meta_config
            if first_layer is not None:                                    
                meta_config['first_layer'] = default_first_layer_config
        else:
            # Get the config and add in any
            # missing keys

            missing_keys = [k for k in default_meta_config if k not in meta_config.keys()]
            #print(missing_keys)
            for k in missing_keys:
                meta_config[k] = default_meta_config[k]
            if 'first_layer' in meta_config.keys():
                first_missing = [k for k in default_first_layer_config if k not in meta_config.keys()]
                for f in first_missing:
                    meta_config['first_layer'][f] = default_first_layer_config[f]



        # Put everything into a model config
        model_config = {'model':{'meta_arch':meta_config}}
        model_config['model']['block'] = block.config
        # Set some useful attributes
        self.num_layers = meta_config['num_layers']
        self.compression = meta_config['compression']
        self.num_classes = meta_config['num_classes']
        self.block = block
        
        if 'first_layer' in meta_config.keys():
            self.first_layer = True
        else:
            self.first_layer = False
            
        super().__init__( model_config)

    def first_layer_fn(self):
        if self.first_layer:
            fl_config = self.meta_config['first_layer']
            if not 'filters' in fl_config.keys():
                fl_config['filters'] = self.block.config['filters']
            return lambda x: Conv2D(**fl_config)(x)
        else:
            return None       
    
    def final_layer_fn(self):
        conv_config = self.meta_config['final_layer']
        #if not 'filters' in conv_config.keys():
        conv_config['filters'] = self.num_classes        
        return lambda x: Conv2D(**conv_config)(x)
    
    def main_model_fn(self):
        return lambda x: self.__CNN_fn(x)
    
    def __CNN_fn(self, inputs):
        out = inputs
        num_filters = self.init_filters
        compression = self.compression
        num_layers = self.num_layers
        block = self.block
        for i in range(num_layers):
            
            out = block.base_block(tag = str(i),
                            filters = num_filters)(out)
            
            num_filters = num_filters*compression

            out = block.down_sample(tag = str(i), 
                            filters = num_filters)(out)
                 
        return out 

