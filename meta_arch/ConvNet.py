from meta_arch.MetaModel import MetaModel
from keras.layers import Conv2D
import json
from blocks.building_blocks import binary_output_layer_1d, multiclass_output_layer_1d

# functions to use for final layer
# should be added here. They must take as
# MUST HAVE PARAMETERS:
#   filters, 
#  activations
# Any parameters not covered in __init__(final_*) must
# be supplied by meta_config.
FINAL_LAYERS = {
            'binary-1d': binary_output_layer_1d,
            'multiclass-1d': multiclass_output_layer_1d
            }


class ConvNet(MetaModel):
    def __init__(self,
                meta_config=None,
                num_receivers=3,
                num_layers = 2, 
                num_classes = 2,
                compression = 2,
                name = 'cnn',
                block = None,
                first_layer = False,
                first_kernel_size = (3,1),
                first_activation = 'relu',
                first_padding = 'same',
                final_kernel_size = (1,1),
                final_strides = (1,3),
                final_padding = 'same',
                final_filters = 4,
                final_activation = 'softmax',
                final_name = 'binary-1d'):
        """
        A MetaModel subclass for convolutional models. Most meta-architectures can be
        subclasses of ConvNet since they share many parameters and thus will have
        similar configurations. 

        This is a 1D model. So if input is shape (N,r,f), num_layers is 3, and
        compression is 2 the output of the main_model_fn would be of shape (N/(2^3),r,f).
        In default parameters r=3 and final layer will have stride=(1,3) so the output
        of the ConvNet.build_model() would be (N/(2^3),num_classes).

        Parameters:
        -----------
            meta_config: A dictionary containing all or some of the below parameters. Assumed
                to have structure:
                    meta_config = {final_layer: {...}, 
                                    first_layer: {...},
                                    remaining_params: values}
            num_receivers: The number of receivers in the seismic data. If input
                data shape is (N,r,f) num_receivers should be r.
            num_layers: The number of convolutional blocks to use.
            num_classes: The number of classes in the sematic segmentation problem.
            compression: The factor to downsample the blocks in each layer.
            name: What to call the network. Keras layers will be 'name/layer_name'.
            block: a Block object to use in the layers. (See blocks.Block)
            first_layer: A keras layer for the first layer of model. Not all
                meta_architectures will have one.
            first_kernel_size: The kernel size to use in the first layer.
            first_activation: The activation to use in the first layer.
            first_padding: The padding to use in the first layer.
            final_layer: A keras layer type object to make predictions. 
            final_kernel_size: The kernel size to use in the final layer.
            final_activation: The activation to use in the final layer.
            final_padding: The padding to use in the final layer.
            final_filters: The number of filters in final convolution layer
                before prediction.
            final_strides: The strides to use in the final kernel for the final 2D
                convolution before prediction. 
            final_name: Name of final layer. Must be in FINAL_LAYERS.
        
        Returns:
        --------
            A MetaModels subclass ConvNet object for 1D sesimeic semantic segmentation. 

        Attributes: (For MetaModels inheritance see MetaModels.py)
        -----------            
            num_layers: Number of blocks in main_model_fn().
            compression: The factor to downsample input at each
                block.
            num_receivers: Number of recievers in input. If input
                is of shape (N,r,f) then num_recievers should be r. 
            block: A Block object. 
            first_layer: A boolean for whether the model should have
                a first layer added to its's meta_arch config.
        
        Methods:
        --------
            first_layer_fn(self):
                Returns: A function for a keras convolutional layer
                    with kernel_size, activation, and padding given
                    by first_kernel_size, first_activation, and
                    first_padding respectively. 
            final_layer_fn(self):
                Returns: The keras function FINAL_LAYERS['name']

        """
        default_meta_config = {
                'name': name,
                'compression': compression, #TODO: meta_arch.config and block.config 
                                            # both have compression. Put it in just one.
                'num_classes': num_classes,
                'num_layers': num_layers,
                'num_receivers': num_receivers
            } 

        default_meta_config['final_layer'] = {
                                        'name':final_name,
                                        'filters': final_filters,
                                        'activation': final_activation
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
            # Add missing keys from meta config
            missing_keys = [k for k in default_meta_config if k not in meta_config.keys()]
            for k in missing_keys:
                meta_config[k] = default_meta_config[k]

            # Add missing keys from final layer config
            if 'final_layer' in meta_config.keys():
                missing_final_keys = [k for k in default_meta_config['final_layer'].keys() if k not in meta_config['final_layer'].keys()]
                for k in missing_final_keys:
                    meta_config['final_layer'][k] = default_meta_config['final_layer'][k]
            else:
                meta_config['final_layer'] = default_meta_config['final_layer']
                
            # Add missing keys from first layer config
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
        self.num_receivers = meta_config['num_receivers']
        self.block = block
        
        if 'first_layer' in meta_config.keys():
            self.first_layer = True
        else:
            self.first_layer = False
            
        super().__init__( model_config)

    def first_layer_fn(self):
        """
        Returns a keras function if first_layer is True. Else it returns
        None. The layer is a Conv2D layer with parameters defined in
        first_* in the __init__. If the first_layer_config does not have
        'filters' key then the layer returns the same filters as used in 
        block.

        Returns:
            A keras function first_layer_fn(). For and input with shape (N,r,m) 
            and block with filters f then first_layer_fn()(input) has shape:
                - (N,r,f) if 'first_layers' was not set.
                - (N,r,fl) if 'first_layers':fl.
        """
        if self.first_layer:
            fl_config = self.meta_config['first_layer']
            if not 'filters' in fl_config.keys():
                fl_config['filters'] = self.block.config['filters']
            return lambda x: Conv2D(**fl_config)(x)
        else:
            return None       
    
    def final_layer_fn(self):
        """
        Returns a keras function for the final layer of the model. 
        See FINAL_LAYERS for model choices and blocks.building_blocks
        for their descriptions.
        """
        conv_config = self.meta_config['final_layer']
        final_fn = FINAL_LAYERS[conv_config['name']]
        return lambda x: final_fn(inputs = x, num_receivers = self.num_receivers, num_classes = self.num_classes,**conv_config)
    
    def main_model_fn(self):
        """
        Wraps __ConvNet_fn in a lambda.
        Returns:
        --------
            A keras function representing __ConvNet_fn as the model
            function.
        """
        return lambda x: self.__ConvNet_fn(x)
    
    def __ConvNet_fn(self, inputs):
        """
        A convolutional neural network defined by the self.config.
        It is an L block network where each block is topped with a
        transition layer that downsamples the first axis. A birds eye
        view

        inputs -> block.base -> block.transition -> ... ->block.base -> block.transition -> out
        
        If inputs has shape (N,r,f) then out will have shape
            ( N/(c^M), r, self.block.filters)
        
        Where c is self.compression and M is self.num_layers.
        Parameters:
        -----------
            inputs: A rank 3 tensor.
        Returns:
        --------
            A rank 3 tensor whose first axis is downsampled by a factor
            of self.compression^(self.num_layers).
        """
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

