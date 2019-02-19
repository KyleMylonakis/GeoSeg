#   A UNet class based on the design in [1] using a block. The
#   model is composed of an encoding branch, bridge, and decoding branch. 
#   We refer to an L-layer UNet as one which has an L-block Encoding branch
#   and an L-block Decoding branch. The network's block has a compression factor, c, which 
#   controls the resolution size change. On the Encoding/Decoding branch each block 
#   down/upsamples by a factor of c. The branches are connected through 
#   a bridge and concatenations from the encoding to decoding branch. 
#   
#   Overall structure is 
#   input -> first_layer -> Encode-> |
#                  V          V    bridge 
#   output <- final_layer <- Decode <-|


#   References:
#       1) O. Ronnenberger, et. al.: U-net: Convolutional networks for 
#           biomedical image segmentation. (2015) 
#           https://arxiv.org/abs/1505.04597.

from keras.layers import Concatenate, Conv2D

import json
from meta_arch.MetaModel import MetaModel
from meta_arch.ConvNet import ConvNet

class UNet(ConvNet):
    def __init__(self,
                meta_config=None,
                block = None,
                transfer_branch = None,
                noise = None,
                num_layers = 2, 
                num_classes = 2,
                compression = 2,
                init_filters = 4,
                name = 'UNet',
                first_layer = False,
                first_kernel_size = (3,1),
                first_activation = 'relu',
                first_padding = 'same',
                final_kernel_size = (1,1),
                final_strides = (1,3),
                final_padding = 'same',
                final_activation = 'softmax',
                final_name = 'binary-1d'):
        """
        Creates UNet subclass of ConvNet. For description of
        Parameters, Attributes, and methods see ConvNet.py.

        The only difference in the objects is the method main_model_fn()
        which now wraps  __unet_model_fn().
        """
        super().__init__(meta_config = meta_config,
                        block = block,
                        transfer_branch= transfer_branch,
                        noise=noise,
                        num_layers=num_layers,
                        num_classes=num_classes,
                        compression=compression,
                        name = name,
                        first_layer = first_layer,
                        first_kernel_size = first_kernel_size,
                        first_activation = first_activation,
                        first_padding = first_padding,
                        final_kernel_size = final_kernel_size,
                        final_strides = final_strides,
                        final_padding = final_padding,
                        final_activation = final_activation,
                        final_name = final_name)
    # Override ConvNet's main_model_fn method
    def main_model_fn(self):
        """
        Wraps __encoder_decorder_fn() with
        parameters from EncoderDecoder object
        fed into it.
        """
        return lambda x:self.__unet_model_fn(x)
    
    def __unet_model_fn(self,inputs):
        """
        A UNet style network [1] for input segmentation problems.
        This portion is the encode and decode branches connected 
        with a bridge which need to be topped with a prediction 
        layer later depending on what kind of segmentation problem 
        is being tackled.

        The input is fed into the encoding branch where the input 
        and number of filters is decreased/increased by a factor 
        of compression. Then fed into a final encoding block, 
        the `bridge`, and finally fed into the decoding branch where
        the input and number of  features is increased/decreased by a 
        factor of compression. 

            input -> first_layer -> Encode-> |
                          V           V     bridge 
            output <- final_layer <- Decode <-|

        The input and d-out should have the same shape.
        If this is for an L-layer EncoderDecorder then if our input
        has shape (N,r,f) e-out has shape:
        1D: (N/(c^L),r,f*c^L)
        2D: (N/(c^L),r/(c^L),f*c^L)

        Parameters:
        -----------
            inputs: A rank three tensor.
        Retruns:
            A rank three tensor of the same shape as inputs.
        """
        compression = self.compression
        num_layers = self.num_layers 
        block = self.block 

        out = inputs
        end_points = [0]*(num_layers)

        num_filters = block.config['filters']
        # Encoding branch
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
        # Decoding branch
        for i in range(num_layers):
            fine_in = end_points[i]

            num_filters = int(num_filters // compression)

            out = block.up_sample(tag = str(i), 
                                filters = num_filters)(out)
            
            coarse_in = block.base_block(tag = 'up_'+str(i),
                            filters = num_filters)(out)
            out = Concatenate()([coarse_in,fine_in])

        # Connect inputs with last encoding output
        fine_in = inputs
        num_filters = int(num_filters // compression)
        coarse_in = block.up_sample(tag = str(num_layers+1), 
                            filters = num_filters)(out)
        out = Concatenate()([coarse_in,fine_in])
        return out 

    
    