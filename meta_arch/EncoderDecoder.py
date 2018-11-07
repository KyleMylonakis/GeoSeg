#   
#   EncoderDecoder class
#   subclass of ConvNet.
#   Description: A SegNet [1] style network. The Encoder-Decoder language is borrowed from [2]. 
#   Each network is composed of an Encoding branch and a Decoding branch. We refer
#   to an L-layer Encoder-Decoder network as one which has an L-block Encoding branch
#   and an L-block Decoding branch. The network's block has a compression factor, c, which 
#   controls the resolution size change. On the Encoding/Decoding branch each block 
#   down/upsamples by a factor of c. 
#   
#   Overall structure is 
#   input -> first_layer -> Encode -> Decode -> final_layer.
#   
#   References:
#       1) E. Shelhamer et. al.: Fully Convolutional Networks for Semantic Segmentation (2015)
#           https://arxiv.org/abs/1411.4038
#       2) Z. Zhang et. al.: Road extraction by deep residual u-net (2017)
#           https://arxiv.org/abs/1711.10684

import json
from meta_arch.MetaModel import MetaModel
from meta_arch.ConvNet import ConvNet
from keras.layers import Conv2D
class EncoderDecoder(ConvNet):
    def __init__(self,
                meta_config=None,
                transfer_branch = None,
                block = None,
                num_layers = 2, 
                num_classes = 2,
                compression = 2,
                init_filters = 4,
                name = 'EnDe',
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
        Creates EncoderDecoder subclass of ConvNet. For description of
        Parameters, Attributes, and methods see ConvNet.py.

        The only difference in the objects is the method main_model_fn()
        which now wraps  __encoder_decorder_fn().
        """
        super().__init__(meta_config = meta_config,
                        transfer_branch=transfer_branch,
                        block = block,
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
    # Override ConvNet's main_model_fn method.
    def main_model_fn(self):
        """
        Wraps __encoder_decorder_fn() with
        parameters from EncoderDecoder object
        fed into it.
        """
        return lambda x: self.__encoder_decoder_fn(x)

    def __encoder_decoder_fn(self, inputs):
        """
        A SegNet style network [1] for input segmentation problems.
        This portion is the encode and decode branches which need to be 
        topped with a prediction layer later depending on what
        kind of segmentation problem is being tackled.

        The input is fed into the encoding branch where the input and number
        of filters is decreased/increased by a factor of compression. This
        is then fed into the decoding branch where the input and number of 
        features is increased/decreased by a factor of compression. 

        input -> Encode -> e-out -> Decode -> d-out

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
        num_filters = block.config['filters']        
        
        # Encoding branch
        for i in range(num_layers):            
            out = block.base_block(tag ='down_'+ str(i),
                            filters = num_filters)(out)            
            
            num_filters = num_filters*compression
            out = block.down_sample(tag = str(i), 
                            filters = num_filters)(out)            

        #Decoding branch
        for i in range(num_layers):
            out = block.base_block(tag = 'up_'+str(i),
                            filters = num_filters)(out)
            num_filters = int(num_filters // compression)
            out = block.up_sample(tag = str(i), 
                                filters = num_filters)(out)        
        return out 
