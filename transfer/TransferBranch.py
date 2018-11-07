#
#   The TransferBranch class. These are small fully convolutional networks used 
#   to learn how to reshape input data. The intent is to increase the receiver resolution 
#   and perhaps downsample the time resolution.
#

from transfer import transfer_blocks
from keras.layers import Lambda

class TransferBranch():
    def __init__(self,
                config = None,
                input_shape = [32,32,3],
                num_layers = 2,
                transfer_block = 'basic-up',
                kernel_size = (3,3),
                filters = 4,
                activation = 'relu',
                batch_norm = True,
                padding = 'same',
                dropout = 0.5
                ):
        """
        An input transfer branch to add to a meta_arch. Essentially
        a feed forward CNN designed to reshape input based on the 
        transfer block chosen. A k-layer transfer branch with input
        shape (N,r,3) will have output
            up: (N,r*2**k,f)
            down_up: (n/(2**k),r*2**k,f)
        
        The downsampling is performed by strided convolutions and the up
        sampling by strided convolution transposes.
        
        Parameters:
        -----------
            config: A dictionary setting some or all of the below parameters. (dict)
            num_layers: The number of layers in the transfer branch. (int)
            transfer_block: Name of a transfer block from transfer_blocks.TRANSFER_BLOCKS. (str)
            activation: Name of a keras activation function. (str)
            batch_norm: Whether or not to apply batch normalization. (bool)
            filters: Number of output filters. Can be lowered if bottlenecking.(int)
            kernel_size: The size of kernel to use for the convolution.
            dropout: The percent probability for each individual layer's dropout
        """

        default_config = {
                    'num_layers': num_layers,
                    'input_shape': input_shape,
                    'block' : {
                        'name': transfer_block,
                        'kernel_size' : kernel_size,
                        'filters' : filters,
                        'activation': activation,
                        'batch_norm': batch_norm,
                        'padding' : padding,
                        'dropout' : dropout
                            }
                        }

        if config is None:
            config = default_config
        
        else:
            missing_keys = [k for k in default_config if k not in config.keys()]
            
            for k in missing_keys:
                config[k] = default_config[k]
            # Add in missing bock keys
            missing_block_keys = [k for k in default_config['block'] if k not in config['block'].keys()]
            
            for k in missing_block_keys:
                config['block'][k] = default_config['block'][k]
            
        self.config = config
        self.input_shape = config['input_shape']
        self.block_name = config['block']['name']
        assert self.block_name in transfer_blocks.TRANSFER_BLOCKS.keys(), 'Expected transfer_block from {} but got {}'.format(transfer_blocks.TRANSFER_BLOCKS.keys(),self.block_name)
        
        # First element is block second is output shape function
        self.block_data = transfer_blocks.TRANSFER_BLOCKS[self.block_name]
        self.block_fn = self.block_data[0] 
        
        self.num_layers = config['num_layers']
        self.output_shape = self.block_data[1](self.input_shape,self.num_layers)

    def transfer_inputs_branch(self):
        return lambda x: self.__transfer_inputs_fn(x)

    def __transfer_inputs_fn(self,inputs):
        num_layers = self.num_layers

        out = inputs 
        
        for i in range(num_layers):
            blk_name = 'transfer_branch_%d'%i
            block_params = {k: self.config['block'][k] for k in self.config['block'].keys() if k != 'name'}
            out = self.block_fn(inputs = out, name = blk_name, **block_params)        
        return out
    