from transfer.transfer_blocks import basic_down_up_sample, basic_up_sample, residual_down_up_sample, residual_up_sample
from keras.layers import Lambda

TRANSFER_BLOCKS = {
    'basic-up': basic_up_sample,
    'basic-down-up': basic_down_up_sample,
    'res-up': residual_up_sample,
    'res-down-up': residual_down_up_sample
    }

class TransferBranch():
    def __init__(self,
                transfer_config = None,
                num_layers = 2,
                transfer_block = 'basic-up',
                kernel_size = (3,3),
                filters = 4,
                activation = 'relu',
                batch_norm = True,
                padding = 'same',
                dropout = 0.5
                ):
        
        default_config = {
                    'num_layers': num_layers,
                    'block' : 
                    {'name': transfer_block,
                    'kernel_size' : kernel_size,
                    'filters' : filters,
                    'activation': activation,
                    'batch_norm': batch_norm,
                    'padding' : padding,
                    'dropout' : dropout}}

        if transfer_config is None:
            transfer_config = default_config
        
        else:
            missing_keys = [k for k in default_config if k not in transfer_config.keys()]
            
            for k in missing_keys:
                transfer_config[k] = default_config[k]
        
        self.config = transfer_config

        self.block_name = transfer_config['block']['name']
        assert self.block_name in TRANSFER_BLOCKS.keys(), 'Expected transfer_block from {} but got {}'.format(TRANSFER_BLOCKS.keys(),self.block_name)
        
        self.block_fn = TRANSFER_BLOCKS[self.block_name]


    def transfer_inputs_branch(self):
        return lambda x: self.__transfer_inputs_fn(x)

    def __transfer_inputs_fn(self,inputs):
        num_layers = self.config['num_layers']

        out = inputs 
        
        for i in range(num_layers):
            blk_name = 'transfer_branch_%d'%i
            block_params = {k: self.config['block'][k] for k in self.config['block'].keys() if k != 'name'}
            print(block_params)
            #block_params['name'] = blk_name
            out = self.block_fn(inputs = out, name = blk_name, **block_params)
        
        return out
