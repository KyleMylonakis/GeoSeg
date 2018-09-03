def _repeat_block(block, num_layers):    
    def result_fn(inputs):
        out = inputs
        for i in range(num_layers):
            out = block(inputs)
        return out 
    return result_fn

import json 
def _dump_config(thing,path):
    with open(path+'.json','w') as fw:
        json.dump(thing.config,fw,indent=2)
