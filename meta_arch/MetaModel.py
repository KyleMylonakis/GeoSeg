from abc import ABC, abstractmethod
from keras import Model, Input
from keras.layers import Reshape
import json 

class MetaModel(ABC):
    def __init__(self,config):
        
        super().__init__()
        self.config = config
        self.meta_config = config['model']['meta_arch']
        
        if self.config['model']['block']['name'] == 'res':
            print('RESIDUAL BLOCK DETECTED')
            msg = "Residual blocks must have a first layer"
            assert self.first_layer, msg
    
    def dump_config(self,path):
        with open(path+'.json','w') as f:
                json.dump(self.config,f,indent=2)

    @abstractmethod
    def main_model_fn(self):
        pass 

    @abstractmethod
    def final_layer_fn(self):
        pass
    
    def build_model(self, input_shape, output_shape = None):
        if output_shape is None:
            output_shape = input_shape[0]
        
        self.input_shape = input_shape

        inputs = Input(shape = input_shape, name = 'inputs')
        if self.first_layer:
            out = self.first_layer_fn()(inputs)
        else:
            out = inputs

        out = self.main_model_fn()(out)
        out = self.final_layer_fn()(out)
        out = Reshape((output_shape,self.meta_config['num_classes']))(out)

        model = Model(inputs = inputs, outputs = out)
        return model