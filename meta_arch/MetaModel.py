from abc import ABC, abstractmethod
from keras import Model, Input
from keras.layers import Reshape
import json 

class MetaModel(ABC):
    """
    An abstract base class for all meta-architectures. Each
    MetaModel subclass will have a main model function that processe
    the data and a final layer which outputs predictions for the model.
    
    Attributes:
    -----------
        config: A configuration dictionary for the model. 
        meta_config: config['model']['meta_arch']
    
    Methods: (See functions for details)
    --------
        dump_config(self,path): Dumps a json of the config to path.json.
        main_model_fn(self): A keras function which takes an input tensor.
            Must be defined in subclass.
        final_layer(self): A keras function which takes the output of main_model_fn.
            Must be defined in subclass
        build_model(self,input_shape, output_shape = None): Compiles the 
            keras model: 
                inputs -> main_model_fn -> final_layer -> Reshape
            where the reshape is (outshape,meta_config['num_classes'])
    """
    def __init__(self,config):
        """
        The MetaModel class with the above methods and attributes. Represents
        an abstraction of model architectures to keep functionality of training
        and inference of models for various architectures.

        Parameters:
        -----------
            config: A dictionary defining the model architecture.
                Ex: config = {'model':{
                        'meta_arch':{...},...}
                        'block':{...}}
        Returns:
        -----------
            A MetaModel class.
        """
        super().__init__()
        self.config = config
        self.meta_config = config['model']['meta_arch']
        
        # A check for resblocks to make sure fiters match up
        # for first layer.
        if self.config['model']['block']['name'] == 'res':
            msg = "Residual blocks must have a first layer"
            assert self.first_layer, msg
    
    def dump_config(self,path):
        """
        Dumps the config dictionary as path.json
        
        Parameters:
        -----------
            path: The path to save the config json.
        
        Returns:
        --------
            None. Creates the file path.json from the 
            configuration dictioanry.
        """
        with open(path+'.json','w') as f:
                json.dump(self.config,f,indent=2)

    @abstractmethod
    def main_model_fn(self):
        """
        The model function to be used. It must return a function
        which can be compiled by a keras.Model class. 

        Ex: If your model function is my_clever_model(inputs,params), then
        main_model_fn could be
            def main_model_fn(self):
                return lambda x: my_clever_model(x,self.params)
        """
        pass 

    @abstractmethod
    def final_layer_fn(self):
        """
        A composition of keras layers to be used for final prediction
        of models. Should return a function which can be compiled by
        a keras.Model object. 
        
        Ex: If your final layer is my_clever_pred(inputs,params), then
        main_model_fn could be
            def final_layer_fn(self):
                return lambda x: my_clever_pred(x,self.params)
        """
        pass
    
    def build_model(self, input_shape, output_shape = None):
        """
        Compiles a keras Model object with architecture:
            inputs -> main_model_fn -> final_layer_fn -> Reshape
        
        Assumes the network is a semantic segmentation type model so 
        Reshape is by default (input_shape, num_classes). If final_layer_fn
        is a softmax then Reshape(i,j) can be interpreted as the probability 
        that pixel i belongs to class j. 

        Currently only supports 1-D output for depth segmentation problems. 

        Parameters:
        -----------
            input_shape: Shape of the input tensor to be fed into 
                main_model_fn().
            output_shape: Shape of Reshape layer output (output_shape, num_classes). If
                None then output_shape is assumed to be same as input_shape[0]. 
        Returns:
        --------
            A keras Model object.
        """
        if output_shape is None:
            output_shape = input_shape[0]
        
        output_shape = [output_shape] + [self.meta_config['num_classes']]
        
        self.input_shape = input_shape

        inputs = Input(shape = input_shape, name = 'inputs')
        if self.first_layer:
            out = self.first_layer_fn()(inputs)
        else:
            out = inputs

        out = self.main_model_fn()(out)
        out = self.final_layer_fn()(out)
        out = Reshape(output_shape)(out)

        model = Model(inputs = inputs, outputs = out)
        return model