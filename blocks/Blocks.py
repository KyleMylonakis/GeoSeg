from abc import ABC, abstractmethod
import json

class Blocks(ABC):
    def __init__(self, name, config, config_path = None):
        super().__init__()
        
        assert 'base' in config.keys(), 'config must have "base" key'
        assert 'transition' in config.keys(), 'config must have "transition" key'

        self.config = config

        if not config_path is None:
            with open(config_path+'.json','w') as f:
                print('SAVING BLOCK', config_path)
                json.dump(config,f,indent=2)

    @abstractmethod
    def base_block(self,tag):
        pass 
    
    @abstractmethod
    def up_sample(self,tag,filters):
        pass 
    
    @abstractmethod
    def down_sample(self,tag,filters):
        pass
