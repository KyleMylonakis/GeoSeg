#
# Base class for all blocks
#
#   A Block object consists of a base_block, up_sample,
# and down_sample. The base_block is a composition
# of keras Layers objects which preserves resolution of input;
# whereas; the up and down sample will expand and contract their
# input resolution's respectively. 
from abc import ABC, abstractmethod

class Blocks(ABC):
    def __init__(self, name, config):
        super().__init__()
        
        assert 'base' in config.keys(), 'config must have "base" key'
        assert 'transition' in config.keys(), 'config must have "transition" key'

        self.config = config

    @abstractmethod
    def base_block(self,tag):
        """
        A keras Layers type object which may change the number 
        of features channels in an input but cannot change
        any of the other dimensions. Accepts 'tag' input to 
        help meta_archs with naming.
        """
        pass 
    
    @abstractmethod
    def up_sample(self,tag,filters):
        """
        A keras Layers type object which increases the resolution
        of its input and has 'filters' output channels. Must increase
        resolution by same factor that down_sample decreases resolution.
        """
        pass 
    
    @abstractmethod
    def down_sample(self,tag,filters):
        """
        A keras Layers type object which decreases the resolution
        of its input and has 'filters' output channels. Must decrease
        resolution by same factor that up_sample increases resolution.
        """
        pass
