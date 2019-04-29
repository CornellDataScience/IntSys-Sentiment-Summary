from abc import ABC, abstractmethod

class Optimizer(ABC):

    '''
    Raises KeyError if not all values in keys are keys in dict,
    otherwise returns the values in dict associated with the keys
    in keys in the order each key appears in keys
    '''
    def _unpack_dict(self, dict, *keys):
        out = []
        for key in keys:
            v = dict.get(key)
            if v is None:
                raise ValueError("Key (" + key +") missing from dict")
            out.append(v)
        return tuple(out)


    @abstractmethod
    def optimize(self, X):
        pass
