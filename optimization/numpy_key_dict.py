

class NumpyKeyDict:
    def __init__(self):
        self.__dict = {}

    def __getitem__(self, k):
        return self.__dict[tuple(k)]

    def __setitem__(self, k, v):
        self.__dict[tuple(k)] = v

    def get(self, k):
        return self.__dict.get(tuple(k))
