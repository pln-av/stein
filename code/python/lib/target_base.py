import abc 
class Target(abc.ABC):
    def __init__(self):
        pass 

    @property 
    def dimension(self):
        pass 

    @abc.abstractmethod
    def __call__(self, z):
        pass 

    @abc.abstractmethod
    def log_grad(self, z):
        pass
    