import abc 
class Kernel(abc.ABC):
    def __init__(self):
        pass 

    @property 
    def dimension(self):
        pass 

    @abc.abstractmethod
    def __call__(self, x, y):
        pass 

    @abc.abstractmethod
    def grad_x(self, x, y):
        pass
    
    @abc.abstractmethod
    def grad_y(self, x, y):
        pass 
    
    @abc.abstractmethod
    def grad_xy(self, x, y):
        pass
    
    @abc.abstractmethod
    def grad_yx(self, x, y):
        pass