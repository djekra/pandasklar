
# https://stackoverflow.com/questions/6198372/most-pythonic-way-to-provide-global-configuration-variables-in-config-py

class Config:
    '''
    Configuration Management 
    
    Get: 
    from pandasklar.config import Config
    verbose = Config.get('VERBOSE')  
    
    Set:
    pak.Config.set('VERBOSE', False)
    '''
    
    __conf = {
                'VERBOSE': False,
    }
    __setters = ['VERBOSE', ]

    @staticmethod
    def get(name):
        return Config.__conf[name]

    @staticmethod
    def set(name, value):
        if name in Config.__setters:
            Config.__conf[name] = value
            print('Config set', name,'=', value)
        else:
            raise NameError("Name not accepted in set() method")