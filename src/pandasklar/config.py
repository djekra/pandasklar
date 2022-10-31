


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
            
            # verbose?
            if Config.get('VERBOSE'):  
                print( name,'=', value)
                if name == 'VERBOSE':
                    print( '--> setting verbose=True as default for all pandasklar functions\n' )
        else:
            raise NameError("Name not accepted in set() method")