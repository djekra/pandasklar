


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
    __setters = ['VERBOSE', 'GRID_BACKEND']

    @staticmethod
    def get(name):
        try:
            return Config.__conf[name]
        except:
            return None

    
    
    @staticmethod
    def set(name, value):
        if name in Config.__setters:
            Config.__conf[name] = value
            
            # verbose?
            if Config.get('VERBOSE'):  
                print( name,'=', value)
                if name == 'VERBOSE':
                    print( '--> setting verbose={} as default for all pandasklar functions\n'.format(value) )
                elif name == 'GRID_BACKEND':
                    print( '--> setting backend={} as default for all pandasklar functions\n'.format(value)  )                    
        else:
            raise NameError("Name not accepted in set() method")