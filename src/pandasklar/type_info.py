
import collections, warnings

import numpy  as np
import pandas as pd

from pandasklar.config       import Config

#from pandasklar.analyse  import val_first_valid, val_last_valid



#
#from functools   import partial  
#


#
#
#from pandas.api.types import is_string_dtype
#from pandas.api.types import is_numeric_dtype
#
#
#from pandasklar.pandas   import scale, drop_cols, rename_col, move_cols, reset_index, dataframe, group_and_agg
#from pandasklar.pandas   import top_values, top_values_count
#
#
#
#try:
#    import qgrid
#except ImportError:
#    print('qgrid nicht importierbar')
#    
    


#################################################################################
# ...............................................................................
# Klasse type_info
# ...............................................................................
#################################################################################
    
class type_info:
    """ Liefert Informationen zu Pandas-Types und standardisiert diese.
        Wird mit Irgendwas initialisiert, z.B. mit dem Namen einer Klasse, oder mit der Klasse selbst.
        Oder, noch besser, mit einer Series.
        Bsp.: i = type_info('Int32')     
              i.info()          # liefert alle Attribute, darunter zum Beispiel:
              i.class_object    # das Klassenobjekt
              i.name            # der Name des Dtypes
              i.name_instance   # type der Inhalte der Series
              i.instance1       # ein Beispielexemplar, das nicht NaN ist
    
    """
    
    def __init__(self, search):
        
        from pandasklar.analyse  import val_first_valid, val_last_valid
        
        # Beispielinstanzen
        self.instance1 = None
        self.instance2 = None
        if isinstance(search, pd.Series): # Es wurde eine Series übergeben
            self.instance1 = val_first_valid(search)
            self.instance2 = val_last_valid(search)     
            search = str(search.dtype)
        elif not isinstance(search, str): # Es wurde eine Klasse übergeben
            search = str(search)
            
        # search ist jetzt str
        self.search        = search
        self.name          = None          
        self.framework     = ''        
        self.name_short    = search
        self.name_long     = None    
        self.class_object  = None
        self.is_hashable   = None
        self.nan_allowed   = True
        self.name_instance = ''
        self.xmin          = None
        self.xmax          = None        

        # framework
        if 'Dtype' in search:
            self.framework = 'pd'  
            
        # name_instance
        if self.instance1 is not None:
            if (str(type(self.instance1)) == str(type(self.instance2))): 
                self.name_instance = type(self.instance1).__name__
            else: 
                self.name_instance = 'mix'
                
        # is_hashable
        if self.name_instance == 'mix':
            self.is_hashable = False    
        else:
            self.is_hashable = isinstance(self.instance1, collections.Hashable)  and  isinstance(self.instance2, collections.Hashable)
            
        # name_short
        if '.' in search:
            self.name_short = search.split('.')[-1]
        self.name_short = self.name_short.replace("<class '","").replace("'>","").replace("numpy.","").replace("Dtype","")       

            
        # name_long
        if self.name_short.startswith( ('int','uint') ):
            self.name_long   = 'np.' + self.name_short
            self.nan_allowed = False 
            self.framework   = 'np'             
            
        elif self.name_short.startswith( 'float' ):
            self.name_long  = 'np.' + self.name_short    
            self.framework  = 'np'              
            
        elif self.name_short.startswith( 'Int' ):
            self.name_long = 'pd.' + self.name_short + 'Dtype'  
            self.framework = 'pd'              
            
        elif self.name_short.startswith( ('str','Str') ):
            self.name_long  = 'pd.StringDtype'  
            self.name_short = 'string' # Namen standardisieren, damit astype ihn versteht
            self.framework  = 'pd'    
            
        elif self.name_short.startswith( 'obj' ):
            self.name_long  = 'object'  
            self.framework  = ''      
            
        elif self.name_short.startswith( ('category','Categorical') ):
            self.name_short = 'category'              
            self.name       = 'pd.category'              
            self.name_long  = 'pd.Categorical'  
            self.framework  = 'pd'                
     
        # name
        if self.framework:
            self.name = self.framework + '.' + self.name_short
        else:
            self.name = self.name_short
            
        # class_object    
        if self.name_long:
            self.class_object = eval(self.name_long)
            
        # xmin und xmax
        #print(self.name_short)
        if self.framework == 'np'  and  self.name_short.startswith( ('int','uint') ):
            self.xmin = np.iinfo(self.class_object).min
            self.xmax = np.iinfo(self.class_object).max   
        elif self.framework == 'pd'  and  self.name_short.startswith( 'Int' ):
            co = eval(  'np.' + self.name_short.lower()  ) # class_object faken
            self.xmin = np.iinfo(co).min  
            self.xmax = np.iinfo(co).max    
        elif self.framework == 'np'  and  self.name_short.startswith( 'float' ):
            self.xmin = np.finfo(self.class_object).min
            self.xmax = np.finfo(self.class_object).max                
            

        
    def info(self):
        """ Liefert alle Attribute"""
        result = dict(self.__dict__) # Kopie ziehen
        del result['search']
        return result
    
    

