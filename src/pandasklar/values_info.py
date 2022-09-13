
import numpy  as np
import pandas as pd

from pandas.api.types import is_string_dtype, is_numeric_dtype

from pandasklar.config       import Config

#import collections, warnings
#
#from functools   import partial  
#

#
#
#
#
#from pandasklar.pandas     import scale, drop_cols, rename_col, move_cols, reset_index, dataframe, group_and_agg
#from pandasklar.pandas     import top_values, top_values_count
#from pandasklar.type_info  import type_info
#
#
#
#try:
#    import qgrid
#except ImportError:
#    print('qgrid nicht importierbar')
#    
    
#import seaborn as sns
#import matplotlib.pyplot as plt
##sns.set()


#try:
#    import seaborn
#except ImportError:
#    print('seaborn nicht importierbar')  
    


#import locale 
#locale.setlocale(locale.LC_ALL, '') 

#from pandasklar.analyse  import type_info, values_info, val_first_valid


 

    
#############################################################################################
# ...........................................................................................
# Klasse values_info
# ...........................................................................................
#############################################################################################
    
class values_info:
    """ Liefert Informationen zu den Werten in einer Series. 
        Bsp.: i = values_info(series)     
              i.info()          # liefert alle Attribute
        übrigens liefert dieses hier ein ähnliches Ergebnis:
        df.agg([ 'count', 'nunique', 'min', 'max', nnan ]).transpose()
        * category_maxsize: Wie groß darf eine category werden, damit sie als datatype_suggest vorgeschlagen wird?
        * nanless_ints: Werden Numpys Integerklassen (die kein NaN kennen) als datatype_suggest vorgeschlagen?            
    """
    
    def __init__(self, data, category_maxsize=-1, nanless_ints=False):
        
        from pandasklar.analyse   import ntypes
        from pandasklar.type_info import type_info
        
        self.n         = int(data.shape[0])
        self.nnan      = int(data.isna().sum())
        self.ntypes    = ntypes(data)
        try:
            self.nunique   = int(data.nunique())
        except:
            self.nunique   = int(data.apply(str).nunique())
            if self.nnan > 0:
                self.nunique -= 1
        self.ndups     = self.n - self.nunique - self.nnan
        
        if is_numeric_dtype(data):
            self.vmin    = round(float(data.min()),2)
            self.vmax    = round(float(data.max()),2)
            self.vmean   = round(float(data.mean()),2)
            self.vsum    = round(float(data.sum()),2)
            try:
                self.vmedian = round(float(data.quantile(0.5)),2) 
            except:
                self.vmedian = np.NaN
                
        else: 
            try:
                self.vmin    = (data.min())
                self.vmax    = (data.max())
            except:
                self.vmin    = np.NaN
                self.vmax    = np.NaN                
            self.vmean   = np.NaN   
            self.vmedian = np.NaN
            self.vsum    = np.NaN
            
        # datatype_suggest
        self.datatype_suggest = ''
        if (self.ntypes == 1):
            self.type_info = type_info(data)  
            
            # int_versuche: Dtypes für Integer
            if self.nnan == 0  and nanless_ints:
                int_versuche = ['np.int8','np.uint8','np.int16','np.uint16','np.int32']
            else:
                int_versuche = ['pd.Int8','pd.Int16','pd.Int32']             
            
            # category oder string?
            if   (self.type_info.name_instance in ['str','_ElementUnicodeResult'])  and  self.type_info.name_short in ['object','string']:
                if (self.nunique < category_maxsize):
                    self.datatype_suggest = 'pd.category'  
                else:
                    self.datatype_suggest = 'pd.string'         
                    
            # float in int?        
            if self.type_info.name_short.startswith('float')   and   self.datatype_suggest=='': #and   self.nnan == 0:    
                try:
                    if (int(self.vmin)-self.vmin == 0)   and    (int(self.vmax)-self.vmax == 0)   and   ((data.astype('Int64')-data).sum() == 0): 
                        for versuch in int_versuche :
                            i = type_info(versuch)            
                            if i.xmin <= self.vmin   and   i.xmax >= self.vmax:
                                self.datatype_suggest = versuch
                                break                       
                except:
                    pass
                
            # float verkleinern?
            if self.type_info.name_short.startswith('float')   and  self.datatype_suggest=='':
                for versuch in ['np.float32'] : # float16 nehmen wir nicht, da kann man keinen Mittelwert berechnen
                    i = type_info(versuch)            
                    if i.xmin < self.vmin   and   i.xmax > self.vmax:
                        self.datatype_suggest = versuch
                        break
            
            # int verkleinern?
            # object in int?
            if (self.type_info.name_short.startswith(('int','uint','Int'))   and   self.datatype_suggest=='')   or   \
               ((self.type_info.name_instance =='int')   and   (self.type_info.name_short == 'object')   and   self.datatype_suggest==''):
                for versuch in int_versuche :
                    i = type_info(versuch)            
                    if i.xmin <= self.vmin   and   i.xmax >= self.vmax:
                        self.datatype_suggest = versuch
                        break   
                                            
            # wieder löschen wenn schon erfüllt
            if self.datatype_suggest == self.type_info.name:
                self.datatype_suggest = ''
        # Ende if (self.ntypes == 1)
        
        
    def info(self):
        """ Liefert alle Attribute"""
        result = dict(self.__dict__) # Kopie ziehen
        #del result['data']
        return result    
    
    
    