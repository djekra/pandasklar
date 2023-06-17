import warnings
import numpy  as np
import pandas as pd

from pandas.api.types import is_string_dtype, is_numeric_dtype

from .config       import Config
from .type_info    import type_info



 

    
#############################################################################################
# ...........................................................................................
# Class values_info
# ...........................................................................................
#############################################################################################
  
# this one returns a similar result:
# df.agg([ 'count', 'nunique', 'min', 'max', nnan ]).transpose()    
#
class values_info:
    """    
    Provides information on the values in a series. 
    Ex: i = values_info(series)     
          i.info() # returns all attributes

    * category_maxsize: How big can a category get to be suggested as datatype_suggest?
    * nanless_ints: Are numpy's integer classes (that don't know NaN) suggested as datatype_suggest?     
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
            data_dropna = data.dropna()
            try:
                self.vmin    = data_dropna.min()
                self.vmax    = data_dropna.max()
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

        # datatype_identified
        self.datatype_identified = identify_datatype(data) 
        if self.datatype_identified == '' and self.datatype_suggest.endswith('string'):
            self.datatype_identified = 'string'
        if self.datatype_identified.startswith('float'):
            self.datatype_identified = 'float'  
        if self.datatype_identified.startswith('int'):
            self.datatype_identified = 'int'                 
        
    def info(self):
        """Returns all attributes"""
        result = dict(self.__dict__) # Kopie ziehen
        #del result['data']
        return result    



def identify_datatype(series):
    '''    
    # Example usage
    data = pd.Series([1, 2, '3', pd.Timestamp('2022-01-01'), 'text'])
    data_type = identify_data_type(data)
    print(data_type)
    '''
    result = type_info(series).name_short 
    if result == 'object':
        result = ''

    filtered_series = series.dropna()
    str_series = filtered_series.astype(str)

    try:
        # Try converting to float
        float_series = pd.to_numeric(str_series, errors='raise').astype(float)
        if (float_series == filtered_series).all():
            result = 'float'
    except:
        pass
        
    try:
        # Try converting to int
        int_series = pd.to_numeric(str_series, errors='raise').astype(int)
        diff_sum = (int_series - filtered_series).abs().sum()
        if diff_sum < 0.00001:
            result = 'int'    
    except:
        pass

    try:
        # Try converting to datetime
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            datetime_series = pd.to_datetime(str_series, errors='raise')
        if (datetime_series == filtered_series).all():
            result = 'datetime'
    except:
        pass

    try:
        # Count comma
        if result in ['','string']:
            kommas = str_series.str.count(',').sum()
            alle = str_series.str.len().sum() 
            duplicate_count = series.duplicated().sum()
            total_count = len(series)
            if kommas / alle > 0.05:
                result = 'list'
            if duplicate_count / total_count > 0.2   and kommas / alle > 0.02:
                result = 'list'
            if duplicate_count / total_count > 0.4   and kommas / alle > 0.01:
                result = 'list'                
    except:
        pass        

    return result    
    
    