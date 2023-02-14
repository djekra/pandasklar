
## TODO: Unit tests!!!


import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype

from .config       import Config
from .pandas       import dataframe, drop_cols
from .analyse      import col_names



# ==================================================================================================
# compare_series
# ==================================================================================================

def compare_series(s, t, format='dict'):
    '''
    Compares the content of two Series.
    Returns several indicators of equality as dict or DataFrame:
        name:    same name    
        dtype:   nearly same dtype (Float32 == Float64)
        len:     same shape        
        nnan:    same number of NaNs   
        content: same content, ignoring index and sort
        sort:    same sort order, ignoring index
        eq:      same relations index->data, ignoring sort
        
    '''
    # result
    result = None
    
    # Only one series given
    if t is None:
        result = {'name': 'left_only', 'dtype':None,  'len': None, 'nnan':None, 'content':False, 'sort':None, 'eq':False,}   
    
    if s is None:
        result = {'name': 'right_only', 'dtype':None,  'len': None, 'nnan':None, 'content':False, 'sort':None, 'eq':False,}   
        
    if not result:
    
        # Vorbereitungen
        if not isinstance(s, pd.Series):   
            raise ValueError('First argument must be a pandas.Series')
        if not isinstance(t, pd.Series) and not t is None:   
            raise ValueError('Second argument must be a pandas.Series')        

        result = {'name': False, 'dtype':False,  'len': False, 'nnan':False, 'content':False, 'sort':False, 'eq':False,}

        # name
        if (s.name == t.name):  
            result['name'] = True        

        # len
        if len(s) == len(t):
            result['len'] = True           

        # nnan
        s_nnan = s.isnull().sum()
        t_nnan = t.isnull().sum()    
        if s_nnan == t_nnan:
            result['nnan'] = True 

        # dtype 
        if s.dtype == t.dtype:
            result['dtype'] = True   
        elif str(s.dtype) in ['Float64','Float32']  and  str(t.dtype) in ['Float64','Float32']:
            result['dtype'] = True               

        if result['len']  and  result['nnan'] :

            # Für weitere Vergleiche: nan ersetzen
            s = fillna(s,'special')
            t = fillna(t,'special')     

            # content
            try:
                if list(s).sort() == list(t).sort():
                    result['content'] = True    
            except:
                result['content'] = None               

            # sort
            if list(s) == list(t):
                result['sort'] = True    

            # eq
            if s.eq(t).all(skipna=False): 
                result['eq']      = True
                result['content'] = True    # falls es None war  
        
    # return result
    if format in ['dict','d']:
        return result
    result = pd.Series(result)
    if s is not None:
        result.name = s.name
    else:
        result.name = t.name        
    if format in ['series','Series','s']:  
        return result        
    if format in ['dataframe','DataFrame','Dataframe','df']:
        return dataframe(result,verbose=False)    

     



# ==================================================================================================
# Compare two DataFrames
# ==================================================================================================

def compare_dataframes(df1, df2, reset_index=False, format='df'):
    '''
    Compares the content of two DataFrames column by column.
    Returns several indicators of equality:
        name:    True, left_only or right_only. True means the column exists in both DataFrames.     
        dtype:   columns have same dtype     
        nnan:    columns have same number of NaNs   
        content: columns have same content, ignoring index and sort
        sort:    columns have same sort order, ignoring index
        eq:      columns have same relations index->data, ignoring sort    
    
    * reset_index: Set True to ignore index and sort order or the rows.    
    * format:      'DataFrame', 'Series', 'dict' or 'bool' (or abbreviations of this):
                   Output format. format='DataFrame' will return detailed information.
                   format='Series', 'dict' or 'bool' will return a summary.
    '''    
    # Vorbereitungen
    if not isinstance(df1, pd.DataFrame):   
        raise ValueError('First argument must be a pandas.DataFrame')
    if not isinstance(df2, pd.DataFrame):   
        raise ValueError('Second argument must be a pandas.DataFrame')      
    
    if reset_index:
        try:
            df1 = df1.sort_values(list(df1.columns)).reset_index(drop=True)    
            df2 = df2.sort_values(list(df2.columns)).reset_index(drop=True)           
        except:
            pass
    
    # cols_leftonly
    cols_leftonly = list(filter(lambda x:x not in list(df2.columns), list(df1.columns)))
    result_leftonly = [compare_series(df1[col], None, format='s') for col in cols_leftonly]
    result_leftonly = dataframe(result_leftonly, verbose=False).transpose()    
    
    # cols_intersection
    cols_intersection = list(filter(lambda x:x in list(df1.columns), list(df2.columns)))
    result_intersection = [compare_series(df1[col], df2[col], format='s') for col in cols_intersection]
    result_intersection = dataframe(result_intersection, verbose=False).transpose()
    
    # cols_rightonly
    cols_rightonly = list(filter(lambda x:x not in list(df1.columns), list(df2.columns)))
    result_rightonly = [compare_series(None, df2[col], format='s') for col in cols_rightonly]
    result_rightonly = dataframe(result_rightonly, verbose=False).transpose()     
    
    # concat to result
    result = pd.concat([result_leftonly, result_intersection, result_rightonly])
    result = drop_cols(result,['len'])
    
    if reset_index:
        result['sort'] = None
    
    # Total row
    result.loc['(Total)',:] = True    
    if list(df1.columns) != list(df2.columns):
        result.loc['(Total)','sort'] = False         
    result.loc['(Total)','name']     = list(result['name'].unique()) == [True]    
    result.loc['(Total)','dtype']    = result['dtype'].all(skipna=False)    
    result.loc['(Total)','nnan']     = result['nnan'].all(skipna=False)       
    result.loc['(Total)','content']  = result['content'].all(skipna=False)    
    result.loc['(Total)','sort']     = result['sort'].all(skipna=False)    
    result.loc['(Total)','eq']       = result['eq'].all(skipna=False)        
    
    # return result as DataFrame
    if format in ['dataframe','DataFrame','Dataframe','df']:
        return result       
    
    # return only the Total row as Series
    if format in ['series','Series','s']:  
        return result.loc['(Total)',:]        
    
    # return only the Total row as dict
    if format in ['dict','d']:
        return dict(result.loc['(Total)',:] )

    # return only the Total eq as bool    
    if format in ['bool','b']:
        return result.loc['(Total)','eq'] 




def check_equal(obj1, obj2, reset_index=False ):
    '''
    Compares the content of two DataFrames column by column.
    Two DataFrames are equal, if 
    * they have the same shape
    * they have the same column names
    * and compare_dataframes(format='bool') is True
    '''
    
    # Vorbereitungen
    if isinstance(obj1, pd.Series) and  isinstance(obj2, pd.Series):
        return obj1.eq(obj2).all(skipna=False)
    
    # Unterschiedliche shape?
    if obj1.shape != obj2.shape:
        return False
    
    # Unterschiedliche Spaltennamen?
    if set(obj1.columns) != set(obj2.columns):     
        return False 
    
    # use compare_dataframes
    return compare_dataframes(obj1,obj2, reset_index=reset_index, format='bool')
    

    
    
def compare_col_dtype(df1, df2):
    '''
    Returns the column names of two DataFrames whose dtype differs.
    '''
    result = []
    for col in df1.columns:
        try:        
            ist_gleich = (df1[col].dtype == df2[col].dtype)
            if not ist_gleich:
                result += [col]
        except:
            result += [col]
    return result    

    
    

def get_different_rows(df1, df2, indicator=True):
    """ 
    Returns the rows of two DataFrames that differ. 
    Additional or missing columns are ignored.
    Float columns may cause mistakes.
    """
    
    # Spalten hashable machen
    df1 = df1.copy()
    df2 = df2.copy()  
    
    cols = col_names(df1, query='not is_hashable')
    if cols:
        df1[cols] = df1[cols].apply(lambda x: str(x))
        
    cols = col_names(df2, query='not is_hashable')  
    if cols:
        df2[cols]   = df2[cols].apply(lambda x: str(x))    
          
    df1 = df1.sort_values(list(df1.columns)).reset_index(drop=True)    
    df2 = df2.sort_values(list(df2.columns)).reset_index(drop=True)                  
    
    # get_different_rows
    merged_df = df1.merge(df2, indicator=True, how='outer')
    mask = (merged_df['_merge'] != 'both') 
    result = merged_df[mask] # alle geänderten Zeilen
    if not indicator:
        result = result.drop('_merge', axis=1)
    return result
    
    




    
    
# ==================================================================================================
# Hilfsroutinen
# ==================================================================================================


def fillna(ser, method='zero'):
    '''
    Automatic fillna. 
    method='zero':    Fills with 0 and empty String
    method='special': Fills with -777 and '∅'   
    '''
    if method=='zero':
        if is_string_dtype(ser):
            return ser.fillna('')
        elif is_numeric_dtype(ser):
            return ser.fillna(0)
        else:
            return ser.fillna('')
        
    elif method=='special':
        if is_string_dtype(ser):
            return ser.fillna('∅')
        elif is_numeric_dtype(ser):
            return ser.fillna(-777)
        else:
            return ser.fillna('∅')    

    else:
        raise ValueError('method must be in ["zero","special"]')  
    

