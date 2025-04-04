
import pandas as pd
import bpyth  as bpy
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_float_dtype, is_integer_dtype

from .config       import Config

from .dataframe    import dataframe
from .pandas       import drop_cols
from .analyse      import col_names



# ==================================================================================================
# compare_series
# ==================================================================================================

def compare_series(s, t, format='dict', decimals=None):
    '''
**Compares two Pandas Series and returns indicators of equality.**

This function compares two Pandas Series and provides detailed information about their similarities and differences.
It checks for equality in various aspects, including name, data type, length, number of NaNs, content, sort order, and index-data relations.

**Args:**
- `s` (`pd.Series`): The first Pandas Series.
- `t` (`pd.Series`): The second Pandas Series.
- `format` (`str`, optional): Output format for the comparison results.
  - `'dict'` or `'d'`: Returns a dictionary.
  - `'series'` or `'Series'` or `'s'`: Returns a Pandas Series.
  - `'dataframe'` or `'DataFrame'` or `'Dataframe'` or `'df'`: Returns a Pandas DataFrame.
  Defaults to `'dict'`.
- `decimals` (`int`, optional): The number of decimal places to round to when comparing numeric values.
  If `None`, no rounding is performed. Defaults to `None`.

**Returns:**
`dict`, `pd.Series`, or `pd.DataFrame`: Comparison results, depending on the `'format'` parameter.
The output contains the following keys/indices:
- `'name'`: `True` if the series have the same name
- `'dtype'`: `True` if the series have the same dtype (or both are `float32`/`float64`), `False` otherwise.
- `'len'`: `True` if the series have the same length, `False` otherwise.
- `'nnan'`: `True` if the series have the same number of NaNs, `False` otherwise.
- `'nan_pat'`: `True` if the series have the same pattern of NaNs, `False` otherwise.
- `'content'`: `True` if the series have the same content (ignoring index, sort and NaNs), `False` otherwise.
  - For numeric series: If `decimals` is not `None`, values are rounded before comparison.
- `'sort'`: `True` if the series have the same sort order (ignoring index), `False` otherwise.
- `'eq'`: `True` if the series have the same index-data relations (ignoring sort), `False` otherwise.

**Examples:**
```python
>>> s1 = pd.Series([1, 2, 3], name='numbers')
>>> s2 = pd.Series([1, 2, 3], name='numbers')
>>> compare_series(s1, s2, format='dict')
{'name': True, 'dtype': True, 'len': True, 'nnan': True, 'nan_pat': True, 'content': True, 'sort': True, 'eq': True}

>>> s3 = pd.Series([1.1, 2.2, np.nan], name='floats')
>>> s4 = pd.Series([1.1, 2.2, np.nan], name='floats')
>>> compare_series(s3, s4, format='series', decimals=1)
name        True
dtype       True
len         True
nnan        True
nan_pat     True
content     True
sort        True
eq          True
Name: floats, dtype: object

>>> s5 = pd.Series([1, 2, 3], name='numbers')
>>> s6 = pd.Series([3, 2, 1], name='numbers')
>>> compare_series(s5, s6, format='df')
         name  dtype   len  nnan  nan_pat content   sort     eq
numbers  True   True  True  True  True    True  False   True
    '''
    # result
    result = None
    # Only one series given
    if t is None:
        result = {'name': 'left_only', 'dtype':None,  'len': None, 'nnan':None, 'nan_pat':None, 'content':False, 'sort':None, 'eq':False,}
    
    if s is None:
        result = {'name': 'right_only', 'dtype':None,  'len': None, 'nnan':None, 'nan_pat':None, 'content':False, 'sort':None, 'eq':False,}
        
    if not result:
    
        # Vorbereitungen
        if not isinstance(s, pd.Series):   
            raise ValueError('First argument must be a pandas.Series')
        if not isinstance(t, pd.Series) and not t is None:   
            raise ValueError('Second argument must be a pandas.Series')        

        result = {'name': False, 'dtype':False,  'len': False, 'nnan':False, 'nan_pat':False, 'content':False, 'sort':False, 'eq':False,}

        # name ---------------------------------------------------------------------------------------------------------
        if (s.name == t.name):  
            result['name'] = True        

        # len ----------------------------------------------------------------------------------------------------------
        if len(s) == len(t):
            result['len'] = True           

        # nnan ---------------------------------------------------------------------------------------------------------
        s_nnan = s.isnull().sum()
        t_nnan = t.isnull().sum()    
        if s_nnan == t_nnan:
            result['nnan'] = True 

        # dtype --------------------------------------------------------------------------------------------------------
        if s.dtype == t.dtype:
            result['dtype'] = True   
        elif str(s.dtype) in ['Float64','Float32']  and  str(t.dtype) in ['Float64','Float32']:
            result['dtype'] = True

        if result['len']  and  result['nnan'] :

            # nan_pat --------------------------------------------------------------------------------------------------
            if s_nnan == 0 and t_nnan == 0:
                result['nan_pat'] = True
            else:
                nan_mask_s = s.reset_index(drop=True).isna()
                nan_mask_t = t.reset_index(drop=True).isna()
                if nan_mask_s.eq(nan_mask_t).all(skipna=True):
                    result['nan_pat'] = True


            # Im Folgenden: Nur Werte vergleichen, bei denen in beiden Series kein NaN ist
            nan_mask_s = s.isna()
            nan_mask_t = t.isna()
            s = s[~nan_mask_s]
            t = t[~nan_mask_t]


            # eq -------------------------------------------------------------------------------------------------------
            try:
                if s.eq(t).all(skipna=False):
                    result['eq']      = True
                    result['content'] = True
            except:
                result['eq'] = False


            # content --------------------------------------------------------------------------------------------------

            if not result['content']:

                try:
                    if sorted(list(s)) == sorted(list(t)):
                        result['content'] = True

                    elif is_numeric_dtype(s) and decimals is not None:
                        if sorted(list(s.round(decimals))) == sorted(list(t.round(decimals))):
                            result['content'] = True

                    elif sorted(s.astype(str)) == sorted(t.astype(str)) :
                            result['content'] = True

                except:
                    result['content'] = None

            # Wenn decimals nicht None ist, auch noch mit decimals=None vergleichen und ggf. überschreiben
            if decimals is not None:
                result_without_decimals = compare_series(s, t, format='dict', decimals=None)
                if result_without_decimals['content']:
                    result['content'] = True


            # sort -----------------------------------------------------------------------------------------------------
            try:
                if list(s) == list(t):
                    result['sort'] = True
                else:
                    index_s = s.reset_index(drop=True).sort_values().index.to_list()
                    index_t = t.reset_index(drop=True).sort_values().index.to_list()
                    if index_s == index_t:
                        result['sort'] = True
            except:
                try:
                    ss = s.astype(str)
                    tt = t.astype(str)
                    if list(ss) == list(tt):
                        result['sort'] = True
                    else:
                        index_s = ss.reset_index(drop=True).sort_values().index.to_list()
                        index_t = tt.reset_index(drop=True).sort_values().index.to_list()
                        if index_s == index_t:
                            result['sort'] = True
                except:
                    result['sort'] = None


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

def compare_dataframes(df1, df2, format='df', decimals=None):
    '''
    **Compares two DataFrames column by column and returns indicators of equality.**

    This function compares two Pandas DataFrames and provides detailed information about their similarities and differences.
    It checks for equality in various aspects for each column, including name, data type, number of NaNs, content, sort order, and index-data relations.
    It also provides a summary row (`'(Total)'`) indicating the overall equality of the DataFrames.

    **Args:**
    - `df1` (`pd.DataFrame`): The first DataFrame.
    - `df2` (`pd.DataFrame`): The second DataFrame.
    - `format` (`str`, optional): Output format for the comparison results.
      - `'dataframe'` or `'DataFrame'` or `'Dataframe'` or `'df'`: Returns a Pandas DataFrame.
      - `'series'` or `'Series'` or `'s'`: Returns a Pandas Series (only the `'(Total)'` row).
      - `'dict'` or `'d'`: Returns a dictionary (only the `'(Total)'` row).
      - `'bool'` or `'b'`: Returns a boolean (only the `'eq'` value of the `'(Total)'` row).
      Defaults to `'df'`.
    - `decimals` (`int`, optional): The number of decimal places to round to when comparing numeric values.
      If `None`, no rounding is performed. Defaults to `None`.

    **Returns:**
    `pd.DataFrame`, `pd.Series`, `dict`, or `bool`: Comparison results, depending on the `'format'` parameter.
    The output contains the following columns/keys:
    - `'name'`: `True` if columns exist in both DataFrames, `'left_only'` if the column is only in `df1`, `'right_only'` if the column is only in `df2`.
    - `'dtype'`: `True` if columns have the same dtype (or both are `float32`/`float64`), `False` otherwise.
    - `'nnan'`: `True` if columns have the same number of NaNs, `False` otherwise.
    - `'nan_pat'`: `True` if the columns have the same pattern of NaNs, `False` otherwise.
    - `'content'`: `True` if columns have the same content (ignoring index and sort), `False` otherwise.
      - For numeric columns: If `decimals` is not `None`, values are rounded before comparison.
    - `'sort'`: `True` if columns have the same sort order (ignoring index), `False` otherwise.
    - `'eq'`: `True` if columns have the same index-data relations (ignoring sort), `False` otherwise.
    - `'(Total)'`: A summary row indicating the overall equality of the DataFrames.

    **Examples:**
    ```python
    >>> df1 = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
    >>> df2 = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
    >>> compare_dataframes(df1, df2, format='df')
             name  dtype  nnan  nan_pat content  sort    eq
    A        True   True  True  True    True  True  True
    B        True   True  True  True    True  True  True
    (Total)  True   True  True  True    True  True  True
    '''
    # Vorbereitungen
    if not isinstance(df1, pd.DataFrame) or not isinstance(df2, pd.DataFrame):
        raise ValueError('Both inputs must be pandas DataFrames')
    if df1 is None or df2 is None:
        raise ValueError('Both inputs must not be None')

    # Kopien erstellen
    df1 = df1.copy()
    df2 = df2.copy()

    # Leere DataFrames abfangen
    if len(df1) == 0 and len(df2) == 0:
        result = pd.DataFrame(index=['(Total)'], columns=['name', 'dtype', 'nnan', 'nan_pat', 'content', 'sort', 'eq'])
        result.loc['(Total)'] = [True, True, True, True, True, True, True]
        if format in ['dataframe','DataFrame','Dataframe','df']:
            return result
        elif format in ['series', 'Series', 's']:
                return result.iloc[0]
        elif format in ['dict', 'd']:
            return result.iloc[0].to_dict()
        elif format in ['bool', 'b']:
            return True
        else:
            raise ValueError('Invalid format')

    # cols_leftonly
    cols_leftonly = list(filter(lambda x:x not in list(df2.columns), list(df1.columns)))
    result_leftonly = [compare_series(df1[col], None, format='s', decimals=decimals) for col in cols_leftonly]
    result_leftonly = dataframe(result_leftonly, verbose=False, framework='pandas').transpose()
    
    # cols_intersection
    cols_intersection = list(filter(lambda x:x in list(df1.columns), list(df2.columns)))
    result_intersection = []
    for col in cols_intersection:
        result_intersection += [compare_series(df1[col], df2[col], format='s', decimals=decimals)]
    result_intersection = dataframe(result_intersection, verbose=False, framework='pandas').transpose()
    
    # cols_rightonly
    cols_rightonly = list(filter(lambda x:x not in list(df1.columns), list(df2.columns)))
    result_rightonly = [compare_series(None, df2[col], format='s', decimals=decimals) for col in cols_rightonly]
    result_rightonly = dataframe(result_rightonly, verbose=False, framework='pandas').transpose()
    
    # concat to result
    result = pd.concat([result_leftonly, result_intersection, result_rightonly])
    result = drop_cols(result,['len'])
    
    # Total row
    result.loc['(Total)',:] = True    
    if list(df1.columns) != list(df2.columns):
        result.loc['(Total)','sort'] = False         
    result.loc['(Total)','name']     = list(result['name'].unique()) == [True]    
    result.loc['(Total)','dtype']    = result['dtype'].all(skipna=False)    
    result.loc['(Total)','nnan']     = result['nnan'].all(skipna=False)
    result.loc['(Total)','nan_pat']  = result['nan_pat'].all(skipna=False)
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

    raise ValueError('Invalid format')




def check_equal(obj1, obj2):
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
    return compare_dataframes(obj1,obj2, format='bool')
    

    
    
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


def get_different_rows(df1, df2, use_index=True, indicator=True):
    """
    Returns the rows of two DataFrames that differ.

    This function compares two DataFrames and returns the rows that are different.
    It offers two modes of comparison, controlled by the `use_index` parameter:

    - **`use_index=True` (Index-based comparison):**
      The DataFrames are compared row by row, based on their index.
      Rows with the same index but different content are returned.
      Rows that exist only in one DataFrame are also returned.

    - **`use_index=False` (Content-based comparison):**
      The indexes of the DataFrames are completely ignored.
      Rows are compared based solely on their content (based on the hashable columns).
      Rows that exist in one DataFrame but not in the other (regardless of index) are returned.
      Duplicate rows are considered as one.

    Additional or missing columns are ignored.
    Float columns may cause mistakes due to floating-point precision issues.

    Args:
        df1 (pd.DataFrame): The first DataFrame.
        df2 (pd.DataFrame): The second DataFrame.
        use_index (bool, optional): Determines the comparison mode.
            If True, compares rows based on index.
            If False, compares rows based on content, ignoring index.
            Defaults to True.
        indicator (bool, optional): If True, adds a '_merge' column to the result
            indicating whether a row is 'left_only', 'right_only', or 'both'.
            Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the different rows.
            If `indicator=True`, the result has a '_merge' column.
            If no differences are found, an empty DataFrame is returned.
    """
    # Kopien erstellen
    df1 = df1.copy()
    df2 = df2.copy()

    # only hashable_cols
    hashable_cols = col_names(df1, query='is_hashable')
    df1 = df1[hashable_cols]
    hashable_cols = col_names(df2, query='is_hashable')
    df2 = df2[hashable_cols]

    # only common_cols
    common_cols = list(set(df1.columns) & set(df2.columns))
    common_cols = bpy.sort_by_priority_list(common_cols, list(df1.columns) + list(df2.columns))
    df1 = df1[common_cols]
    df2 = df2[common_cols]

    if not use_index:
        # Content-basierter Vergleich, Indexe ignorieren

        # DataFrames sortieren und Index zurücksetzen
        df1 = df1.sort_values(common_cols).reset_index(drop=True)
        df2 = df2.sort_values(common_cols).reset_index(drop=True)

        # get_different_rows
        merged_df = df1.merge(df2, indicator=True, how='outer', on=common_cols)
        mask = (merged_df['_merge'] != 'both')
        result = merged_df[mask]  # alle geänderten Zeilen

        # Indicator entfernen, falls nicht gewünscht
        if not indicator:
            result = result.drop('_merge', axis=1)

        return result

    else:
        # Index-basierter Vergleich

        # DataFrames nach Index sortieren

        df1 = df1.sort_index()
        df2 = df2.sort_index()

        # Unterschiede finden

        # Indexe, die nur in df1 oder df2 vorhanden sind
        left_only_indices = df1.index.difference(df2.index)
        right_only_indices = df2.index.difference(df1.index)

        # Zeilen, die nur in df1 oder df2 vorhanden sind
        left_only = df1.loc[left_only_indices]
        left_only['_merge'] = 'left_only'
        right_only = df2.loc[right_only_indices]
        right_only['_merge'] = 'right_only'

        # Gemeinsame Indexe
        common_indices = df1.index.intersection(df2.index)
        df1_common = df1.loc[common_indices]
        df2_common = df2.loc[common_indices]

        # Unterschiede in den gemeinsamen Indexen
        differing_rows = df1_common.compare(df2_common).index
        left_only_common = df1_common.loc[differing_rows]
        left_only_common['_merge'] = 'left_only'
        right_only_common = df2_common.loc[differing_rows]
        right_only_common['_merge'] = 'right_only'

        # Ergebnisse zusammenfügen
        result = pd.concat([left_only, right_only, left_only_common, right_only_common])


    # Indicator entfernen, falls nicht gewünscht
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
        elif is_float_dtype(ser):
            return ser.fillna(0.0)
        elif is_numeric_dtype(ser):
            return ser.fillna(0)
        else:
            return ser.fillna('')
        
    elif method=='special':
        if is_string_dtype(ser):
            return ser.fillna('∅')
        elif is_float_dtype(ser):
            return ser.fillna(-77.77)
        elif is_numeric_dtype(ser):
            try:
                return ser.fillna(-777)
            except:
                return ser
        else:
            return ser.fillna('∅')


    else:
        raise ValueError('method must be in ["zero","special"]')
    

