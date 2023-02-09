
import warnings, copy
from functools import partial

import pandas as pd 
import numpy  as np
import bpyth  as bpy
   
from collections import Counter, defaultdict 

from .config     import Config
from .pandas     import dataframe, reset_index, drop_cols, rename_col, move_cols, drop_multiindex, quicksample, add_rows, move_rows
from .type_info  import type_info


###############################################################################################
# .............................................................................................
# Aggregation ...
# .............................................................................................
###############################################################################################


def group_and_agg(df, col_origins, col_funcs=None, col_names=None, dropna=True, optimize=False, verbose=None): 
    '''
    Groups and aggregates. Provides a user interface similar to that of MS Access.
    * col_origins: list of all columns to process
    * col_funcs:   list of all functions to apply to the columns above. 
                   Sometimes you have to use strings, sometimes function names.
                   'group' or '' = grouping. 
    * col_names:   list of new names for the result columns. Optional. Space = default name will be taken.
    * dropna:      Parameter for groupby.
    * optimize:    True to handle duplicated rows seperatly. 
                   Useful in situations with not many duplicated rows and slow functions in col_funcs.
    
    Example:
    df = pak.people()
    pak.group_and_agg( df, 
                       col_origins=['age_class', 'birthplace', 'first_name',  'age', 'age', 'first_name'],
                       col_funcs  =['group',     'group',      pak.agg_words, 'min', 'max', 'min'],
                 )    
    '''
    
    if verbose is None:
        verbose = Config.get('VERBOSE')      
    
    # Steuertabelle bauen
    steuer = dataframe((col_origins,  col_funcs,  col_names), verbose=False)
    steuer.columns =  ['col_origins','col_funcs','col_names']
    #return steuer
    steuer.col_funcs = steuer.col_funcs.fillna('')
    mask = (steuer['col_funcs'].str.len() == 0)   
    steuer.loc[mask,'col_funcs'] = 'group'   
    
    # cols_group: Die Spalten nach denen gruppiert werden soll
    mask = (steuer.col_funcs == 'group')
    cols_group = list(steuer[mask].col_origins)    
    
    g = steuer.groupby('col_origins')    
    s = g.agg(list) #.reset_index()
    s = s.drop(cols_group)
    d = s.to_dict()['col_funcs'] # dict für agg
    
    # name_new errät den Namen des Feldes, das die agg-Funktion ausspuckt
    steuer['name_new'] = steuer.col_origins 
    mask = steuer.col_funcs.apply(lambda x: isinstance(x, str))   &   (steuer.col_funcs != 'group')
    steuer.loc[mask,'name_new'] += '_' + steuer[mask].col_funcs
    mask = steuer.col_funcs.apply(lambda x: callable(x))  
    steuer.loc[mask,'name_new'] += '_' + steuer[mask].col_funcs.astype('str').str.split(' ').str[1].str.replace('>','').str.replace("'",'').str.replace('Series.','',regex=False).str.replace('.','',regex=False) 
    reihenfolge = list(steuer.name_new)

    if col_names:
        mask = (steuer['col_names'].str.len() == 0)
        steuer.loc[mask,'col_names'] = steuer[mask].name_new # Standardnamen übernehmen
        
    if optimize:
        if df.isnull().values.any():
            raise ValueError('Works without NaNs only')
        mask = df.duplicated(subset=cols_group, keep=False)
        df_unique, df_withdups = move_rows(df, mask, msg=None, verbose=False)
        if verbose:
            print( '{0} unique rows and {1} rows with duplicates'.format(df_unique.shape[0] , df_withdups.shape[0])  )
        assert df_unique.shape[0] + df_withdups.shape[0] == df.shape[0]
        result = group_and_agg( df_withdups, col_origins=col_origins, col_funcs=col_funcs, col_names=col_names, dropna=dropna, optimize=False, verbose=False)
        result = add_rows(result,df_unique[col_origins],verbose=False)
        result = reset_index(result)
        if verbose:
            n0 = df.shape[0]
            n1 = result.shape[0]        
            print( '{0} rows less, now {1} rows'.format(n0-n1, n1)  )        
        return result    
    
    # result bauen
    if d:    
        gruppiert = df.groupby(cols_group, dropna=dropna)
        result = gruppiert.agg(d) 
    else:
        result = df.groupby(cols_group, dropna=dropna, as_index=False).first()[cols_group]

    result = drop_multiindex(result, verbose=False)
    if len(cols_group) == 1:
        result = result.reset_index() # in anderen Fällen macht drop_multiindex das
    #print('reihenfolge',reihenfolge)
    result = move_cols(result, reihenfolge)   
    
    if verbose:
        n0 = df.shape[0]
        n1 = result.shape[0]        
        print( '{0} rows less, now {1} rows'.format(n0-n1, n1)  )
    
    
    if not col_names:      
        return result
    #return steuer
    #print(list(result.columns))
    #print(list(steuer['col_names']))
    result.columns = list(steuer['col_names'])
    return result
    





def most_freq_elt(series, inaccurate_limit=(10000,1000) ):  
    '''
    Aggregates a Series to the most frequent scalar element.
    Like Series.mode, but always returns a scalar.
    If two elements are equally frequent, just any one is returned .
    * inaccurate_limit: If the data is bigger than this, examine take a sample of this size.
      The first value is for hashable datatypes, the second for non_hashable datatypes.
      Set inaccurate_limit=(None,None) if you don't accept inaccuraties.

    Example:
    df = pak.people()
    df.groupby('age_class')['first_name'].apply(pak.most_freq_elt)    
    ''' 
    try:
        if type_info(series).is_hashable:
            if inaccurate_limit[0] is None:
                return series.mode().iloc[0]
            if series.shape[0] > inaccurate_limit[0]:
                #print('ungenau hashable')
                return quicksample(series, inaccurate_limit[0]).mode().iloc[0]
            return series.mode().iloc[0]
            
        else:
            if inaccurate_limit[1] is None:
                return np.NaN
            if series.shape[0] > inaccurate_limit[1]:
                #print('ungenau not hashable')
                result = quicksample(series, inaccurate_limit[1])
                with warnings.catch_warnings():
                    warnings.simplefilter(action='ignore', category=UserWarning)
                    return result.mode().iloc[0]
            return series.mode().iloc[0]
    except:
        return np.NaN
        # list(series.mode())[0]
    
    

def top_values(series, limit=3, count=False):
    '''
    Aggregates a Series to a list of the most frequent elements.
    Or, if there is only one, this single element.
    Can also return the counts of the most frequent elements.    
    * limit: limits the length of the resulting list
    * count: count=False shows the most frequent elements (default)
             cont=True   shows the corresponding counts of the elements
    Example:
    df = pak.people()
    df.groupby('age_class')['first_name'].apply(pak.top_values)
    
    Caution, does not work well with very long data sets.
    There are partials preconfigured for 3, 5, 10, 20, 100, 1000 elements,
    i.e. top_values_100 or top_values_count_20
    '''
    
    liste = series.to_list()    
    try:
        c = Counter(liste)
    except TypeError:
        flat_liste = bpy.flatten(liste)
        c = Counter(flat_liste)
        
    häufig = c.most_common(limit)
    if count:
        result = [cnt for word,cnt in häufig]
    else:
        result = [word for word,cnt in häufig]
        
    # lesbarer machen    
    if len(result) == 0:
        return ''
    if len(result) == 1:
        return result[0]  
    
    return result    

def top_values_count(series, limit=3):
    return top_values(series, limit=limit, count=True)

# partials für Aufruf in Aggregationen
# Werden hier nicht als Partial definiert, weil group_and_agg die Spaltennamen dann nicht erraten kann und durcheinander kommt

def top_values_3(series):    return top_values(series, limit=3)
def top_values_5(series):    return top_values(series, limit=5)
def top_values_10(series):   return top_values(series, limit=10)
def top_values_20(series):   return top_values(series, limit=20)
def top_values_100(series):  return top_values(series, limit=100)
def top_values_1000(series): return top_values(series, limit=1000)

def top_values_count_3(series):    return top_values_count(series, limit=3)
def top_values_count_5(series):    return top_values_count(series, limit=5)
def top_values_count_10(series):   return top_values_count(series, limit=10)
def top_values_count_20(series):   return top_values_count(series, limit=20)
def top_values_count_100(series):  return top_values_count(series, limit=100)
def top_values_count_1000(series): return top_values_count(series, limit=1000)



  


def agg_words(series):
    '''
    Aggregates a Series of strings to a long string.
    A space is always placed between the elements,
    the order is preserved.
    '''
    try:
        result = ' '.join(series.fillna(''))    
        result = bpy.superstrip(result)
    except:
        result = np.NaN
    return result

# alter Name
def agg_strings(series):
    warnings.warn('agg_strings wurde umbenannt in agg_words')
    return agg_words(series)





def agg_strings_nospace(series):
    '''
    Aggregates a Series of strings into one long string.
    No separators between the substrings.
    '''
    try:
        result = ''.join(series.fillna(''))
    except:
        result = np.NaN
    return result





def agg_words_nodup(series):
    '''
    Aggregates a Series of strings (e.g. signal words) to a long string.
    A space is always placed between the individual elements,
    the order is preserved,
    duplicates are removed.
    '''
    return agg_words(  series.str.split().explode().drop_duplicates()  )


# alter Name
def agg_strings_like_set(series):
    warnings.warn('agg_strings_like_set wurde umbenannt in agg_words_nodup')
    return agg_words_nodup(series)


    


def agg_to_list(series):
    '''
    Aggregates a Series to a list. 
    Normally this can also be done with a simple 'list', but in combination with transform this does not work.
    Then agg_to_list can be used as a substitute.
    '''
    result = [series.tolist()]*len(series)
    #result = list(s.astype(int))
    return result    



def agg_dicts(series):
    '''
    Aggregates a Series of dicts to a single dict.
    If a key occurs more than once, the value is overwritten.
    '''
    result = {key: value for d in series for key, value in d.items()}
    return result



def agg_dicts_2dd(series):
    '''
    Aggregates a Series of dicts to a single defaultdict(list).
    I.e. multiple keys are allowed. The values are always lists.  
    '''    
    result = defaultdict(list)
    for d in series:
        for key, value in d.items():
            if not value in result[key]:
                result[key].append(value)  
    return result



def agg_defaultdicts(series):
    '''
    Aggregates a Series of defaultdict(list) to a single defaultdict(list).
    '''    
    result = defaultdict(list)
    for d in series:
        for key in d:
            result[key].extend( d[key] )  
            result[key] = list(dict.fromkeys( result[key] )) # Dups entfernen
            
    return result





# ==================================================================================================
# dict
# ==================================================================================================
 

    

def explode_dict(df, col_dict, col_key='key', col_value='value', from_defaultdict=False):
    '''
    Like pandas explode, but for a dict.
    Turns dictionaries into two columns (key, value) and additional rows, if needed.
    * col_dict:          name of the column that contains the dict to explode
    * col_key:           name of the new column for the keys of the dict
    * col_value:         name of the new column for the values of the dict   
    * from_defaultdict:  Should an additional explode be executed? 
                         This can be useful for defaultdicts. Otherwise you get lists.
    '''
    if not df.index.is_unique:
        raise ValueError( 'index must be unique!')    
    result = pd.DataFrame([*df[col_dict]], df.index).stack().rename_axis([None,col_key]).reset_index(1, name=col_value)
    result = df.join(result)
    result = drop_cols(result,col_dict)
    
    if from_defaultdict:
        return result.explode(col_value)
    else:
        return result  
    
    

    
 
    
def implode_to_dict(df, cols_group=None, col_key=None, col_value=None, col_result=None, use_defaultdict=False):
    '''
    Reversal of explode_dict.
    Groups rows and turns two columns (key, value) into one dict. 
    * cols_group       is a string or list of names by which to group.
                       This affects the width of the results. None=no grouping.
    * col_key          is the name of the column containing the keys.
    * col_value        is the name of the column containing the values
    * col_result       is the name of the column containing the results
    * use_defaultdict  specifies whether the result is a dict or a defaultdict. Default: False.
                       With use_defaultdict=True, keys that occur more than once do not overwrite each other,
                       the values are always a list.
    '''
    
    if not col_key or not col_value or not col_result:
        raise
    
    # cols_group
    if not cols_group:                                                        # None
        alle = list(df.columns)
        cols_group = [c for c in alle if c not in [col_key, col_value]  ]
    elif cols_group  and not isinstance(cols_group, list):                     # Einzelner String
        cols_group = [cols_group]    
    
    # group_and_agg
    params1 = cols_group + [col_key, col_value]
    params2 = ['group'] * len(cols_group) + [list, list]
    dfg = group_and_agg(df, params1, params2, params1)
    
    # worker für die zip-Arbeit
    def worker_defaultdict(zeile, col_key, col_value, col_result):
        my_dict = defaultdict(list)
        for k, v in zip(zeile[col_key], zeile[col_value]):
            if not v in my_dict[k]:
                my_dict[k].append(v)          
        zeile[col_result] = my_dict        
        return zeile       
    
    def worker_dict(zeile, col_key, col_value, col_result):
        zeile[col_result] = dict(zip(zeile[col_key], zeile[col_value]))
        return zeile    
    
    # worker anwenden
    if use_defaultdict:
        result = dfg.apply(worker_defaultdict, axis=1, col_key=col_key, col_value=col_value, col_result=col_result)
    else:
        result = dfg.apply(worker_dict,        axis=1, col_key=col_key, col_value=col_value, col_result=col_result)    
    
    # Ende
    result = drop_cols(result,[col_key, col_value] )
    
    return result    
    
    
# implode_to_defaultdict
implode_to_defaultdict = partial(implode_to_dict, use_defaultdict=True)       

   
    
    
def cols_to_dict(df, col_dict='', cols_add=[], use_defaultdict=False, drop=True):
    '''
    Moves columns into a dict or defaultdict.
    This is 
    * col_dict:         name of the target column. Can be empty, but may already contain a dict or defaultdict. 
    * cols_add:         Columns to be packed.
    * use_defaultdict:  Should a defaultdict be used as data structure? Otherwise keys can only occur once.
    * drop:             Should the packed columns be dropped (>> move) or not (>> copy)?
    '''
    
    # Listen erzwingen
    if cols_add  and not isinstance(cols_add, list):
        cols_add = [cols_add]        
    
    
    def worker_dict(zeile, col_dict, cols_add):
        
        # Start
        zr = {}
        if col_dict in zeile.index:
            if isinstance(zeile[col_dict], dict):
                zr = dict(zeile[col_dict]) # copy                    
        
        # cols dazurechnen
        for col in cols_add:
            try:
                if (zeile[col] and not pd.isna(zeile[col]))  or  (zeile[col] == 0):
                    d = {col:zeile[col]}
                    zr.update(d)    
            except: # für Listen, die auch leer sein können
                d = {col:zeile[col]}
                zr.update(d)  
            
        zeile[col_dict] = zr
        return zeile
    
    
    
    
    def worker_defaultdict(zeile, col_dict, cols_add):
        
        # Start
        zr = defaultdict(list)
        if col_dict in zeile.index:
            if isinstance(zeile[col_dict], defaultdict):
                zr = copy.deepcopy(zeile[col_dict])
            elif isinstance(zeile[col_dict], dict):
                startvalue = { k:[v] for k,v in zeile[col_dict].items()}
                zr = defaultdict(list,startvalue)

        # cols dazurechnen
        for col in cols_add:
            if not pd.isna(zeile[col]):
                if zeile[col]  or  (zeile[col] == 0):
                    zr[col].append(zeile[col])     
            
        zeile[col_dict] = zr
        return zeile        
           
    
    
    # worker anwenden
    if use_defaultdict:
        result = df.apply(worker_defaultdict, axis=1, col_dict=col_dict, cols_add=cols_add )
    else:
        result = df.apply(worker_dict,        axis=1, col_dict=col_dict, cols_add=cols_add ) 
        
    
    # Ende
    if drop:
        result = drop_cols(result,cols_add )
    
    return result


def dict_to_defaultdict(df, col=''):
    '''
    Turns the dict in the given column into a defaultdict.
    '''
    # dirty trick
    return cols_to_dict( df, col_dict=col, use_defaultdict=True)  


def cols_to_defaultdict(df, col_dict='', cols_add=[], use_defaultdict=False, drop=True):
    '''
    Moves or copys columns into a defaultdict. 
    See cols_to_dict for more information.
    '''
    return cols_to_dict( df, col_dict=col_dict, cols_add=cols_add, use_defaultdict=True, drop=drop)  








