
import collections, warnings

from functools   import partial  

import numpy  as np
import pandas as pd

from pandas.api.types import is_string_dtype, is_numeric_dtype

import bpyth  as bpy

from .type_info    import type_info
from .values_info  import values_info

from .config       import Config
from .pandas       import dataframe, reset_index, drop_cols, rename_col, move_cols, quicksample, first_valid_value, last_valid_value
from .aggregate    import group_and_agg, top_values, top_values_count, most_freq_elt
from .scale        import scale
from .rank         import rank_without_group


    

# ==================================================================================================
# Load und save (gehört eigentlich nicht hierher, muss aber wg. changedatatype
# ==================================================================================================
# 
#
def load_pickle( filename, resetindex='AUTO', changedatatype=False, verbose=None ): 
    '''
    Convenient function to load a DataFrame from pickle file.
    Optional optimisation of datatypes. Verbose if wanted.
    resetindex = True:    Force reset_index
    resetindex = False:   No reset_index    
    resetindex = 'Auto':  (Standard) Automatic     
    changedatatype:       Should the datatypes get optimized?
    verbose:              True if messages are wanted.
    '''
    
    if verbose is None:
        verbose = Config.get('VERBOSE')  
        
    result = bpy.load_pickle(filename)
    if resetindex == True:
        result = result.reset_index()
    elif resetindex == 'AUTO':
        result = result.reset_index()    
        result = drop_cols(result, 'index')       
    
    if changedatatype:     
        result = change_datatype(result, verbose=verbose)

    if verbose:
        print(result.shape[0], 'rows loaded')  
        
    return result




def dump_pickle( df, filename, changedatatype=True, verbose=None ): 
    '''
    Convenient function to save a DataFrame to a pickle file.
    Optional optimisation of datatypes. Verbose if wanted.
    changedatatype:       Should the datatypes get optimized?
    verbose:              True if messages are wanted.    
    '''
    
    if verbose is None:
        verbose = Config.get('VERBOSE')  
        
    if changedatatype:       
        df = change_datatype(df, verbose=False)
    bpy.dump_pickle(df,filename)

    
    
    
#################################################################################
# ...............................................................................
# Spalten auf datatype untersuchen
# ...............................................................................
#################################################################################    
    

    
def mem_usage(data):
    """Returns the memory consumption of a Series or a DataFrame"""
    
    if isinstance(data, pd.Series): 
        result = data.memory_usage(index=False, deep=True)
        return bpy.human_readable_bytes(result)    
    
    elif isinstance(data, pd.DataFrame):
        result = data.memory_usage(index=False, deep=True).sum()
        return bpy.human_readable_bytes(result)    
        
    else:
        assert 'ERROR'    




# Alle class_info einer Series oder eines Index   
def analyse_datatype(data):
    """ 
    Returns a dict with info about the datatypes of a Series or Index and it's content
    """
    
    # Aufruf mit Index
    if isinstance(data, pd.Index): 
        series = data.to_series()
        series.name = '__index__'
        return analyse_datatype(series)    

    info = type_info(data)
    result = {
        'col_name': data.name,
        'datatype_instance': info.name_instance,  
        'datatype': info.name,        
        'datatype_short': info.name_short,        
        'is_numeric': is_numeric_dtype(data),                
        'is_string': is_string_dtype(data),  
        'is_hashable': info.is_hashable,
        'nan_allowed': info.nan_allowed,   
        'mem_usage': mem_usage(data),        
    }
        
    return result



# Alle class_info eines DataFrame
def analyse_datatypes(df, with_index=True):
    """
    Returns info about the datatypes and the mem_usage of the columns of a DataFrame.  
    """
    if isinstance(df, pd.Series): 
        return dataframe( analyse_datatype(df), verbose=False ) 
    
    # Kleine Probe reicht
    if df.shape[0] > 10:
        return analyse_datatypes(quicksample(df, 10), with_index=with_index)
    
    data  = [] 
    if with_index:
        data += [ analyse_datatype(df.index)]
    data     += [ analyse_datatype(df[col]) for col in df ]

    result = dataframe(data, verbose=False)

    # Zeilen-, Spalten- und Tabellenname
    result = result.rename_axis('col_no')    
    result = result.rename_axis('', axis='columns')
    return result



# Spaltennamen, die bestimmte Kriterien erfüllen
def col_names(df, only='', without='XXXXXX', as_list=True, query=None, sort=False ):
    """ 
    Selects column names based on analyse_cols. Useful to apply a method to specific columns of a DataFrame.
    * only:     Only column names whose datatype begins like this.
    * without:  Without column names whose datatype starts like this
    * as_list:  Output result as list (otherwise as DataFrame, useful for development and control)
    * query:    additional conditions
    * sort:     sorted by nunique
    datatypes are taken from analyse_datatypes, field datatype_short or datatype_instance.

    Example: treat all str columns with fillna
    cols = col_names(df, only='str', query='nnan > 0')
    df[cols] = df[cols].fillna('')        
    """
    if sort:
        df = sort_cols_by_nunique(df)
        
    if query or not as_list:
        info = analyse_cols(     df, with_index=False)      # komplette Analyse holen
    else:
        info = analyse_datatypes(df, with_index=False)      # nur datatypes analysieren      
    
    mask1 =  info.datatype_short.str.startswith(only)      |    info.datatype_instance.str.startswith(only)     |    info.datatype.str.startswith(only)
    mask2 = ~info.datatype_short.str.startswith(without)   &   ~info.datatype_instance.str.startswith(without)  &   ~info.datatype.str.startswith(without)
    result = info[mask1 & mask2]
    
    # Soll noch eine Query angewendet werden?
    if query:
        result = result.query(query)
    
    # Kompletten Dataframe zurückgeben?
    if not as_list:
        return result
    
    # als Liste zurückgeben
    return list(result.col_name)
    

    

    
    
def change_datatype(data, search=None, verbose=None, msg='', category_maxsize=-1, nanless_ints=False):
    """ 
    Converts the datatypes of a DataFrame or a Series.
    If used with a Series:    
    Similar behavior as pandas astype. But it also accepts
    sloppy class names like type_info knows.
    If no target datatype is specified, it will be selected automatically.
    If used with a DataFrame:
    Converts all datatypes automatically.                      

    * category_maxsize: How big can a category get to be suggested as datatype_suggest?
    * nanless_ints: Are numpy's integer classes (that don't know NaN) suggested as datatype_suggest?    
    """

    if verbose is None:
        verbose = Config.get('VERBOSE')   

    
    # Series
    if isinstance(data, pd.Series): 
                
        # vorgegebener Datatype
        if search: 
            i = type_info(search)              
            return data.astype(i.name_short)
            
        # vollautomatisch    
        else: 
            search = values_info(data, category_maxsize=category_maxsize).datatype_suggest  
            if search:
                if verbose:
                    print( '{:<20s} --> {:<10s}'.format(data.name, search) )                    
                return change_datatype(data, search=search, verbose=verbose, msg=msg)  
            else:
                return data

    # DataFrame
    elif isinstance(data, pd.DataFrame)   and not search:
            
        if verbose:
            print('change_datatype', msg )        
        result = data.apply(change_datatype, verbose=verbose   )
        if verbose:
            print('change_datatype','before:', mem_usage(data),'after:',mem_usage(result))
            print()
        return result
    
    else:
        assert 'ERROR'
    
# Synonym    
change_datatypes = change_datatype    
    
    
    
#################################################################################
# ...............................................................................
# Spalten auf values untersuchen
# ...............................................................................
#################################################################################     
    
    
def analyse_values(data, as_list=False, as_dict=False, sort=False, with_index=True, nanless_ints=False):
    """ 
    Returns statistical data for a DataFrame, a Series or an Index     
    """
    
    # Aufruf mit Index
    if isinstance(data, pd.Index): 
        series = data.to_series()
        series.name = '__index__'
        return analyse_values(series, as_list=as_list, as_dict=as_dict, nanless_ints=True)
    
    
    # Aufruf mit DataFrame
    if isinstance(data, pd.DataFrame): 
        info  = []
        if with_index:
            info += [ analyse_values(data.index, as_list=True) ]         
        info     += [ analyse_values(data[col],  as_list=True) for col in data ] 
        result = pd.DataFrame.from_records(info)

        # Zeilen-, Spalten- und Tabellenname
        result.columns = analyse_values(pd.Series(1)).index  # Sample abholen, Series ist dabei egal
        result = result.rename_axis('col_no')    
        result = result.rename_axis('', axis='columns')

        if sort:
            result = result.sort_values(['nunique','col_name'], ascending=[False,True])    
        return result        
    
        
    # Aufruf mit Series  
    assert isinstance(data, pd.Series)   
    info = values_info(data, nanless_ints=nanless_ints)
    result = [
        data.name,
        info.ntypes,        
        info.nunique,            
        info.nnan,
        info.ndups, 
        info.n,   
        info.vmin,      
        info.vmean,        
        info.vmedian,
        info.vmax,      
        info.vsum,            
        info.datatype_suggest,
    ]
    
    # Rückgabe als Liste
    if as_list:
        return result
    
    # In DataFrame wandeln
    result = pd.DataFrame(result)
    result['analyse'] = pd.Series(['col_name','ntypes', 'nunique','nnan','ndups','n','vmin','vmean','vmedian','vmax','vsum','datatype_suggest'])
    result = result.set_index('analyse')
    
    # Rückgabe als dict
    if as_dict:
        result = result.to_dict()[0]
        del result['col_name']
        return result    
    
    # Rückgabe als DataFrame
    return result    
    
    
    

# ==================================================================================================
# analyse_cols
# ==================================================================================================

def analyse_cols(df, sort=False, with_index=True):
    """ 
    Describes the datatypes and the content of a DataFrame.
    Merged info from analyse_datatypes and analyse_values.
    """   
        
    info1  = analyse_datatypes(df, with_index=with_index)
    info2  = analyse_values(   df, with_index=with_index)
    result = pd.merge(info1, info2)    
    result = move_cols(result, ['col_name','datatype_instance','datatype','datatype_short','datatype_suggest'])

    if sort:
        result = result.sort_values(['nunique','col_name'], ascending=[False,True])    
    return result    

            
        
    
##############################################################################################
# ............................................................................................
# NaN, NA, None 
# ............................................................................................
##############################################################################################
    

# ersetzt analyse_nans und nan_anz
def nnan(data, all=False): #, sum=False):
    """ 
    Count NaNs in Series or DataFrames.          
    For a DataFrame: Returns a list of all fields and the number of NaNs. 
    For a Series:  Returns the number of NaNs.    
    * all: Should the result include also the columns without NaNs (for a DataFrame)?
    * sum: Should the result include only the total sum (for a DataFrame)? Identical with nnan(df).sum()          
    """
    result = data.isnull().sum()
    if isinstance(data, pd.Series) or all:
        return result
    
    return result[result!=0]

  
    

# ersetzt assert_no_nans    
def any_nan(data, without=None):
    """ 
    Are there NaNs? Returns True or False.
    Works for Series or DataFrame.
    assert not any_nan(df) ensures that a dataframe does not contain NaNs.
    * without contains a field name or a list of field names that will be excluded.        
    """    
    if not without:
        return data.isnull().values.any()
    
    return drop_cols(data, without).isnull().values.any()
        



    
def nan_rows(df, col=''):   
    """ 
    Returns the rows of a DataFrame that are NaN in the specified column.  
    * col not specified: Returns all rows with NaNs in any column
    * col=0:             Returns all rows with NaN in the FIRST column with NaNs. 
    * col='city':        Returns all rows with NaNs in the column 'city'
    """
    
    # no column specified: 
    if col=='':
        return df[df.isna().any(axis=1)]
    
    # select the first column with NaNs
    elif col==0:
        col = nnan(df).head(1).index.to_list()[0]
    
    # return rows
    mask = df[col].isnull()
    return df[mask]
    
    


 
    
               
                
                
                
# ==============================================================================================
# analyse_freqs 
# ==============================================================================================

def analyse_freqs(data, cols=None, limits=[], splits=[], sort_count=True ):
    """ 
    Frequency analysis that includes a subordinate frequency analysis. 
    Provides e.g. the most important examples per case. Splits strings and lists.
    * cols:        Columns to which the analysis is to be applied. Name or list of names.
                   If one of the addressed columns is a list, it is exploded.
    * limits:      List of limits, corresponding to cols.
                   Example: [None,5] does not limit the frequencies of the first col 
                   but limits the data for the second col to the 5 most common values.
    * splits:      List of split characters, corresponding to cols.
                   Used to explode sentences into words
    * sort_count:  True =>  result shows the most common contents first
                   False => result is sorted by group
    See the jupyter notebook for examples.
    """
    
    # generelle Fehler abfangen
    if data is None:
        return 'Nothing to analyze'
    
    if not isinstance(data, pd.DataFrame):  
        data = dataframe(data, autotranspose=1)
        data.columns = ['item']
        return analyse_freqs( data, data.columns[0], limits=limits, splits=splits, sort_count=sort_count)

    if data.shape[0] == 0:
        return 'No rows'
    
    # Parameter cols 
    if type(cols) is str:
        cols = [cols] #let the command take a string or list    
        
    # Parameter limits 
    if type(limits) is int:
        limits = [limits] #let the command take an int or list  
    if len(limits) == 0:
        limits = [9999999]
    if not limits[0]:
        limits[0] = 9999999        
    if len(limits) < len(cols):
        limits += [None] * (len(cols)-len(limits)) 
    limits = [ 20 if l is None else l  for l in limits]
    #print('limits',limits)        
            
    # Parameter splits 
    if type(splits) is str:
        splits = [splits] #let the command take a string or list   
    if len(splits) < len(cols):
        splits += [None] * (len(cols)-len(splits)) 
    #print('splits',splits)
    
    # sortieren
    df = data.copy()
    #df = reset_index(data, keep_as='__afxidx').sort_values(cols + ['__afxidx']).reset_index(drop=True) 
    
    # splits realisieren
    for i, col in enumerate(cols):
        
        # Sätze zu Wortlisten
        if (splits[i] or splits[i]=='') and col:         
            df[col] = df[col].str.split(splits[i])    
    
        # Soll nach einem list-Feld gruppiert werden?
        if i == 0 and analyse_datatype(df[col])['datatype_instance'] in ['list']:
            df = df.explode(col)
    
    # Rumpf erstellen
    if len(cols) == 1:
        v = countgrid( df[cols[0]], sort=sort_count )  
        tops = v.head(limits[0])
        return tops    # nur den Rumpf zurückgeben
    else:
        v = countgrid( df[cols[0]], sort=sort_count )  #, style='top', stat=False)      
        tops = v.head(limits[0])    
    tops = drop_cols(tops,['graph'])
    
    mask        = (df[ cols[0] ].isin( tops[ cols[0] ]) )   
    feldliste   = [cols[0]] + [val                                         for val in cols[1:]   for i in (0, 1)] # erstes Feld einfach, die anderen doppelt
    funcliste   = ['group'] + [partial(top_values,       limit=lim)        if i==0 else \
                               partial(top_values_count, limit=lim)        for lim in limits[1:] for i in (0, 1)]     
    namensliste = [cols[0]] + [val                                         if i==0 else \
                               val + '_count'                              for val in cols[1:]   for i in (0, 1)]     
    
    #print(feldliste)
    #print(funcliste)
    #print(namensliste)
    
    result = group_and_agg( df[mask], feldliste, funcliste, namensliste, verbose=False )
    #return result
    return tops.merge(result)
    



                


# ==============================================================================================
# Kurze Einzelroutinen
# ==============================================================================================


#def val_first_valid(series):
#    """ 
#    Returns the first notna value of a series
#    """
#    
#    try:
#        result = series.loc[series.first_valid_index()]
#        if isinstance(result, pd.Series): # das liegt an nonunique Index
#            return result.iloc[0]
#        else:
#            return result
#    except:
#        return None
#
#
#def val_last_valid(series):
#    """
#    Returns the last notna value of a series
#    """
#    try:
#        result = series.loc[series.last_valid_index() ]
#        if isinstance(result, pd.Series): # das liegt an nonunique Index
#            return result.iloc[-1]
#        else:
#            return result        
#    except:
#        return None


    
def val_most(series):    
    """
    Returns the most frequent value of a series
    """
    mask = ~(series.isna().fillna(False))
    ohnenull = series[mask]   
    return ohnenull.value_counts().idxmax()


# wird von analyse_cols verwendet.
def nunique(series):
    """
    Returns the number of different values.
    Safe from TypeError: unhashable type: 'list'.
    """
    try:
        return series.nunique()
    except: 
        return series.apply(lambda x: str(x)).nunique()
    
    

def ntypes(series):
    '''
    Returns the number of different types. Examines all values of the series.
    NaN values are not counted.
    '''
    return series[series.notna()].map(type).nunique()





##############################################################################################
# ............................................................................................
# DataFrames 
# ............................................................................................
##############################################################################################



# sortiert die Spalten neu, vielfältigste Spalten zuerst 
def sort_cols_by_nunique(df, inaccurate_limit=100000):
    """
    Returns the DataFrame with reordered columns.
    It is sorted by nunique.
    * inaccurate_limit: If the dataframe is bigger than this, take a sample of this size
    """
    
    
    if df.shape[0] > inaccurate_limit:
        return sort_cols_by_nunique(quicksample(df, inaccurate_limit))
    
    spaltendict = dict()
    for col in df.columns:
        try:
            spaltendict[col] = df[col].nunique()
        except:
            spaltendict[col] = -1
            
    # sortieren
    spaltendict = {k: v for k, v in sorted(  spaltendict.items(), key=lambda item: item[1], reverse=True   )}

    #spalten = list(analyse_values(df, sort=True, with_index=False).col_name)
    df = df.reindex(spaltendict.keys(), axis=1)
    return df





# ==================================================================================================
# analyse_groups
# ==================================================================================================


def analyse_groups(df, exclude=[], tiefe_max=3):
    """
    Analyses a DataFrame for uniqueness and redundancy.
    Groups by many combinations of columns and counts the duplicates that are created in the process.
    Interpretation:
    0 dups => This combination of columns is unique
    Same number of dups than other combination of columns => Indication of redundancy
    
    """
    #return analyse_groups_worker(df, exclude, tiefe_max)
    try:
        return analyse_groups_worker(df, exclude, tiefe_max)
    except:
        # alles in String wandeln
        print('wandle in Strings...')
        df = df.applymap(lambda x: str(x))   
        return analyse_groups_worker(df, exclude, tiefe_max)


    
# ==================================================================================================
# same_but_different
# ==================================================================================================

def same_but_different(df, same, different, sort=True, return_mask=False):
    """ 
    Returns the rows of a DataFrame that are the same on the one hand and different on the other:
    They are the same in the fields named in same.
    And they differ in the field named in different.
    This is useful for analysing whether fields correlate 100% with each other or are independent.
    * same:       Array of column names.
    * different:  Single column name.  This column is used to search for differences.
    """
    mask = df.groupby(same)[different].transform('nunique') > 1
    if return_mask:
        return mask
    if sort:
        return df[mask].sort_values(same)
    return df[mask]    
        

    
    
    

#################################################################################################
# ...............................................................................................
# Lib-interne Hilfsfunktionen 
# ...............................................................................................
#################################################################################################



# Häufigkeitsverteilung für disjunkte Daten
# sort=True:  häufigstes zuerst
# sort=False: in der Reihenfolge der Eingabedaten
def countgrid( series, sort=True ):
    if series.shape[0] == 0:
        warnings.warn('No data')
        return 
    result = pd.DataFrame(   series.value_counts()   ).reset_index()
    countname   = series.name + '_count'
    percentname = series.name + '_percent'
    result.columns = [series.name, countname]
    result[percentname] = scale(result[countname], method='rel').round(3)*100 
    result['graph'] = result[percentname].apply( lambda x: int(x*0.5)*'#')
    if sort:
        return reset_index(result.sort_values(countname, ascending=False))
    return reset_index(result.sort_values(series.name))







# wird von analyse_groups verwendet.
# Komplettanalyse auf Eindeutigkeit und Redundanz
def analyse_groups_worker(df, exclude=[], tiefe_max=3):
    
    # Wichtigste Spalten zuerst
    df = sort_cols_by_nunique(df)
    
    # alle Teilmengen einer Menge
    # liefert list 
    from itertools import chain, combinations
    def powerlist(iterable, len_min=1, len_max=0):
        "powerlist('abcd',2, 2)  -->  [['a', 'b'], ['a', 'c'], ['a', 'd'], ['b', 'c'], ['b', 'd'], ['c', 'd']]"
        s = list(iterable)
        if len_max==0:
            len_max = len(s)
        result = chain.from_iterable(combinations(s, r) for r in range(len_min,len_max+1))
        result = [list(r) for r in result]
        return result    
    
    # spalten ermitteln
    spalten = [s for s in col_names(df, without='float') if not s in exclude]
    #print(spalten)
    
    # Leeres Ergebnis
    result_0 = []
    
    # Erst die Einerkombis, dann die Zweierkombis
    fertig = 0
    menge_eindeutiger_cols = set()
    
    for level in range(1,tiefe_max+1):
        
        if level > tiefe_max:
            break        
        
        # spaltenkombis
        spaltenkombis = powerlist(spalten,level,level)

        # Einzelne Zeile
        for sp in spaltenkombis:
            
            # sinnlose Kombination?
            p = len(set(sp).intersection(menge_eindeutiger_cols))
            if p > 0:
                continue
                
            a = sp                                      # columns
            b = level                                   # level = group_size
            c = df.duplicated(subset=sp).sum()          # dups_abs
            if c == 0 and len(sp) == 1:
                menge_eindeutiger_cols.add(sp[0]) 
            anfügen = (a, b, c)
            result_0.append( (a, b, c ) )         # alles anfügen            
        
        #print(menge_eindeutiger_cols)
        
        # Ergebnis aufbereiten
        result_1 = pd.DataFrame.from_records(result_0)
        result_1.columns = ['columns','level','dups_abs']
        
        # vorzeitigen Abbruch vormerken, sobald es spaltenkombis ohne dups gibt
        if len(menge_eindeutiger_cols) > 0  and fertig==0:
            tiefe_max = level +1 # Level noch zuende machen
            fertig = 1

    
    # Vollständiges Ergebnis endgültig aufbereiten    
    result_1 = result_1.sort_values(['dups_abs','level'])
    result_1 = reset_index(result_1)
    result_1['dups_rel'] = result_1.dups_abs / df.shape[0]     
    return result_1



    
def memory_consumption( iteration_of_objects, limit=10, use_rtype=True):
    '''
    Returns the memory consumption of Python objects.
    * iteration_of_objects: can be e.g. a DataFrame or just locals()
    * limit: Limits the output size
    * use_rtype: Use rtype instead of type?
    
    For the memory consumption of the biggest 10 local variables call:
    bpy.memory_consumption( locals() )
    '''    
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        result = bpy.memory_consumption( iteration_of_objects, limit=limit, use_rtype=use_rtype)
        result = dataframe(result, verbose=False)
        if use_rtype:
            result.columns = ['name','rtype','size']
        else:
            result.columns = ['name','type','size']
        return result


