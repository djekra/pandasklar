
import warnings, copy

import pandas as pd 
import numpy  as np
import bpyth  as bpy
   
from functools   import partial    
from collections import Counter, defaultdict 

from .config       import Config
from .rank         import rank


#################################################################################################
# ...............................................................................................
# Einzelner Dataframe
# ...............................................................................................
#################################################################################################




    
# ==================================================================================================
# Create
# ==================================================================================================
# 
#
def dataframe(inp, test=False, autotranspose=False, verbose=None):
    """ 
    Converts multidimensional objects into dataframes.
    Dictionaries and Tuples are interpreted column-wise, Lists and Counters by rows.
    (Exception: A list of Series is also interpreted column-wise, like a tuple of Series.)
    * test:           Do additional test if the shape is ok
    * autotranspose:  False => no autotranspose
                      n     => If the result is n times wider than long, it is transposed.  
    """
    
    if verbose is None:
        verbose = Config.get('VERBOSE')   
        
    if isinstance(inp, pd.Series):
        return pd.DataFrame(inp)
    
    def do_test(inp, result, test, verbose, inp_rtype, inp_shape, gedreht):
        if verbose:
            print('rotated='+str(gedreht) + ' Output rtype=' + str(bpy.rtype(result)), 'shape=' + str(bpy.shape(result)))
        if not test:
            return True
        
        # Nur die ersten beiden Dimensionen interessieren
        inp_shape = inp_shape[:2]
        
        # Erste Dimension ggf. ergänzen
        if len(inp_shape) == 1:
            inp_shape = (1,inp_shape[0])          
            
        # spaltenweise interpretiert?
        elif gedreht:
            inp_shape = (inp_shape[1],inp_shape[0])

        
        if inp_shape != bpy.shape(result)[:2]:
            if verbose:
                print('Shape does not fit')            
            return False
        return True
    
    
    # Spaltennamen
    def cols_benennen(result):
        if result.shape[1] > 52: # zu breit, kann man nicht umbenennen
            return result
        # gibt es Duplikate in den Spaltennamen?                       oder sind die Spalten rein numerisch?
        if (len(list(result.columns))  !=  len(set(result.columns)))   or   isinstance(result.columns, pd.RangeIndex):     
            result.columns = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')[0:result.shape[1]] 
        return result
    
    # Analyse
    inp_rtype = bpy.rtype(inp)
    gedreht = False # normale Orientierung
    try:
        inp_shape = bpy.shape(inp) 
    except:
        inp_shape = (-77,-77)
    if verbose:
        print('Input rtype=' + str(inp_rtype), 'shape=' + str(inp_shape))
    
    # tuple of Series oder list of Series: 
    # in dict wandeln und gedreht vormerken 
    if len(inp_rtype)>=2  and  inp_rtype[0] in ['list','tuple']  and  inp_rtype[1] == 'Series':
        sp = tuple(a.name for a in inp) 
        if (len(list(sp))  !=  len(set(sp))): # dups?    
            sp = tuple('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')[0:len(inp)]        
        inp = dict(zip(sp, inp)) 
        gedreht = True
        
    # tuple: 
    # in dict wandeln und gedreht vormerken    
    elif inp_rtype[0] == 'tuple':
        sp = tuple('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')[0:len(inp)] 
        inp = dict(zip(sp, inp))  
        gedreht = True
        
    # dict: 
    # gedreht vormerken   
    elif inp_rtype[0] in ['dict']:
        gedreht = True        
    
    # 0 Dimensionen: Leeres DataFrame
    if len(inp_shape) == 0:
        result = pd.DataFrame()
        assert do_test(inp, result, test=test, verbose=verbose, inp_rtype=inp_rtype, inp_shape=inp_shape, gedreht=gedreht)
        return result
    
    # 1 Dimensionen: Eine Zeile
    if len(inp_shape) == 1:
        result = pd.DataFrame([inp])
        assert do_test([inp], result, test=test, verbose=verbose,inp_rtype=inp_rtype, inp_shape=inp_shape, gedreht=gedreht)
        if autotranspose  and  result.shape[0]*autotranspose < result.shape[1]:  # Hundertmal breiter als lang: transpose
            result = result.transpose()    
        result = cols_benennen(result)             
        return result    
    
    if isinstance(inp, dict): 
        result = pd.DataFrame.from_records(inp, columns=inp.keys())
    else:
        result = pd.DataFrame.from_records(inp)

    assert do_test(inp, result, test=test, verbose=verbose, inp_rtype=inp_rtype, inp_shape=inp_shape, gedreht=gedreht)
    if autotranspose  and  result.shape[0]*autotranspose < result.shape[1]: # Hundertmal breiter als lang: transpose
        result = result.transpose()
    result = cols_benennen(result)        
    return result



# ==================================================================================================
# Columns
# ==================================================================================================

def drop_cols(df, colnames):
    '''
    Drops a column or a list of columns.
    Does not throw an error if the column does not exist.
    '''
    
    if type(colnames) is str:
        colnames = [colnames] #let the command take a string or list
        
    result = df.copy()
    for c in colnames:
        while c in result.columns:
            result = result.drop(c, axis=1)        
    return result



def rename_col(df, name_from, name_to):
    '''
    Renames a column of a DataFrame.
    If you try to rename a column again, no error is thrown (better for the workflow in jupyter notebooks).
    '''
    if name_from == name_to:
        return df
    if (name_to in df.columns) and (not name_from in df.columns):
        return df # if you try to rename a column again, no error is thrown.
    if name_to in df.columns:
        raise ValueError( name_to + ' already exists!')
    if not name_from in df.columns:
        raise ValueError( name_from + ' does not exist!')        
    return df.rename( columns = { name_from: name_to } )    



def move_cols( df, colnames=None, to=0 ):
    ''' 
    Reorders the columns of a DataFrame. 
    The specified columns are moved to a numerical position or behind a named column.
    If no arguments are given, sorts the columns lexicographically.
     * colnames: String or list of column names
     * to:       0: move to the front
                -1: move to the back
               <i>: move to position i (buggy)
         <colname>: move behind a specific column
    '''   
    if colnames is None:
        return df.reindex(sorted(df.columns), axis=1)  
    if type(colnames) is str:
        colnames = [colnames] #let the command take a string or list
        
    if type(to) is str:       
        to = list(df.columns).index(to) + 1
        
    colnames_sort  = [c for c in colnames   if c in df.columns]      
    colnames_other = [c for c in df.columns if c not in colnames]

    
    if to==0:
        return df[ colnames_sort + colnames_other ]
    elif to==-1:    
        return df[ colnames_other + colnames_sort] 
    else:
        cols_start = [c for c in list(df.columns)[:to]  if c not in colnames_sort]
        cols_rest  = [c for c in df.columns             if c not in cols_start and c not in colnames_sort]   
        #print(cols_start)
        #print(colnames_sort)        
        #print(cols_rest)            
        return df[ cols_start + colnames_sort + cols_rest] 

    
        

        
    

# ==================================================================================================
# update_col
# ==================================================================================================


def update_col(df_to, df_from, on=[], left_on=[], right_on=[], col='', col_rename='', col_score='', func='', cond='', keep='', return_mask=False, verbose=None):
    '''
    Transfers one column of data from one dataframe to another dataframe.
    Unlike a simple merge, the index and the dtypes are retained. 
    Handles dups and conditions. Verbose if wanted.
    
    df_to:        Dataframe to be changed
    df_from:      Dataframe that contains the new data. Does not have to be duplicate-free.
    on:           Column name or list of column names whose values must match 
    left_on:      Different for left and right if necessary
    right_on:     Different for left and right if necessary    
    col:          Name of the column to be transmitted. Must exist in df_from. 
                  If not present in df_to, it will be appended. 
                  If already present, matching values will be overwritten.
    col_rename:   New name for col, if specified. If already present, matching values will be overwritten and not matching values are preserved.      
    col_score:     Name of a score column used for picking rows from df_from with maximum score. This is useful to avoid conflicting values in case of dups in df_from.
                  If empty: No dup avoidance by rank.
    func:         Name of the function used for dup avoidance. E.g. 'max'. If empty: No dup avoidance by func.
                  Might be slow, use col_score if possible.   
    cond:         Empty, 'min','max' or 'null'. 
                  'min','max': Only write if the new value is smaller / larger than the existing value
                  'null'     : Only write if there is no existing value in df_to (it's Null or empty string) 
    keep:         Should the original value be kept in a separate column? 
                  The string keep is used as suffix for this column.
                  NaN if record is unchanged!  
    return_mask:  If True, the result is a tuple of the resulting dataframe plus a mask masking the affected rows
    verbose:    Messages on / off
    '''
    
    if verbose is None:
        verbose = Config.get('VERBOSE')      
    
    # col_rename
    if col_rename=='':
        col_rename = col
    
    # keep ggf. korrigieren
    if not col in df_to.columns:
        keep = ''    
    
    # auf left_on und right_on umstellen
    if on:
        left_on  = on
        right_on = on            
        
    # Listen erzwingen
    if left_on  and not isinstance(left_on, list):
        left_on = [left_on]
        
    if right_on and not isinstance(right_on, list):
        right_on = [right_on] 
        
        
    # df_from vorbereiten
    if func:
        anz = df_from.shape[0]
        df_from = df_from[  right_on+[col]  ].copy()
        gruppiert = df_from.groupby(right_on)
        df_from = gruppiert.agg(
            col_max = (col, func ),                          
        ).reset_index()   
        df_from = rename_col(df_from, 'col_max', col)
        if (df_from.shape[0] == anz)  and  verbose:
            print('update_col:','func' ,func, 'applied, but it was pointless!')
        elif (df_from.shape[0] != anz)  and  verbose:
            print('update_col:','func' ,func, 'applied,', anz - df_from.shape[0], 'records less!')   
            
    elif col_score:
        anz = df_from.shape[0]
        df_from = rank(df_from, col_score=col_score, cols_group=right_on, on_conflict='first', verbose=False)[ right_on+[col] ].copy()
        if (df_from.shape[0] == anz)  and  verbose:
            print('update_col:','col_score' ,col_score, 'applied, but it was pointless!')
        elif (df_from.shape[0] != anz)  and  verbose:
            print('update_col:','col_score' ,col_score, 'applied,', anz - df_from.shape[0], 'records less!')           
        
    else:
        df_from = df_from[ right_on+[col] ].copy()
    
    # umbenennen wie left bzw. df_to
    df_from.columns = left_on+[col_rename]
        
    # Index merken
    df_to = df_to.copy()
    df_to['copy_index'] = df_to.index
    copy_index_name = df_to.index.name
    
    #merge
    result = df_to.merge(df_from, on=left_on, how='left', suffixes= ('', '_new')) 
    
    # Index restaurieren
    result = result.set_index('copy_index')
    result.index.name = copy_index_name

    # keep
    if keep:
        result[col_rename+keep] = result[col_rename]  
    
    # Zielspalte existiert schon
    if col_rename in df_to.columns: 
        col_new = col_rename + '_new'
        # result[col_rename] ist der alte Wert
        # result[col_new]    ist der neue Wert        
        if cond == 'min':
            mask = result[col_new].notnull()   &   (result[col_new] < result[col_rename]) 
        elif cond == 'max':
            mask = result[col_new].notnull()   &   (result[col_new] > result[col_rename]) 
        elif cond == 'null':
            mask = result[col_new].notnull()   &   (  result[col_rename].isnull()   |   (result[col_rename]=='')   )        
        else:         
            mask = result[col_new].notnull() 
        if verbose:
            print('update_col:', result[mask].shape[0], 'cells written into existing column')
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)   
            result.loc[mask, col_rename] = result.loc[mask, col_new]
        result = copy_datatype( result, df_to )    # datatypes restaurieren               
        #result[col_rename] = copy_datatype( result[col_rename], df_to[col_rename] )    # datatype restaurieren
        result = drop_cols(result,[col_new])
        result_mask = mask.copy()
    
    # Die Spalte ist neu
    else:
        #print(list(df_from.columns))
        result             = copy_datatype( result,             df_to        )           # datatypes restaurieren        
        result[col_rename] = copy_datatype( result[col_rename], df_from[col_rename] )    # datatype übernehmen    
        result_mask = result[col_rename].notnull()
        if verbose:
            print('update_col:', result[result_mask].shape[0], 'cells written into new column')        
    
    if keep:
        mask = result[col_rename+keep] == result[col_rename]
        result.loc[mask,col_rename+keep] = np.NaN
        
    if df_to.shape[0] != result.shape[0]:
        print('update_col:','WARNING: df_from identifier not unique.','I call this again with func="max"')
        return update_col(df_to, df_from, left_on=left_on, right_on=right_on, col=col, col_rename=col_rename, func='max', keep=keep, verbose=verbose)
    
    if return_mask:
        return result, result_mask
    return result

    
    
def write_empty_col(df,col_name, content):
    '''
    Writes empty iterables into a column and sets datatype.
    * content: Can be 'list','set','dict','defaultdict', 'string' or any empty iterable.
    Example: write_empty_col(df,'mycol','list') writes empty lists in mycol.
    '''
    result = df.copy()
    if content == 'string':
        result[col_name] = ''
        result[col_name] = result[col_name].astype('string')      
        return result
    if content == 'list':
        content = list()
    elif content == 'set':
        content = set()     
    elif content == 'dict':
        content = dict()   
    elif content == 'defaultdict':
        content = defaultdict(list)

    result[col_name] = [ content for x in range(len(df.index))]
    result[col_name] = result[col_name].astype('object')
    return result   
    
    
    
def copy_datatype(data_to, data_from):
    """
    Copies the dtypes from data_from to data_to. 
    Usable for Series and DataFrames.
    When applied on a DataFrame, it's applied to all column names that match.
    """
    
    result = data_to.copy()
    
    if isinstance(data_to, pd.DataFrame)   and  isinstance(data_from, pd.DataFrame):
        for c in result.columns:
            if c in data_from.columns:
                try:
                    result[c] = result[c].astype( data_from[c].dtypes.name )  
                except:
                    pass
        return result
                
    elif isinstance(data_to, pd.Series)   and  isinstance(data_from, pd.Series):
        #print(data_from.dtypes.name)
        result = result.astype(data_from.dtypes.name)
        return result
    
    else: 
        assert 'Wrong type'



    

    

    

# ==================================================================================================
# Index
# ==================================================================================================
    

def reset_index(df, keep_as=None):
    '''
    Creates a new, unnamed index.
    * keep_as: If keep_as is given, the old index is preserved as a row with this name.
    Otherwise the old index is dropped.
    '''
    
    if len(df.index.names) != 1:
        raise ValueError('index must be 1dim')
    
    if keep_as:
        df.index.names = [keep_as]    
        result = df.reset_index()
        result.index.names = [None]  
        result.columns.name = None
        return result
    
    else:
        df.index.names = [None]   
        result = df.reset_index(drop=True)
        result.columns.name = None
        return result        

    

# benennt den Index um
def rename_index(df, soll):
    '''
    Renames the index
    '''
    df.index.names = [soll]   
    return df




def drop_multiindex(df, verbose=None):
    '''
    Converts any MultiIndex to normal columns and resets the index. 
    Works with MultiIndex in Series or DataFrames, in rows and in columns.
    '''
    
    if verbose is None:
        verbose = Config.get('VERBOSE')      
    
    # Series mit MultiIndex
    if isinstance(df, pd.Series)   and   (df.index.nlevels > 1): 
        if verbose:
            print('Series with MultiIndex')
        return df.reset_index()    
    
    # Series ohne MultiIndex
    if isinstance(df, pd.Series): 
        if verbose:
            print('Series without MultiIndex')        
        return df    
    
    # DataFrame ohne MultiIndex: 
    if (df.columns.nlevels <= 1)   and   (df.index.nlevels <= 1):
        if verbose:
            print('DataFrame without MultiIndex')             
        return df
    
    # DataFrame mit Zeilen-MultiIndex: 
    if (df.columns.nlevels <= 1)   and   (df.index.nlevels > 1):
        if verbose:
            print('DataFrame with Row-MultiIndex')         
        return df.reset_index()   
    
    # DataFrame mit Spalten-MultiIndex: 
    if (df.columns.nlevels > 1)   and   (df.index.nlevels <= 1):
        if verbose:
            print('DataFrame with Column-MultiIndex')            
        result = df.copy()
        result.columns = ['{}_{}'.format(col[0], col[1]) for col in result.columns]
        return result
    
    # DataFrame mit Zeilen- und Spalten-MultiIndex: 
    if (df.columns.nlevels > 1)   and   (df.index.nlevels > 1):
        if verbose:
            print('DataFrame with Row- and Column-MultiIndex')          
        result = df.copy()
        result.columns = ['{}_{}'.format(col[0], col[1]) for col in result.columns]
        return result.reset_index()   
   

    #print([(col[0], col[1]) for col in df.columns])

    
    
    
# Liefert einen Dataframe mit dem angegebenen Index.
# Wenn als Index ein Dataframe angegeben wird, wird dessen Index verwendet.
# Bsp:
# force_index(lemmas_w, ['tagZ','lemma_lower'])
# force_index(lemmas_w, stat)
#
def force_index(df, soll):
    
    import warnings
    warnings.warn("deprecated, use update_col for the old usecase", DeprecationWarning)    
    
    # ist und soll festlegen
    index_soll = soll
    if type(soll)== pd.DataFrame:
        index_soll = soll.index.names
    index_ist = df.index.names
    
    if index_ist == index_soll:
        return df
    else:
        result = reset_index(df)
        return result.set_index( index_soll)

# ==================================================================================================
# Rows
# ==================================================================================================


def drop_rows(df, mask, verbose=None):
    '''
    Drops rows identified by a binary mask.
    * verbose: True if you want to print how many rows are droped.
    (If you want to delete the rows to a trash, use move_rows.)
    '''

    if verbose is None:
        verbose = Config.get('VERBOSE') 
        
    # damit die Negation funktioniert
    try:
        mask = mask.fillna(False)
    except:
        mask = np.nan_to_num(mask)    
        
    # verbose Statusmeldung
    if verbose:
        anz = df[mask].shape[0]
        if anz == 0: 
            print('No rows deleted')        
        else: 
            print('Delete', anz, 'rows from', df.shape[0])    
        
    return df[~mask].copy() 




def move_rows(df_from, df_to=None, mask=None, msg='', msgcol='msg', verbose=None, zähler=[0]):
    '''
    Moves rows identified by a binary mask from one dataframe to another (e.g. into a trash).
    Returns two DataFrames.
    The target dataframe gets an additional message column by standard (to identify why the rows were moved).
    If you don't give a message, move_rows will generate one: just the count of usage of the function.
     
    * df_from:   Origin DataFrame
    * df_to:     Target DataFrame or None
    * mask:      Binary mask, identifies the rows to be deleted
    * msg:       All moved rows are marked with this message, it's written in the message column of the target dataframe
    * msgcol:    Name of the message column. Set msg or msgcol to None if no message wanted.
    * verbose:   True if you want to print how many rows are moved.
    
    Examples:
    df, df_trash = move_rows( df, mask )           # move rows away (and create a new trash). 
                                                   # The second argument is detected as mask, not as df_to.
    df, df_trash = move_rows( df, df_trash, mask ) # move rows away (into the existing trash)   
    '''
    
    if verbose is None:
        verbose = Config.get('VERBOSE') 
        
    if not isinstance(df_from, pd.DataFrame):
        raise ValueError('df_from must be a DataFrame')         
        
    # df_to enthält mask    
    if isinstance(df_to, pd.Series):
        mask  = df_to
        df_to = None
        
    # msg   
    if not df_to is None:
        zähler[0]+=1   
    if msg=='': 
        msg = str(zähler[0])        
        
    # damit die Negation funktioniert
    try:
        mask = mask.fillna(False)
    except:
        mask = np.nan_to_num(mask) 
        
    # anz    
    anz = df_from[mask].shape[0]         
       
    # verbose Statusmeldung
    if verbose:
        if anz == 0: 
            print('No rows moved')               
        else: 
            print('Move', anz, 'rows from', df_from.shape[0])     
            
    # anz == 0    
    if anz == 0 and not df_to is None:      
        return df_from, df_to       
    if anz == 0 and df_to is None:     
        return df_from, pd.DataFrame()        
    
    # Löschen bestehendem trash    
    if not df_to is None:     
        t = df_from[mask].copy()
        if msgcol and not msg is None:        
            t[msgcol] = msg  # kennzeichnen   
            t[msgcol] = t[msgcol].astype('string')
        r1 = df_from[~mask].copy()
        r2 = pd.concat([df_to, t])
        r2 = copy_datatype(r2, r1)    # falls neue Spalten dazugekommen sind         
        return r1, r2   
    
    # Löschen mit neuem trash    
    if df_to is None:     
        t = df_from[mask].copy()
        if msgcol and not msg is None:
            t[msgcol] = msg # kennzeichnen  
            t[msgcol] = t[msgcol].astype('string')
        return df_from[~mask].copy() , t
        
    return "ERROR"            
    









def add_rows(df_main, df_add, only_new=None, reindex=True, assert_subset=False, verbose=None):
    '''
    Like concat, with additional features only_new and verbose.
    * df_main:       The new rows are added to the end of this dataframe.
    * df_add:        Rows to add. The dtypes are adapted to those of df_main.
                     Series or list are also accepted for appending.
    * only_new:      Avoid duplicates by setting this to a list of column names.
                     This combination must contain new content.
                     A single column name as string works the same way.
                     Or set only_new=True, this will avoid duplicate row indexes.
    * reindex:       Will the result get a fresh index without dups?
    * assert_subset: Check if all columns in df_add exist in df_main already?
    * verbose:       Print status messages how many rows affected
    '''
    
    if not type(df_main) is pd.DataFrame:
        df_main = dataframe(df_main, verbose=False)
        
    if not type(df_add) is pd.DataFrame:
        df_add = dataframe(df_add, verbose=False)
    
    if verbose is None:
        verbose = Config.get('VERBOSE')   
        
    if assert_subset: 
        assert set(df_add.columns) <= set(df_main.columns)        
    
    # only_new
    if only_new:
        
        #let the command take a string as well 
        if isinstance(only_new,str):  
            only_new = [only_new]  
            
        # only_new is list >> recursiv call
        if isinstance(only_new,list):  
            mask = ~isin(df_add, df_main, on=only_new)  
            if verbose:
                print(df_add.shape[0]-df_add[mask].shape[0], 'rows not attached')        
            return add_rows(df_main, df_add[mask], reindex=reindex, verbose=verbose)
        
        # only_new=True >> recursiv call
        if only_new==True:
            mask = ~df_add.index.isin(df_main.index)   
            if verbose:
                print(df_add.shape[0]-df_add[mask].shape[0], 'rows not attached')        
            return add_rows(df_main, df_add[mask], reindex=reindex, verbose=verbose)
            
        raise ValueError('only_new must be str, list or True') 
        
    # dtypes anpassen
    df_add = copy_datatype(df_add, df_main)     
    
    # result  
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)     
        result = pd.concat([df_main, df_add], ignore_index=reindex) 
    
    # Statusmeldung
    if verbose:
        if result.index.is_unique:        
            print(df_add.shape[0], 'rows added, now a total of', result.shape[0])
        else:
            print(df_add.shape[0], 'rows added, now a total of', result.shape[0], 'Warning: Index is not unique!')            

            
    
    return result



def quicksample(data, size):
    '''
    Returns a subset of the input with the given size,
    including the first and the last rows.
    Works with DataFrame or Series.
    '''
    if size <= 1:
        return data.head(size)
    if data.shape[0] <= size:
        return data    

    if size == 2:
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)   
            result = pd.concat( [data.head(1),data.tail(1)] )
        return result
    
    if size < 10:
        n = 1
    elif size < 100:
        n = 2        
    else:
        n = 3        
    r1 = data.head(n)
    r2 = data.iloc[n:-n].sample(size-n-n)
    r3 = data.tail(n)    
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)      
        result = pd.concat([r1,r2,r3])
    return result 


    


# ==================================================================================================
# first_valid_value und last_valid_value
# ==================================================================================================
 

    
def first_valid_value(series):
    '''
    Returns the first not-nan values of a Series. 
    '''
    try:
        idx = series.first_valid_index()
        if idx is None:
            return None
        result = series.loc[idx]
        if isinstance(result, pd.Series): # das liegt an nonunique Index
            return result.iloc[0]
        else:
            return result    
    except:
        return None        
    


def last_valid_value(series):
    '''
    Returns the last not-nan values of a Series. 
    '''    
    try:
        idx = series.last_valid_index()
        if idx is None:
            return None
        result = series.loc[idx]
        if isinstance(result, pd.Series): # das liegt an nonunique Index
            return result.iloc[-1]
        else:
            return result    
    except:
        return None     
    




# ==================================================================================================
# list
# ==================================================================================================
 

#
def find_in_list( df, col_list_of_strings, searchstring ):
    '''
    Searches a column with a list of strings.
    Returns a binary mask for the rows containing the searchstring in the list.    
    '''
    zusatzfelder = pd.DataFrame(  df[col_list_of_strings].explode()  )
    mask_z  = (  zusatzfelder[col_list_of_strings] == searchstring  )
    auswahl = zusatzfelder[mask_z]
    mask    = df.index.isin(auswahl.index)   
    return mask   




def apply_on_elements(series, funktion):
    '''
    Applies a function to all elements of a Series of lists.
    Example:
    df = pak.people()
    df['history2'] = pak.apply_on_elements(df.history, lambda x: x+'2' if x==x else '')    
    Also works with sets.
    '''
    return series.explode().apply(funktion).groupby(level=0).apply(list)




def list_to_string(series, sep=','):
    '''
    Converts a Series of lists of strings into a Series of strings.
    * sep: The separator, default is ','
    Example:
    df = pak.people()
    df['history2'] = pak.list_to_string(df.history)    
    '''

    def try_join(l):
        if not l:
            return ''
        try:
            return sep.join(map(str, l))
        except TypeError:
            return str(l)

    result = [try_join(l) for l in series]
    result = pd.Series(result).astype('string')
    return result
    
    


####################################################################################################
# ..................................................................................................
# Mehrere Dataframes 
# ..................................................................................................
####################################################################################################



# ==================================================================================================
# isin
#   
def isin( df1, df2, on=[], left_on=[], right_on=[] ):
    '''
    isin over several columns. 
    Returns a mask for df1: The rows of df1 that match the ones in df2 in the specified columns.
    
    '''
    if on:
        left_on  = on
        right_on = on        
        
    i1 = df1.set_index(left_on).index
    i2 = df2.set_index(right_on).index    
    result = i1.isin(i2)   
    result = pd.Series(result).to_numpy(na_value=False)
    return result   



#################################################################################################
# ...............................................................................................
# Series
# ...............................................................................................
#################################################################################################


def repeat(content, size):
    '''
    Creates a Series with defined size by repeating a list 
    '''
    s = int(size / len(content))
    result = pd.Series(content).repeat(s)
    if size % len(content) == 0:
        return result.reset_index(drop=True)
    else:
        return pd.concat([result,pd.Series(content)]).head(size).reset_index(drop=True)



