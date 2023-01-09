import warnings
import pandas as pd

from functools     import partial  

from .pandas       import dataframe, reset_index, drop_cols, rename_col, move_cols, first_valid_value, last_valid_value
from .aggregate    import most_freq_elt
from .rank         import rank_without_group
from .analyse      import col_names




# ==================================================================================================
# specials
# ==================================================================================================


def specials(df, find=['head','first','min','most','max','nan','last','tail'], indicator=None, sort='index' ):
    '''
    Returns rows representing all special values per column.
    The resulting dataframe has the same minimums and maximums per column as the input dataframe, 
    and if a column in the input contains NaNs the result will contain NaNs as well.
    * find: List of what values are searched. 
      Possible values: 'head','first','min','most','max','nan','last','tail'
    * indicator: Show additional column with a note, why the row is in the result
    '''
    
    if type(find) is str:
        find = [find] #let the command take a string or list 
        
    result = df.head(0)
    result['__indicator__'] = ''
    result['__score__']     = 0
    group_cols = []
   
    # head
    if 'head' in find:   
        try: 
            zeile = df.head(1).copy()
            zeile['__indicator__'] = 'head'
            zeile['__score__']     = 9999            
            result = pd.concat([result,zeile]) 
        except:
            pass   


    for col in col_names(df,sort=True):
        df_col = df[col].dropna()
        
        # first
        if 'first' in find:   
            try:        
                value = first_valid_value(df_col)
                mask = result[col] == value       # schon drin?
                if result[mask].shape[0] > 0:
                    result.loc[mask,'__indicator__'] += ' ' + col + '_first'
                else:
                    mask = df[col] == value
                    zeile = df[mask].head(1)
                    zeile['__indicator__'] = col + '_first'
                    result = pd.concat([result,zeile]) 
                if not col in group_cols:
                    group_cols += [col]
            except:
                pass    
            
         

        # min
        if 'min' in find:
            try:
                mask = result[col] == df_col.min() # schon drin?
                if result[mask].shape[0] > 0:
                    result.loc[mask,'__indicator__'] += ' ' + col + '_min'
                else:
                    zeile = rank_without_group(df, col, find='min')
                    zeile['__indicator__'] = col + '_min'
                    result = pd.concat([result,zeile])
                if not col in group_cols:
                    group_cols += [col]                    
            except:
                pass
            
        # most
        if 'most' in find:   
            try:        
                value = most_freq_elt(df_col)
                mask = result[col] == value # schon drin?
                if result[mask].shape[0] > 0:
                    result.loc[mask,'__indicator__'] += ' ' + col + '_most'
                else:
                    mask = df[col] == value
                    zeile = df[mask].head(1)
                    zeile['__indicator__'] = col + '_most'
                    result = pd.concat([result,zeile]) 
                if not col in group_cols:
                    group_cols += [col]                    
            except:
                pass               

        # max
        if 'max' in find:     
            try:            
                mask = result[col] == df_col.max() # schon drin?
                if result[mask].shape[0] > 0:
                    result.loc[mask,'__indicator__'] += ' ' + col + '_max'
                else:
                    zeile = rank_without_group(df, col, find='max')
                    zeile['__indicator__'] = col + '_max'
                    result = pd.concat([result,zeile])  
                if not col in group_cols:
                    group_cols += [col]                    
            except:
                pass                    

        # nan
        if 'nan' in find:   
            try:            
                mask = result[col].isna() # schon drin?
                if result[mask].shape[0] > 0:
                    result.loc[mask,'__indicator__'] += ' ' + col + '_nan'
                else:
                    mask = df[col].isna()
                    zeile = df[mask].head(1)
                    zeile['__indicator__'] = col + '_nan'
                    result = pd.concat([result,zeile])  
                if not col in group_cols:
                    group_cols += [col]                    
            except:
                pass     
            

            
        # last
        if 'last' in find:   
            try:        
                value = last_valid_value(df_col)
                mask = result[col] == value # schon drin?
                if result[mask].shape[0] > 0:
                    result.loc[mask,'__indicator__'] += ' ' + col + '_last'
                else:
                    mask = df[col] == value
                    zeile = df[mask].head(1)
                    zeile['__indicator__'] = col + '_last'
                    with warnings.catch_warnings():
                        warnings.simplefilter(action='ignore', category=FutureWarning) 
                        result = pd.concat([result,zeile]) 
                if not col in group_cols:
                    group_cols += [col]                    
            except:
                pass                 
            
    # tail
    if 'tail' in find:   
        try: 
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=FutureWarning) 
                zeile = df.tail(1).copy()
                zeile['__indicator__'] = 'tail' 
                #zeile['__score__']     = 9998                
                result = pd.concat([result,zeile]) 
        except:
            pass               
                
    # sortieren
    mask = result['__score__'].isna()
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)    
        result.loc[mask,'__score__'] = result[mask]['__indicator__'].str.split().str.len()
    result = result.sort_values('__score__', ascending=False)
    
    # Dups entfernen
    mask = ~result.index.duplicated(keep='first')
    result = result[mask]
    
    #print(group_cols)
    
    if indicator is None:
        result = drop_cols(result,'__indicator__')
    else:
        result = rename_col(result,'__indicator__',indicator)
    
    result = drop_cols(result,'__score__')
    if sort == 'index':
        return result.sort_index()
    else:
        return result




# ==================================================================================================
# sample
# ==================================================================================================


def sample(df, size=None):
    ''' 
    Returns some sample rows.
    Always the beginning and the end, 
    plus the other specials() --see there--,
    plus some random rows.
    * size: Number of rows to return. If size=None, all specials() are returned.
    '''
    if df is None:
        return None
    if size is None:
        return specials(df)
    if size <= 0:
        return df.head(0)
    if df.shape[0] <= size:
        return df
    
    sp = specials(df, sort='score')
    
    # specials sind groß genug
    if sp.shape[0] >= size: 
        return sp.head(size).sort_index()
    
    sa = df.iloc[1:-1].sample(size-2)
    return pd.concat([sp,sa]).head(size).sort_index()


sample_10     = partial(sample, size=10)    
sample_20     = partial(sample, size=20)   
sample_100    = partial(sample, size=100) 
sample_1000   = partial(sample, size=1000) 
sample_10000  = partial(sample, size=10000) 
sample_100000 = partial(sample, size=100000) 



#def sample_notnull(df, size=6):
#    ''' 
#    Returns some sample rows. Prefers notnull rows if possible.
#    Always the beginning and the end, plus some random rows in the middle.
#    * size: Number of rows returned
#    '''    
#    if size <= 0:
#        return df.head(0)
#    if df.shape[0] <= size:
#        return df    
#    df1 = df.sample(size*10, replace=True).dropna()
#    df2 = df.sample(size*10, replace=True).dropna(thresh=2)
#    df3 = df.sample(size,    replace=True)
#    result = pd.concat( [df1, df2, df3] )  
#    result = result[~result.index.duplicated(keep='first')]
#    result = result.head(size).sort_index()
#    return result



       

