

from .config     import Config


# ==================================================================================================
# Ranking
# ==================================================================================================

def rank_without_group(df, col_score, find='max', col_target='', on_conflict='first'):
    '''
    Select the max row. Or the min.  
    Or mark the rows instead of selecting them. 
    * col_score:    Name of the column whose minimum or maximum is to be found.
    * find:         'min' or 'max', default is 'max'.
    * col_target:   Should a ranking column be added? If yes, then give the name here.
                    If col_target is given, then the dataframe is returned completely, plus the new column.
                    If no col_target is given, only one row with rank 1 is returned (on_conflict is ignored).  
    '''
    
    # Nur Extremzeile ausgeben    
    if col_target=='':
        if find=='max':
            idx = df[col_score].idxmax()
        else:
            idx = df[col_score].idxmin()  
        
        if idx is None:
            return None
        return df.loc[[idx]]

      
    # rank Spalte erzeugen und alle Zeilen ausgeben
    else:
        if find=='max':
            ascending = False
        else:
            ascending = True          
        result = df.copy()
        result[col_target] = result[col_score].rank(method=on_conflict, ascending=ascending)
        maxrank = int('9'*len(str(df.shape[0]+1)))
        result[col_target] = result[col_target].fillna(maxrank)
        if on_conflict != 'average':
            result[col_target] = result[col_target].astype(int)    
        return result
    

def rank(df, col_score, cols_group=None, find='max', col_target='', on_conflict='first',verbose=None):
    ''' 
    Select the max row per group. Or the min.
    Or mark the rows instead of selecting them. 
    * cols_group:   Name of the columns to be grouped by. None if no grouping needed.
    * col_score:    Name of the column whose minimum or maximum is to be found.
    * find:         'min' or 'max', default is 'max'.
    * col_target:   Should a ranking column be added? If yes, then give the name here.
                    If col_target is given, then the dataframe is returned completely, plus the new column.
                    If no col_target is given, only rank 1 is returned.
    * on_conflict:  How to rank the group of records that have the same value. 
                    Possible values: 'min','max','average','dense' and 'first', see pandas rank.
    '''    
    
    if verbose is None:
        verbose = Config.get('VERBOSE')  
        
    if cols_group is None:
        result = rank_without_group(df=df, col_score=col_score, find=find, col_target=col_target, on_conflict=on_conflict)
        if verbose:
            n0 = df.shape[0]
            n1 = result.shape[0]        
            print( 'rank: {0} rows less, now {1} rows'.format(n0-n1, n1)  )
        return result
    
    if find=='max':
        ascending = False
    else:
        ascending = True    
            
    if col_target=='':
            result = df.sort_values(by=col_score, ascending=ascending, kind='mergesort').drop_duplicates(cols_group)
            
    else:
        result = df.copy()
        result[col_target] = result.groupby(cols_group)[col_score].rank(method=on_conflict, ascending=ascending)
        maxrank = int('9'*len(str(df.shape[0]+1)))        
        result[col_target] = result[col_target].fillna(maxrank)
        if on_conflict != 'average':
            result[col_target] = result[col_target].astype(int)         
        
    if verbose:
        n0 = df.shape[0]
        n1 = result.shape[0]        
        print( 'rank: {0} rows less, now {1} rows'.format(n0-n1, n1)  )        
    return result 

    
    
# Alternative Methode für rank   
# see # https://stackoverflow.com/questions/50381064/select-the-max-row-per-group-pandas-performance-issue  
# ok wenn wenige große Gruppen
# ok wenn viele kleine Gruppen
# Ergebnis ist sortiert    
#
def rank_sort(df, cols_group, col_score, find='max'):
    
    if find=='max':
        keep = 'last'
    else:
        keep = 'first'
        
    sort_spalten = cols_group + [col_score]
    found        = df.sort_values(sort_spalten).drop_duplicates(cols_group, keep=keep).index
    
    return df.loc[found]    
    
    
    
# Alternative Methode für rank       
# rank_idxminmax:
# sehr schnell wenn wenige große Gruppen
# extrem langsam wenn viele kleine Gruppen
# Ergebnis ist sortiert    
def rank_idxminmax(df, cols_group, col_score, find='max'):
    
    if find=='max':
        found = df.groupby(cols_group)[col_score].idxmax()
    else:
        found = df.groupby(cols_group)[col_score].idxmin()
            
    return df.loc[found]       


