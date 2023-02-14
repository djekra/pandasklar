
import re

import pandas as pd 
import numpy  as np

from bisect              import bisect_right

from .config   import Config
from .pandas   import rename_col





#verboten = ['zumeist ','nach ','in ']
#pat = r'\b(?:{})\b'.format('|'.join(verboten))   # 
#ALTERNATIVE: pattern = re.compile(r'\b(' + r'|'.join(verboten) + r')\b\s*')
#self.node_meta = self.node_meta.replace( pat, '', regex=True )

def remove_str(series, remove_list, safemode=True):
    '''
    Removes a list of unwanted substrings from a Series of strings.
    * remove_list: list of substrings to remove
    * safemode:    Selects the algorithm.
                   safemode=True:  Each substring is removed separately
                   safemode=False: Works with one regular expression.
                                   Special characters such as asterisks must be backslashed.    
    '''
    if safemode:
        for r in remove_list:
            series = series.str.replace(r, '', regex=False)
        return series.str.strip()    
    
    else:
        remove_list = [re.escape(w) for w in remove_list]
        pat = r'(?:{})'.format('|'.join(remove_list))   # 
        return series.str.replace( pat, '', regex=True ).str.strip()    
    
  
    
    
def remove_words(series, remove_list):
    '''
    Removes a list of unwanted words from a Series of strings.
    Works by regular expression, so special characters such as asterisks must be backslashed.  
    '''
    pat = r'\b(?:{})\b'.format('|'.join(remove_list))   # 
    return series.str.replace( pat, '', regex=True ).str.strip()        
    
    
    
def replace_str(series, translationtable):
    '''
    Replaces substrings from a Series of strings according to a translationtable.
    * translationtable: Can be a dict, a list of tuples or a DataFrame with two columns.
      Example: {'President Trump':'Trump',   'HELLO':'Hello'}
      or      [('President Trump','Trump'), ('HELLO','Hello')]
    '''
    if type(translationtable).__name__ == 'DataFrame':
        translationtable = dict(zip(translationtable.iloc[:, 0], translationtable.iloc[:, 1]))
    if not isinstance(translationtable, dict):
        translationtable = dict(translationtable )
        
    # translationtable is dict now    
    for old, new in translationtable.items():
        series = series.str.replace(old, new, regex=False)
    return series.str.strip()    
    
    
    
    


# ==================================================================================================
#  Wörter trennen
#  


def count_words(series, pat=None):
    """
    Counts the number of words (separated by space) in the strings of a series.
    Returns a series with ints.
    """
    return series.str.split(pat).str.len()    




# ==================================================================================================
#  Wörter trennen
#   

def split_col(df, col, pat=None):
    """ 
    Splits the strings of a column into individual parts, e.g. into individual words.
    The results are written in columns named A, B, C, ... consecutively.    
    - df: Dataframe
    - col: Column with strings that are to be split.
    - pat: String or regular expression to split on. If not specified, split on whitespace.
    """
    zusatzspalten = df[col].str.split(pat, expand=True)
    zusatzspalten = zusatzspalten.fillna('')
    zusatzspalten.columns = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')[0:zusatzspalten.shape[1]]
    result = pd.concat([df, zusatzspalten], axis=1, join='inner')    
    return result
    

    
# ==================================================================================================
#  Slice
#   
    

def slice_string(df, col_text, col_start, col_end, col_result):
    """ 
    Slices a column of strings based on indexes in another columns.
    * col_text:   Name of the column containing the  text.
    * col_start:  Name of the column containing the  start index OR the start index numeric.  
    * col_end:    Name of the column containing the  end index   OR the end index numeric.      
    * col_result: Name of the column to hold the result.        
    """
    
    if isinstance(col_start, int):
        startpositions = pd.Series( col_start, index=range(df.shape[0]) )
    else:
        startpositions = df[col_start].fillna(0).astype(int)

        
    if isinstance(col_end, int):
        endpositions   = pd.Series( col_end, index=range(df.shape[0]) )
    else:        
        endpositions   = df[col_end].fillna(65535).astype(int)  
    
    df[col_result] = [ text[a:z] for text, a, z in  zip(df[col_text], startpositions, endpositions)  ]
    df[col_result] = df[col_result].astype('string')
    return df    




# ==================================================================================================
#  Encode integers as strings (to be able to work with strings instead of lists)
#   

def encode_int(s):
    """ Converts a series of small integers into strings"""
    num_rep = {10:'a',11:'b',12:'c',13:'d',14:'e',15:'f',16:'g',17:'h',18:'i',19:'j',
               20:'k',21:'l',22:'m',23:'n',24:'o',25:'p',26:'q',27:'r',28:'s',29:'t',
               30:'u',31:'v',32:'w',33:'x',34:'y',35:'z'}
    s = s.astype(int)
    mask = (s >= 10) 
    s.loc[mask] = s[mask].apply(lambda x: num_rep[x])   
    return s.astype(str)



def decode_int(s):
    """ Converting strings back into small integers"""    
    num_rep = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
               'a':10,'b':11,'c':12,'d':13,'e':14,'f':15,'g':16,'h':17,'i':18,'j':19,
               'k':20,'l':21,'m':22,'n':23,'o':24,'p':25,'q':26,'r':27,'s':28,'t':29,
               'u':30,'v':31,'w':32,'x':33,'y':34,'z':35 }    
    return s.apply(lambda x: num_rep.get(x) )  




# =================================================================================================================
# fast_startswith
# fast_endswith
# siehe Jupyter-Notepad zum Thema Pandas/Strings
#


def fast_startswith(df, col_search, col_found, searchfor, find_longest=True, find_identical=True):
    """
    Searches string columns for matching beginnings.
    Like pandas str.startswith(), but much faster for large amounts of data,
    and it returns the matching fragment. 
    * col_search:     Name of the column to be searched
    * col_found:      Names of the column into which the result is to be written
    * searchfor:      Series or List of strings to be searched for
    * find_longest:   Should the longest substring be given as the result? Otherwise the shortest.
    * find_identical: Should it be counted as a result if a string matches completely?
    """
    
    # startswith alternative, works only if all strings in searchme have the same length. Also returns the matching fragment
    def startwiths(data, searchme, find_identical):
        prefix = searchme[bisect_right(searchme, data)-1]
        if ((data!=prefix) or find_identical ) and data.startswith(prefix): 
            return prefix    
    
    search = pd.DataFrame(searchfor)
    search.columns = ['searchstring'] 
    search['len'] = search.searchstring.str.len()
    grouped = search.groupby('len')
    lengroups = grouped.agg(list).reset_index().sort_values('len', ascending=find_longest)  
    result = df.copy()
    result[col_found] = None
 
    for index, row in lengroups.iterrows():
        result[col_found].update(result[col_search].apply(startwiths, searchme=sorted(row.searchstring), find_identical=find_identical)  )  
        
    result[col_found] = result[col_found].astype('string')    
    return result



 
def fast_endswith(df, col_search, col_found, searchfor, find_longest=True, find_identical=True):
    '''
    Searches string columns for matching endings.
    Like pandas str.endswith(), but much faster for large amounts of data,
    and it returns the matching fragment. 
    * col_search:     Name of the column to be searched
    * col_found:      Names of the column into which the result is to be written
    * searchfor:      Series or list of strings to be searched for
    * find_longest:   Should the longest substring be given as the result? Otherwise the shortest.
    * find_identical: Should it be counted as a result if a string matches completely?
    ''' 
    # umkehren
    df = df.copy()
    df['ssdgzdgd'] = df[col_search].str[::-1]
    result = fast_startswith(df, 'ssdgzdgd', col_found, searchfor.str[::-1], find_longest=find_longest, find_identical=find_identical)
    # Ergebnis auch umkehren
    result[col_found] = result[col_found].str[::-1].copy()
    result = result.drop('ssdgzdgd', axis=1)
    return result




def preprocess_strings( dirty, how=''):
    raise('preprocess_strings ist jetzt in bj_nlp')


