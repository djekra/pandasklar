
import re

import pandas as pd 
import numpy  as np

from bisect              import bisect_right

from pandasklar.config   import Config
from pandasklar.pandas   import rename_col
from pandasklar.analyse  import col_names

# import locale
# locale.setlocale(locale.LC_ALL, '') 


#verboten = ['zumeist ','nach ','in ']
#pat = r'\b(?:{})\b'.format('|'.join(verboten))   # 
#ALTERNATIVE: pattern = re.compile(r'\b(' + r'|'.join(verboten) + r')\b\s*')
#self.node_meta = self.node_meta.replace( pat, '', regex=True )

def remove_str(series, remove_list, safemode=True):
    '''
    Entfernt unerwünschte Zeichen und Wörter aus einer Series.
    Bei safemode=False müssen Sonderzeichen wie Sternchen gebackslashed werden.
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
    Entfernt unerwünschte Zeichen und Wörter aus einer Series
    '''
    pat = r'\b(?:{})\b'.format('|'.join(remove_list))   # 
    return series.str.replace( pat, '', regex=True ).str.strip()        
    
    
    
def replace_str(series, replace_dict):
    '''
    Ersetzt Zeichen und Wörter aus einer Series
    '''
    for old, new in replace_dict.items():
        series = series.str.replace(old, new, regex=False)
    return series.str.strip()    
    
    
    
    
# ==================================================================================================
#  Vereinfachung und Säuberung 
# 

def preprocess_strings( dirty, how=''):
    """ Vereinfachung und Säuberung
        * dirty ist Series oder DataFrame. Bei einem Dataframe werden alle str-Spalten behandelt.
        * how legt fest, welche Säuberungsschritte ausgeführt werden.
          - fillna wird sowieso immer ausgeführt
          - strip
          - minus2space ersetzt Bindestriche durch Spaces
          - lower
          - filter_letters löscht alles, was nicht Buchstabe oder Zahl oder Space ist
          - umlaut2single ersetzt Umlaute durch Einzelbuchstaben
          - umlaut2double ersetzt Umlaute durch Buchstabenkombinationen          
          - solowhite ersetzt multiple Whitespaces durch einzelnen Space
          - but_anything nimmt dann doch das Original, falls am Ende sonst gar nichts übrig bleibt
        * für how gibt es auch Presets, diese werden durch ein Kürzel signalisiert
    """
    
    # Presets
    if how == '':
        how = 'solowhite strip'
    elif how == 'STD_0': # lower_and_std
        how += ' strip minus2space lower filter_letters solowhite but_anything' 
    elif how == 'GROB_1': # str_grob
        how += ' umlaut2single esszett2ss'       
        
    # Series
    if   type(dirty) == pd.Series:
        
        result      = pd.DataFrame(dirty) 
        result.columns = ['A'] 
        result['A'] = result.A.fillna('') # Kopie, falls für but_anything benötigt
        result['B'] = result.A.copy()
        
        if 'strip'         in how:
            result.B = result.B.str.strip()           
        if 'minus2space'   in how:
            result.B = result.B.str.replace('-', ' ', regex=False)     
        if 'lower'         in how:
            result.B = result.B.str.lower() 
        if 'filter_letters'  in how:
            result.B = result.B.str.replace(r'[^ÄÖÜäüößA-Z a-z0-9]+', '', regex=True)   
        if 'umlaut2single' in how:
            table = str.maketrans({'ä':'a','ö':'o','ü':'u','Ä':'A','Ö':'O','Ü':'U' })
            result.B = result.B.str.translate(table)  
        if 'umlaut2double' in how:
            table = str.maketrans({'ä':'ae','ö':'oe','ü':'ue','Ä':'Ae','Ö':'Oe','Ü':'Ue' })
            result.B = result.B.str.translate(table)              
        if 'esszett2ss'    in how:
            result.B = result.B.str.replace('ß','ss', regex=False) 
        if 'solowhite'     in how:
            result.B = result.B.replace('\s+', ' ', regex=True)                
        if 'strip'         in how:
            result.B = result.B.str.strip()    
        if 'but_anything'      in how:  
            mask = (result.B == '') # nichts übrig geblieben?
            result.loc[mask,'B'] = result[mask].A                 

        return result.B        


    # DataFrame: auf alle Spalten anwenden
    elif type(dirty) == pd.DataFrame:
        
        result = dirty.copy()      
        cols = col_names(result, only='str')
        
        # ausführen
        for col in cols: 
            result[col] = preprocess_strings( result[col], how=how )
        return result
        
    else:
        assert 'wrong datatype'
            
    return result





# ==================================================================================================
#  Wörter trennen
#  


def count_words(series, pat=None):
    """
    Zählt die Anzahl der (mit Space) getrennten Wörter in den Strings einer Series.
    Liefert eine Series mit Ints zurück.
    """
    return series.str.split(pat).str.len()    




# ==================================================================================================
#  Wörter trennen
#   

def split_col(df, col, pat=None):
    """ Splittet die Strings einer Spalte in Einzelteile, z.B. in einzelne Wörter.
    Die Ergebnisse werden in fortlaufend A, B, C, .. benannte Spalten geschrieben.    
    - df: Dataframe
    - col: Spalte mit Strings, die gesplittet werden sollen
    - pat: String or regular expression to split on. If not specified, split on whitespace.
    """
    zusatzspalten = df[col].str.split(pat, expand=True)
    zusatzspalten = zusatzspalten.fillna('')
    zusatzspalten.columns = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')[0:zusatzspalten.shape[1]]
    result = pd.concat([df, zusatzspalten], axis=1, join='inner')    
    return result
    

    
# ==================================================================================================
#  Wslice
#   
    
# https://stackoverflow.com/questions/45523025/how-to-slice-strings-in-a-column-by-another-column-in-pandas
def slice_string(df, col_text, col_start, col_end, col_result):
    """ 
    Slice String based on columns
    * df
    * col_text:   Name der Spalte, die den Text       enthält
    * col_start:  Name der Spalte, die den Startindex enthält  ODER der Startindex numerisch  
    * col_end:    Name der Spalte, die den Endindex   enthält  ODER der Endindex   numerisch      
    * col_result: Name der Spalte, die das Ergebnis   aufnehmen soll        
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
#  Integerzahlen als Strings codieren (um dann mit Strings statt mit Listen arbeiten zu können)
#   

def encode_int(s):
    """ Wandelt eine Series von kleinen Integern in Strings um"""
    num_rep = {10:'a',11:'b',12:'c',13:'d',14:'e',15:'f',16:'g',17:'h',18:'i',19:'j',
               20:'k',21:'l',22:'m',23:'n',24:'o',25:'p',26:'q',27:'r',28:'s',29:'t',
               30:'u',31:'v',32:'w',33:'x',34:'y',35:'z'}
    s = s.astype(int)
    mask = (s >= 10) 
    s.loc[mask] = s[mask].apply(lambda x: num_rep[x])   
    return s.astype(str)



def decode_int(s):
    """ Zurückwandelung von Strings in kleine Integer"""    
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


def fast_startswith(df, searchfieldname, foundfieldname, searchseries, find_longest=True, find_identical=True):
    """fast startswith alternative
    
    Finds the longest / shortest matching fragment and writes it into the field foundfieldname.
    * df: Der DataFrame der durchsucht werden soll
    * searchfieldname: Name des Feldes, das durchsucht werden soll
    * foundfieldname: Names des Feldes, in das das Ergebnis geschrieben werden soll
    * searchseries: Strings, nach denen gesucht werden soll
    * find_longest: Soll der längste Teilstring als Ergebnis angegeben werden? Sonst der kürzeste.
    * find_identical: Soll es als Ergebnis gewertet werden, wenn ein String komplett übereinstimmt?
    """
    
    # startswith alternative, works only if all strings in searchme have the same length. Also returns the matching fragment
    def startwiths(data, searchme, find_identical):
        prefix = searchme[bisect_right(searchme, data)-1]
        if ((data!=prefix) or find_identical ) and data.startswith(prefix): 
            return prefix    
    
    search = pd.DataFrame(searchseries)
    search.columns = ['searchstring'] 
    search['len'] = search.searchstring.str.len()
    grouped = search.groupby('len')
    lengroups = grouped.agg(list).reset_index().sort_values('len', ascending=find_longest)  
    result = df.copy()
    result[foundfieldname] = None
 
    for index, row in lengroups.iterrows():
        result[foundfieldname].update(result[searchfieldname].apply(startwiths, searchme=sorted(row.searchstring), find_identical=find_identical)  )  
        
    result[foundfieldname] = result[foundfieldname].astype('string')    
    return result



 
def fast_endswith(df, searchfieldname, foundfieldname, searchseries, find_longest=True, find_identical=True):
    """fast endswith alternative
    
    Finds the longest / shortest matching fragment and writes it into the field foundfieldname.
    * df: Der DataFrame der durchsucht werden soll
    * searchfieldname: Name des Feldes, das durchsucht werden soll
    * foundfieldname: Names des Feldes, in das das Ergebnis geschrieben werden soll
    * searchseries: Strings, nach denen gesucht werden soll
    * find_longest: Soll der längste Teilstring als Ergebnis angegeben werden? Sonst der kürzeste.
    * find_identical: Soll es als Ergebnis gewertet werden, wenn ein String komplett übereinstimmt?
    """    
    # umkehren
    df = df.copy()
    df['ssdgzdgd'] = df[searchfieldname].str[::-1]
    result = fast_startswith(df, 'ssdgzdgd', foundfieldname, searchseries.str[::-1], find_longest=find_longest, find_identical=find_identical)
    # Ergebnis auch umkehren
    result[foundfieldname] = result[foundfieldname].str[::-1].copy()
    result = result.drop('ssdgzdgd', axis=1)
    return result







