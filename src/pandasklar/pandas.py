
import warnings, copy

import pandas as pd 
import numpy  as np
import bpyth  as bpy
   
from functools   import partial    
from collections import Counter, defaultdict 

from pandasklar.config       import Config


#from .python     import flatten, rtype, shape, superstrip
#import locale
#locale.setlocale(locale.LC_ALL, '') 



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
def dataframe(inp, test=False, verbose=None):
    """ 
    Wandelt mehrdimensionale Objekte in Dataframes um.
    dict und tuple werden spaltenweise interpretiert,
    list zeilenweise.
    """
    
    if verbose is None:
        verbose = Config.get('VERBOSE')      
    
    def do_test(inp, result, test, verbose, inp_rtype, inp_shape, gedreht):
        if verbose:
            print('gedreht='+str(gedreht) + ' Output rtype=' + str(bpy.rtype(result)), 'shape=' + str(bpy.shape(result)))
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
                print('Shape passt nicht')            
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
        return cols_benennen(result)    
    
    if isinstance(inp, dict): 
        result = pd.DataFrame.from_records(inp, columns=inp.keys())
    else:
        result = pd.DataFrame.from_records(inp)

    assert do_test(inp, result, test=test, verbose=verbose, inp_rtype=inp_rtype, inp_shape=inp_shape, gedreht=gedreht)
    result = cols_benennen(result)
    
    # Hundertmal breiter als lang: transpose
    if result.shape[0]*100 < result.shape[1]:
        result = result.transpose()
        
    return result



# ==================================================================================================
# Columns
# ==================================================================================================

def drop_cols(df, colnames):
    
    if type(colnames) is str:
        colnames = [colnames] #let the command take a string or list
        
    result = df.copy()
    for c in colnames:
        while c in result.columns:
            result = result.drop(c, axis=1)        
    return result



def rename_col(df, name_from, name_to):
    # Umbenennung ist wahrscheinlich schon erfolgt
    if (name_to in df.columns) and (not name_from in df.columns):
        return df
    if name_to in df.columns:
        raise ValueError( name_to + ' gibts schon!')
    if not name_from in df.columns:
        raise ValueError( name_from + ' gibts nicht!')        
    return df.rename( columns = { name_from: name_to } )    



#https://stackoverflow.com/questions/13148429/how-to-change-the-order-of-datadf-columns
def move_cols( df, colnames, to=0 ):
    """ Sortiert die Spalten um. Die spezifizierten Spalten kommen an den Anfang oder ans Ende.
    
    df:         Dataframe
    colnames:   String oder Liste von Spaltennamen
    to:         0: Vorne anhängen; -1: hinten anhängen; <i>: mittig anhängen; <colname>: mittig anhängen 
    """    
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


def update_col(df_to, df_from, on=[], left_on=[], right_on=[], col='', col_rename='', func='', cond='', keep='', verbose=None):
    """ 
    Überträgt Daten von einem Dataframe zu einem anderen Dataframe und liefert das Ergebnis.
    Anders als bei einem simplen merge bleibt dabei der Index und auch die dtypes erhalten. 
    
    df_to:      Dataframe, der geändert werden soll
    df_from:    Dataframe, der die neuen Daten enthält. Muss nicht dupfrei sein.
    on:         Feldname oder Array von Feldnamen, deren Werte übereinstimmen müssen 
    left_on:    ggf. unterschiedlich für links und rechts
    right_on:   ggf. unterschiedlich für links und rechts    
    col:        Name der Spalte, die übertragen werden soll. Muss in df_from vorhanden sein. 
                Wenn nicht vorhanden in df_to, wird sie angefügt. Wenn schon vorhanden, werden matchende Werte überschrieben.
    col_rename: Neuer Name für col, falls angegeben        
    func:       Name der Funktion, die zur Dup-Vermeidung verwendet wird. Z.B. 'max'. Wenn leer: Keine Dup-Vermeidung
    cond:       Leer, 'min','max' oder 'null'. Nur schreiben wenn der neue Wert kleiner / größer als der bestehende Wert ist, 
                oder bei 'null': wenn es keinen bestehenden Wert gibt.
    keep:       Soll der ursprüngliche Wert behalten werden? Wenn ja, wird er in eine Spalte mit Namen _keep geschrieben. NaN, falls Datensatz unverändert!  
    verbose:    Meldungen an / aus
    """
    
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
            print('update_col','func' ,func, 'angewendet, aber es war sinnlos!')
        elif (df_from.shape[0] != anz)  and  verbose:
            print('update_col','func' ,func, 'angewendet,', anz - df_from.shape[0], 'Datensätze weniger!')            
    else:
        df_from = df_from[ right_on+[col] ].copy()
    
    # umbenennen wie left bzw. df_to
    df_from.columns = left_on+[col_rename]
    
    # copy_datatype (ja, das ist richtig rum)
    df_from = copy_datatype(df_from, df_to)
    
    # Index merken
    df_to = df_to.copy()
    df_to['copy_index'] = df_to.index
    copy_index_name = df_to.index.name
    
    #merge
    result = df_to.merge(df_from, on=left_on, how='left', suffixes= ('', '_new')) 
    
    # Index restaurieren
    result = result.set_index('copy_index')
    result.index.name = copy_index_name

    # https://stackoverflow.com/questions/11976503/how-to-keep-index-when-using-pandas-merge
    #result = df_to.reset_index().merge(df_from, on=left_on, how='left', suffixes= ('', '_new')) #.set_index(df_to.index.names)
    
    # keep
    if keep:
        result[col_rename+keep] = result[col_rename]  
    
    if col_rename in df_to.columns: # oder wenn die Zielspalte überhaupt existiert. 
        col_new = col_rename + '_new'
        # result[col_rename] ist der alte Wert
        # result[col_new]    ist der neue Wert        
        if cond == 'min':
            mask = result[col_new].notnull()   &   (result[col_new] < result[col_rename]) 
        elif cond == 'max':
            mask = result[col_new].notnull()   &   (result[col_new] > result[col_rename]) 
        elif cond == 'null':
            mask = result[col_new].notnull()   &   result[col_rename].isnull()            
        else:         
            mask = result[col_new].notnull() 
        if verbose:
            print(result[mask].shape[0], 'Datensätze geschrieben')
        result.loc[mask, col_rename] = result.loc[mask, col_new]
        result = drop_cols(result,[col_new])
    
    if keep:
        mask = result[col_rename+keep] == result[col_rename]
        result.loc[mask,col_rename+keep] = np.NaN
        
    if df_to.shape[0] != result.shape[0]:
        print('update_col','ERROR: df_from nicht eindeutig','Wird nochmal mit func aufgerufen')
        return update_col(df_to, df_from, left_on=left_on, right_on=right_on, col=col, col_rename=col_rename, func='max', keep=keep, verbose=verbose)
        
    return result

    
    
def copy_datatype(data_to, data_from):
    """Kopiert die dtypes von df_from auf df_to für alle Spaltennamen die übereinstimmen"""
    
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

# Liefert einen Dataframe mit dem angegebenen Index.
# Wenn als Index ein Dataframe angegeben wird, wird dessen Index verwendet.
# Bsp:
# force_index(lemmas_w, ['tagZ','lemma_lower'])
# force_index(lemmas_w, stat)
#
def force_index(df, soll):
    
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
    
    
    

def reset_index(df, keep_as=None):
    '''
    Creates a new, unnamed index.
    If keep_as is given, the old index is preserved as a row with this name.
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
    df.index.names = [soll]   
    return df




# https://stackoverflow.com/questions/39092466/drop-multi-index-and-auto-rename-columns
#
def drop_multiindex(df, verbose=None):
    '''
    Löscht jegliche MultiIndex eines DataFrame oder einer Series
    '''
    
    if verbose is None:
        verbose = Config.get('VERBOSE')      
    
    # Series mit MultiIndex
    if isinstance(df, pd.Series)   and   (df.index.nlevels > 1): 
        if verbose:
            print('Series mit MultiIndex')
        return df.reset_index()    
    
    # Series ohne MultiIndex
    if isinstance(df, pd.Series): 
        if verbose:
            print('Series ohne MultiIndex')        
        return df    
    
    # DataFrame ohne MultiIndex: 
    if (df.columns.nlevels <= 1)   and   (df.index.nlevels <= 1):
        if verbose:
            print('DataFrame ohne MultiIndex')             
        return df
    
    # DataFrame mit Zeilen-MultiIndex: 
    if (df.columns.nlevels <= 1)   and   (df.index.nlevels > 1):
        if verbose:
            print('DataFrame mit Zeilen-MultiIndex')         
        return df.reset_index()   
    
    # DataFrame mit Spalten-MultiIndex: 
    if (df.columns.nlevels > 1)   and   (df.index.nlevels <= 1):
        if verbose:
            print('DataFrame mit Spalten-MultiIndex')            
        result = df.copy()
        result.columns = ['{}_{}'.format(col[0], col[1]) for col in result.columns]
        return result
    
    # DataFrame mit Zeilen- und Spalten-MultiIndex: 
    if (df.columns.nlevels > 1)   and   (df.index.nlevels > 1):
        if verbose:
            print('DataFrame mit Zeilen- und Spalten-MultiIndex')          
        result = df.copy()
        result.columns = ['{}_{}'.format(col[0], col[1]) for col in result.columns]
        return result.reset_index()   
   

    #print([(col[0], col[1]) for col in df.columns])



# ==================================================================================================
# Rows
# ==================================================================================================

# früher: del_rows
def drop_rows(df, mask, trash='KEINER', msg='', msgcol='msg', verbose=None, zähler=[0]):
    """
    * df ist der zu belöschende DataFrame
    * mask kennzeichnet die zu löschenden Zeilen
    * trash=None führt zu einem frischen Papierkorb
    * trash=df_del führt dazu, dass die gelöschten Zeilen dort angehängt werden
    * msg: Die Löschvorgänge werden damit gekennzeichnet. Wenn nicht angegeben, werden sie durchnummeriert.
    * msgcol: Name der Spalte für msg, oder None
    Aufrufe Beispiele:
    df           = drop_rows( df, mask )                      # Löschen ohne Papierkorb    
    df, df_trash = drop_rows( df, mask, None )                # Verschieben, neuer Papierkorb wird angelegt
    df, df_trash = drop_rows( df, mask, df_trash )            # Verschieben, Papierkorb wird ergänzt
        
    df, df2      = move_rows( df, mask )                      # Verschieben ohne Papierkorb-Message, partial für
                                                              # df, df2 = drop_rows( df, mask, None, msgcol=None ) 
    """
    
    if verbose is None:
        verbose = Config.get('VERBOSE')      
    
    # damit die Negation funktioniert
    try:
        mask = mask.fillna(False)
    except:
        mask = np.nan_to_num(mask)
    
    # Methode erkennen
    if trash is None:
        method = 'CREATE'      
    elif type(trash) == pd.DataFrame:
        method = 'MOVE'
        zähler[0]+=1 # Zähler aktualisieren https://stackoverflow.com/questions/21716940/is-there-a-way-to-track-the-number-of-times-a-function-is-called/21717084
    else:
        method = 'DEL'
                
    # msg   
    if msg==''  and  method!='DEL': 
        msg = zähler[0]
    
    # anz
    anz = df[mask].shape[0]    
    
    # verbose Statusmeldung
    anz = df[mask].shape[0]
    if verbose   and anz == 0: 
        print('Keine Löschungen:', msg)        
    elif verbose and method=='DEL'    and anz > 0: 
        print('Lösche', anz, 'Datensätze von', df.shape[0], 'endgültig:', msg)    
    elif verbose and method=='MOVE'   and anz > 0: 
        print('Verschiebe', anz, 'Datensätze von', df.shape[0], 'in bestehenden DataFrame:', msg)     
    elif verbose and method=='CREATE' and anz > 0: 
        print('Verschiebe', anz, 'Datensätze von', df.shape[0], 'in neuen DataFrame:', msg)          
    
    # Löschen ohne trash
    if method=='DEL': 
        return df[~mask].copy()    
    
    # Löschen bestehendem trash    
    if method=='MOVE'    and  anz > 0:     
        t = df[mask].copy()
        if msgcol:        
            t[msgcol] = msg  # kennzeichnen   
        #return df[~mask].copy() , trash.append(t)       
        r1 = df[~mask].copy()
        r2 = add_rows(trash,t)
        return r1, r2   
    
    # Löschen bestehendem trash    
    if method=='MOVE'   and  anz == 0:      
        return df, trash    
    
    # Löschen mit neuem trash    
    if method=='CREATE' and anz > 0:     
        t = df[mask].copy()
        if msgcol:
            t[msgcol] = msg # kennzeichnen  
        return df[~mask].copy() , t
    
    if method=='CREATE' and anz == 0:     
        return df, pd.DataFrame()    
    
    return "ERROR"


# move_rows: Verschiebt Zeilen von einen DataFrame in einen anderen
move_rows = partial(drop_rows, trash=None, msgcol=None)




def add_rows(df_main, df_add, only_new=None, verbose=None):
    """
    Komfort-append.
    Liefert pd.concat(df_main, df_add), allerdings
    * die dtypes von df_add werden an die von df_main angepasst
    * der Index wird neu erstellt, es gibt keine Dups in Index
    * Statusmeldung, wieviele Datensätze angefügt wurden
    * mit only_new können Spaltennamen angegeben werden. Dann werden nur neue Wertkombinationen dieser Spalten angefügt.
    * es werden auch Series oder list zum Anhängen akzeptiert.
    """
    
    if not type(df_main) is pd.DataFrame:
        df_main = dataframe(df_main, verbose=False)
        
    if not type(df_add) is pd.DataFrame:
        df_add = dataframe(df_add, verbose=False)
    
    if verbose is None:
        verbose = Config.get('VERBOSE')      
    
    # only_new
    if only_new:
        if type(only_new) is str:
            only_new = [only_new] #let the command take a string or list        
        mask = ~isin(df_add, df_main, on=only_new)  

    # Statusmeldung
        if verbose:
            print(df_add.shape[0]-df_add[mask].shape[0], 'Datensätze nicht angefügt')        
        return add_rows(df_main, df_add[mask], verbose=verbose)  # rekursiver Aufruf
        
    # dtypes anpassen
    df_add = copy_datatype(df_add, df_main)     
    
    # result
    #result = df_main.append(df_add, ignore_index=True)    
    result = pd.concat([df_main, df_add], ignore_index=True) 
    
    # Statusmeldung
    if verbose:
        print(df_add.shape[0], 'Datensätze angefügt, jetzt insg.', result.shape[0])
    
    return result





# ==================================================================================================
# list
# ==================================================================================================
 

#
def find_in_list( df, suchspalte, suchstring ):
    '''
    Wenn eine Spalte eine Liste von Strings enthält,
    kann man mit dieser Funktion die Datensätze herausfiltern,
    in deren Liste ein bestimmter Suchstring enthalten ist.
    Geliefert wird eine Maske!    
    '''


    zusatzfelder = pd.DataFrame(  df[suchspalte].explode()  )
    mask_z  = (  zusatzfelder[suchspalte] == suchstring  )
    auswahl = zusatzfelder[mask_z]
    mask    = df.index.isin(auswahl.index)   
    return mask   


# Eine Series enthält Listen.
# Wendet eine Funktion elementweise auf jedes Element der Listen an.    
# https://stackoverflow.com/questions/57511904/how-to-remove-empty-values-from-the-pandas-dataframe-from-a-column-type-list
# Beispiel: wiktionary.tag = apply_on_elements( wiktionary.tag, lambda x:  x != '' )
#
def filter_lists(series, funktion):
    return series.explode().loc[funktion].groupby(level=0).apply(list)



# https://stackoverflow.com/questions/45306988/column-of-lists-convert-list-to-string-as-a-new-column
#
def list_to_string(series):
    '''
    Wandelt eine Series in String.
    Wenn sie Listen enthält, werden die Elemente aufgezählt und mit Komma abgetrennt.
    '''

    def try_join(l):
        if not l:
            return ''
        try:
            return ','.join(map(str, l))
        except TypeError:
            return str(l)

    result = [try_join(l) for l in series]
    return result
    
    
    
# ==================================================================================================
# scale
# ==================================================================================================
     
    
# siehe die viel schönere Funktion normiere_rang !!
#
# Wandelt ein Tuple (Rang, Rang_max) in einen Score 0..1 um
# Tuple erzeugt man z.B. so:
# vornam_3['Score'] = list(zip(vornam_3.Rang, vornam_3.Rang_max)) 
# 
def rang2score(inputtuple):
    rang, max = inputtuple
    result = 1-(rang/max) 
    if result > 0.001:
        return result
    else:
        return 0.001





def scale(series, typ, powerfaktor=1 ):
    """ normiert eine series. Siehe Beispiele.
        Der powerfaktor verzerrt die Ergebnisse, so dass die Verteilung nicht mehr linear ist.
    """
    if typ == 'rel':
        return series / series.sum()  # alte Funktion normiere
    elif typ == 'max_abs':
        return series  / series.abs().max()
    elif typ == 'min_max':
        return (series - series.min())           /  (series.max() - series.min())
    elif typ == 'min_max_robust':
        return (series - series.quantile(0.01))  /  (series.quantile(0.99) - series.quantile(0.01))      
    elif typ == 'mean':
        return (series - series.mean()) / series.std()   
    elif typ == 'median': # ist im median 0
        return (series - series.median())  / (series.quantile(0.75) - series.quantile(0.25))     
    elif typ == 'rank':
        return normiere_rang( series, powerfaktor=powerfaktor)
    elif typ == 'faktor':
        median = series.quantile(0.5)        
        result = series.copy()
        result = 1 + scale( series, typ='median'  )  # erst mal für alle, ist im median 1          
        mask = (series < median)           
        result.loc[mask] = scale( series[mask], typ='min_max' )   # kleiner median: 0..1        

        return result




# normiert eine Series auf einen Wert 0..1
# wahrscheinlich ohne die 0 und ohne die 1
# Der powerfaktor verzerrt die Ergebnisse, so dass die Verteilung nicht mehr linear ist
def normiere_rang(s, powerfaktor=1):
    if powerfaktor == 1:
        rang = s.rank(method='dense')
    else:
        rang = np.power(s.rank(method='dense'), powerfaktor)
    maximum = rang.max()
    result = rang / maximum
    abziehen = result.min() / 2
    return result - abziehen
    #return np.sqrt(result - abziehen)


    


####################################################################################################
# ..................................................................................................
# Mehrere Dataframes 
# ..................................................................................................
####################################################################################################



# ==================================================================================================
# isin
#   
def isin( df1, df2, on=[], left_on=[],right_on=[] ):
    """ 
    isin über mehrere Spalten. 
    Liefert eine Maske für df1: Die df1, die mit df2 übereinstimmen.
    
    """
    if on:
        left_on  = on
        right_on = on        
        
    i1 = df1.set_index(left_on).index
    i2 = df2.set_index(right_on).index    
    result = i1.isin(i2)   
    result = pd.Series(result).to_numpy(na_value=False)
    return result   








###############################################################################################
# .............................................................................................
# Aggregation
# .............................................................................................
###############################################################################################


def group_and_agg(df, col_origins, col_funcs=None, col_names=None, dropna=True, verbose=None): 
    '''
    Gruppiert und aggregiert.
    * col_origins: Liste aller columns, die verarbeitet werden sollen
    * col_funcs:   Liste aller Funktionen, die darauf angewendet werden sollen. 
                   Manchmal muss man Strings verwenden, manchmal Funktionsnamen.
                   'group' oder '' = Gruppieren. 
    * col_names:   Liste neuer Namen für die Ergebnisspalten. Optional. Leerzeichen = Standardname wird übernommen.
    * dropna:      Parameter für groupby
    Beispiel:
    group_and_agg(df, 
                  col_origins=['altersklasse','geburtsstadt', 'vorname',     'alter', 'alter', 'vorname'],
                  col_funcs  =['group',       'group',        agg_strings,   'min',   'max',   'min'],
                 )    
    '''
    
    if verbose is None:
        verbose = Config.get('VERBOSE')      
    
    #df['dummy'] = 1
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
    #return steuer    
    
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
        print( '{0} Datensätze weniger, jetzt {1} Datensätze'.format(n0-n1, n1)  )
    
    
    if not col_names:      
        return result
    #return steuer
    #print(list(result.columns))
    #print(list(steuer['col_names']))
    result.columns = list(steuer['col_names'])
    return result
    





def most_freq_elt(s):  
    '''
    Liefert das häufigste Element
    Wie Series.mode, aber liefert immer ein Skalar
    d.h. wenn zwei Elemente gleichhäufig sind, wird einfach irgendeines zurückgegeben 
    ''' 
    try:
        result = list(s.mode())[0]
    except IndexError:
        result = np.NaN
    return result
        
    
    

def top_values(series, limit=3, count=False):
    '''
    Liefert eine Liste der häufigsten Elemente
    oder, wenn es nur eines gibt, dieses Einzelelement
    Beispiel siehe Pandas/LISTS
    df.groupby('sex')['age'].apply(top_values)
    Vorsicht, funktioniert nicht gut bei sehr langen Datensätzen
    Wenn count=True werden nicht die Elemente, sondern deren Häufigkeit geliefert    
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



    
    
    


def agg_words(s):
    '''
    Aggregiert Strings zu einem langen String.
    Es wird immer ein Space zwischen die einzelnen Elemente gesetzt,
    die Reihenfolge bleibt erhalten
    '''
    try:
        result = ' '.join(s.fillna(''))    
        result = bpy.superstrip(result)
    except:
        result = np.NaN
    return result

# alter Name
def agg_strings(s):
    warnings.warn('agg_strings wurde umbenannt in agg_words')
    return agg_words(s)





def agg_strings_nospace(s):
    '''
    Aggregiert Strings zu einem langen String.
    Keine Trennzeichen zwischen den Teilstrings.
    '''
    try:
        result = ''.join(s.fillna(''))
    except:
        result = np.NaN
    return result





def agg_words_nodup(s):
    '''
    Aggregiert Strings (z.B. Signalwörter) zu einem langen String.
    Es wird immer ein Space zwischen die einzelnen Elemente gesetzt,
    die Reihenfolge bleibt erhalten,
    Duplikate werden entfernt.
    '''
    return agg_words(  s.str.split().explode().drop_duplicates()  )


# alter Name
def agg_strings_like_set(s):
    warnings.warn('agg_strings_like_set wurde umbenannt in agg_words_nodup')
    return agg_words_nodup(s)


    


def agg_to_list(s):
    '''
    Aggregiert Elemente zu einer Liste. 
    Das geht zwar normalerweise auch über ein einfaches 'list', aber im Zusammenspiel mit transform funktioniert das nicht.
    Dann lässt sich als Ersatz agg_to_list verwenden.
    Siehe # https://stackoverflow.com/questions/62458837/groupby-transform-to-list-in-pandas-does-not-work
    '''
    result = [s.tolist()]*len(s)
    #result = list(s.astype(int))
    return result    



def agg_dicts(s):
    '''
    Aggregiert dicts zu einem einzelnen dict.
    Kommt ein key mehrfach vor, wird der value überschrieben.
    '''
    result = {key: value for d in s for key, value in d.items()}
    return result



def agg_dicts_2dd(s):
    '''
    Aggregiert dicts oder defaultdict zu einem einzelnen defaultdict(list).
    D.h. mehrfache keys sind erlaubt. Die Values sind immer Listen.  
    '''    
    result = defaultdict(list)
    for d in s:
        for key, value in d.items():
            if not value in result[key]:
                result[key].append(value)  
    return result



def agg_defaultdicts(s):
    '''
    Aggregiert defaultdict(list).
    '''    
    result = defaultdict(list)
    for d in s:
        for key in d:
            result[key].extend( d[key] )  
            result[key] = list(dict.fromkeys( result[key] )) # Dups entfernen
            
    return result




# ==================================================================================================
# dict
# ==================================================================================================
 

    
# https://stackoverflow.com/questions/67336514/pandas-explode-dictionary-to-rows
#
def explode_dict(df, colname, keyname='key', valuename='value', from_defaultdict=False):
    '''
    Wie explode, aber für ein dict.
    df:               Input-Dataframe
    colname:          enthält dict to explode
    keyname:          Name der neuen Spalte für die keys des dict
    valuename:        Name der neuen Spalte für die values des dict   
    from_defaultdict: Soll ein zusätzliches explode ausgeführt werden? 
                      Das kann bei defaultdicts sinnvoll sein. Andernfalls erhält man Listen.
    '''
    
    result = pd.DataFrame([*df[colname]], df.index).stack().rename_axis([None,keyname]).reset_index(1, name=valuename)
    result = df.join(result)
    result = drop_cols(result,colname)
    
    if from_defaultdict:
        return result.explode(valuename)
    else:
        return result  
    
    

    
 
    
def implode_to_dict(df, groupcols=None, keyname=None, valuename=None, resultname=None, use_defaultdict=False):
    '''
    Macht aus zwei Spalten ein dict. 
    Umkehrung von explode_dict.
    * groupcols        ist ein String oder eine Liste von Namen, nach denen gruppiert wird.
                       Das beeinflusst die Breite der Ergebnisse. None=keine Gruppierung.
    * keyname          ist der Name der Spalte, die die Keys enthält
    * valuename        ist der Name der Spalte, die die Values enthält
    * resultname       legt fest, die die Ergebnisspalte heißen soll
    * use_defaultdict  legt fest, ob das Ergebnis ein dict oder ein defaultdict ist. Default: False.
                       Bei use_defaultdict=True überschreiben mehrfach vorkommende Keys einander nicht,
                       die Values sind immer eine Liste.
    '''
    
    if not keyname or not valuename or not resultname:
        raise
    
    # groupcols
    if not groupcols:                                                        # None
        alle = list(df.columns)
        groupcols = [c for c in alle if c not in [keyname, valuename]  ]
    elif groupcols  and not isinstance(groupcols, list):                     # Einzelner String
        groupcols = [groupcols]    
    
    # group_and_agg
    params1 = groupcols + [keyname, valuename]
    params2 = ['group'] * len(groupcols) + [list, list]
    dfg = group_and_agg(df, params1, params2, params1)
    
    # worker für die zip-Arbeit
    def worker_defaultdict(zeile, keyname, valuename, resultname):
        my_dict = defaultdict(list)
        for k, v in zip(zeile[keyname], zeile[valuename]):
            if not v in my_dict[k]:
                my_dict[k].append(v)          
        zeile[resultname] = my_dict        
        return zeile       
    
    def worker_dict(zeile, keyname, valuename, resultname):
        zeile[resultname] = dict(zip(zeile[keyname], zeile[valuename]))
        return zeile    
    
    # worker anwenden
    if use_defaultdict:
        result = dfg.apply(worker_defaultdict, axis=1, keyname=keyname, valuename=valuename, resultname=resultname)
    else:
        result = dfg.apply(worker_dict,        axis=1, keyname=keyname, valuename=valuename, resultname=resultname)    
    
    # Ende
    result = drop_cols(result,[keyname, valuename] )
    
    return result    
    
    
# implode_to_defaultdict
implode_to_defaultdict = partial(implode_to_dict, use_defaultdict=True)       

   
    
    
def cols_to_dict(df, col_dict='', cols_add=[], use_defaultdict=False, drop=True):
    '''
    Verpackt Spalten in ein dict oder defaultdict.
    * df
    * col_dict:        Name der Zielspalte. Kann leer sein, kann aber auch schon ein dict oder defaultdict enthalten. 
    * cols_add:        Spalten, die verpackt werden sollen
    * use_defaultdict: Soll als Datenstruktur ein defaultdict verwendet werden? Andernfalls können keys nur einmal vorkommen.
    * drop:            Sollen die verpackten Spalten gelöscht werden?
    
    cols_to_dict( df, col_dict='A', use_defaultdict=True) # dirty trick: dict in defaultdict wandeln
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
            if (zeile[col] and not pd.isna(zeile[col]))  or  (zeile[col] == 0):
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


# cols_to_defaultdict
cols_to_defaultdict = partial(cols_to_dict, use_defaultdict=True)  



# ==================================================================================================
# Ranking
# ==================================================================================================
#
# https://stackoverflow.com/questions/50381064/select-the-max-row-per-group-pandas-performance-issue # Will man nicht nur einen Datensatz, sondern alle gleichwertigen Datensätze finden,
# verwendet man die eingebaute Funktion groupby.rank
# https://stackoverflow.com/questions/33899369/ranking-order-per-group-in-pandas
    
#rank_sortdrop    
def rank(df, group_spalten, score_spalte, richtung='max', target_spalte='', method='first'):
    """ Gruppiert und ranked. Liefert das beste Ergebnis, oder aber den Ursprungsframe mit zugefügter Ranking-Spalte.
    
    df:              Dataframe
    group_spalten:   Array von Feldnamen, nach denen gruppiert werden soll
    score_spalte:    Name der Spalte, deren Minimum oder Maximum gefunden werden soll.
    richtung:        Soll nach Maximum oder Minimum gesucht werden? Voreinstellung: max
    target_spalte:   Soll eine Ranking-Spalte hinzugefügt werden? Wenn ja, dann hier den Namen angeben
    method:          Wie soll mit uneindeutigen Ergebnissen umgegangen werden?
    """    
    
    if richtung=='max':
        ascending = False
    else:
        ascending = True    
    
    if target_spalte=='':
        result = df.sort_values(by=score_spalte, ascending=ascending, kind='mergesort').drop_duplicates(group_spalten)
    else:
        result = df.copy()
        result[target_spalte] = result.groupby(group_spalten)[score_spalte].rank(method=method, ascending=ascending)
        result[target_spalte] = result[target_spalte].fillna(999999)
        result[target_spalte] = result[target_spalte].astype(int)         
        
    return result 

        
    
def rank_sort(df, group_spalten, score_spalte, richtung='max'):
    
    if richtung=='max':
        keep = 'last'
    else:
        keep = 'first'
        
    sort_spalten = group_spalten + [score_spalte]
    found        = df.sort_values(sort_spalten).drop_duplicates(group_spalten, keep=keep).index
    
    return df.loc[found]    
    
    
def rank_idxminmax(df, group_spalten, score_spalte, richtung='max'):
    
    if richtung=='max':
        found = df.groupby(group_spalten)[score_spalte].idxmax()
    else:
        found = df.groupby(group_spalten)[score_spalte].idxmin()
            
    return df.loc[found]       






# aggregiert Strings 
# Es wird immer ein Space zwischen die einzelnen Elemente gesetzt,
# die Reihenfolge bleibt erhalten
##ef agg_to_set(s): 
##   try:
##       result = sorted(list(set(s.fillna(''))))
##   except:
##       result = np.NaN
##   return result



