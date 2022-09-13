
import collections, warnings

from functools   import partial  

import numpy  as np
import pandas as pd

from pandas.api.types import is_string_dtype, is_numeric_dtype

import bpyth  as bpy

from pandasklar.type_info    import type_info
from pandasklar.values_info  import values_info

from pandasklar.config       import Config
from pandasklar.pandas       import scale, drop_cols, rename_col, move_cols, reset_index, dataframe, group_and_agg
from pandasklar.pandas       import top_values, top_values_count
from pandasklar.pandas       import drop_cols

try:
    from termcolor import colored    
except ImportError:
    pass

    
import seaborn as sns
import matplotlib.pyplot as plt
#sns.set()


try:
    import seaborn
except ImportError:
    print('seaborn nicht importierbar')  
    


#import locale 
#locale.setlocale(locale.LC_ALL, '') 

#from pandasklar.analyse  import type_info, values_info, val_first_valid



# ==================================================================================================
# Load (gehört eigentlich nicht hierher, muss aber wg. changedatatype
# ==================================================================================================
# 
#
def load_pickle( filename, resetindex='AUTO', changedatatype=True, verbose=False ): 
    '''
    Convenient function to load a DataFrame from pickle-File
    resetindex == True:   Force reset_index
    resetindex == False:  No reset_index    
    resetindex == 'Auto': (Standard) Automatic        
    '''
    result = bpy.load_pickle(filename)
    if resetindex == True:
        result = result.reset_index()
        if verbose:
            print('reset_index')
    elif resetindex == 'AUTO':
        result = result.reset_index()
        if verbose:
            print('reset_index')       
        result = drop_cols(result, 'index') 
        if verbose:
            print('drop_col index')        
    else:
        if verbose:
            print('no reset_index')        
    if changedatatype:
        result = change_datatype(result, verbose=False)
    return result

 

    
#################################################################################
# ...............................................................................
# Spalten auf datatype untersuchen
# ...............................................................................
#################################################################################    
    

    
def mem_usage(data):
    """Liefert den Speicherverbrauch einer Series oder eines Dataframe"""
    
    if isinstance(data, pd.Series): 
        result = data.memory_usage(index=False, deep=True)
        return sizeof_fmt(result)    
    
    elif isinstance(data, pd.DataFrame):
        result = data.memory_usage(index=False, deep=True).sum()
        return sizeof_fmt(result)    
        
    else:
        assert 'ERROR'    



# für mem_usage: Menschenlesbare Darstellung von Bytes
def sizeof_fmt(num, suffix=''):
    for unit in [' ',' K',' M',' G',' T',' P',' E',' Z']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Y', suffix)

    





# Alle class_info einer Series oder eines Index   
def analyse_datatype(data):
    """ Liefert Metadaten zu einer Series oder einem Index in Form eines dict 
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
    """ Liefert die datatypes eines DataFrames
    """
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
    """ selektiert Spaltennamen auf Basis von analyse_cols. Sinnvoll um eine Methode auf bestimmte Spalten eines DataFrame anzuwenden.
        * only:     Nur Spaltennamen, deren datatype so beginnt
        * without:  Ohne Spaltennamen, deren datatype so beginnt
        * as_list:  Ergebnis als Liste ausgeben (sonst als DataFrame, sinnvoll zur Entwicklung und Kontrolle)
        * sort: nach nunique sortiert
        datatypes werden aus analyse_datatypes, Feld datatype_short oder datatype_instance entnommen.
        
        Beispiel: Alle str-Spalten mit fillna behandeln
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
    

    
    
def search_str(df, find, without=[]):
    """ Durchsucht alle str-Spalten eines Dataframe.
        * df
        * find:     Was soll gefunden werden?                      String oder Liste von Strings.
        * without:  Welche Spalten sollen nicht durchsucht werden? String oder Liste von Strings.
    """
    # Argumente formatieren
    if type(find) is str:
        find = [find] 
    if type(without) is str:
        without = [without]
        
    cols = col_names(df, only='str')
    cols = [c for c in cols if not c in without]
    mask = df[cols].isin(find).any(axis=1)  
    return df[mask]    
    
    
    
def change_datatype(data, search=None, verbose=None, msg='', category_maxsize=-1, nanless_ints=False):
    """ Wie astype, akzeptiert aber alle Klassenangaben, die auch type_info akzeptiert.
        Ganz ohne Klassenangabe wird der neue datatype vollautomatisch gewählt.
        Funktioniert auch mit ganzen DataFrames (dann darf aber kein datatype vorgegeben werden)
        * category_maxsize: Wie groß darf eine category werden, damit sie als datatype_suggest vorgeschlagen wird?
        * nanless_ints: Werden Numpys Integerklassen (die kein NaN kennen) als datatype_suggest vorgeschlagen?        
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
            print('change_datatype','vorher:', mem_usage(data),'nachher:',mem_usage(result))
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
    """ Liefert Statistikdaten zu einem DataFrame, einer Series oder einem Index   
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
    """ Liefert analyse_datatypes und analyse_values eines DataFrame
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
def nnan(data, all=False, sum=False):
    """ Für eine Serie:    Liefert die Anzahl der NaNs
        Für ein DataFrame: Liefert eine Liste aller Felder und die Anzahl der NaNs 
        * all: Sollen bei einem DataFrame auch die Felder ohne NaNs ausgegeben werden?
        * sum: Soll   bei einem DataFrame nur die Gesamtsumme ausgegeben werden?
          identisch mit nnan(df).sum()
    """
    result = data.isnull().sum()
    if isinstance(data, pd.Series) or all:
        return result
    
    if not sum:
        return result[result!=0]
    return result.sum()
  
    

# ersetzt assert_no_nans    
def any_nan(data, without=None):
    """ Gibt es NaNs? Liefert True oder False.
        Funktioniert für Series oder DataFrame.
        assert not any_nan(df) stellt sicher, dass ein Dataframe keine NaNs enthält.
        * without enthält ein Feldnamen oder eine Liste von Feldnamen, die ausgeschlossen werden
    """    
    if not without:
        return data.isnull().values.any()
    
    return drop_cols(data, without).isnull().values.any()
        



    
# https://stackoverflow.com/questions/43831539/how-to-select-rows-with-nan-in-particular-column
def nan_rows(df, col=''):   
    """ Liefert die Zeilen eines Dataframes, die in der angegebenen Spalte NaN sind  
        Wenn keine Spalte angegeben, wird die erste mit nans verwendet.
    """
    
    if col=='':
        col = nnan(df).head(1).index.to_list()[0]
    mask = df[col].isnull()
    return df[mask]
    
    


 
    
    
    
    

#################################################################################################
# ...............................................................................................
# Series 
# ...............................................................................................
#################################################################################################



# ==============================================================================================
# verteilung
# ==============================================================================================

def verteilung(series, style=None, quantile=1, stat=True):
    '''
    Veraltet. Verwende analyse_freqs stattdessen.
    Liefert Informationen über die Verteilung einer Series
    style:       ('key','top','plot')
    quantile:  wird nur für den Plot verwendet und schneidet ihn ab
    basiert auf seaborn für die grafische Darstellung bzw. auf countgrid für die textuelle Darstellung
    Beispiel siehe Pandas/Analyse
    '''
    warnings.warn('Veraltet. Verwende analyse_freqs stattdessen.')
    from matplotlib import pyplot as plt
    
    # Mini-Statistik
    s = analyse_values(series, as_dict=True)
    if stat:
        print(s)   
        print()
    
    # automatische style-Auswahl
    if not style  and  is_string_dtype(series):
        style = 'key'  
    if not style  and  not s:
        style = 'key'      
    if not style  and  (int(s['nunique'])<=7):
        style = 'key'   
    if not style:  
        style = 'top'         
    
    # textuell
    if (style == 'key'): 
        return countgrid(series, sort=False)
    
    # textuell
    elif (style == 'top'): 
        return countgrid(series, sort=True)    
    
    
    # grafisch
    else:
        mask = (series <= series.quantile(quantile))   &   (series >= series.quantile(1-quantile)) 
        try:
            plt.figure(figsize=(16, 4))
            return seaborn.histplot(series[mask])
        except RuntimeError as re:
            if str(re).startswith("Selected KDE bandwidth is 0. Cannot estimate density."):
                return seaborn.histplot(series[mask], kde_kws={'bw': 0.1})
            else:
                raise re
        except ValueError as error:
            if str(error).startswith("could not convert string to float"):
                plt.figure(figsize=(0, 0))
                return countgrid(series, sort=True)    
            else:
                raise error        

                
                
                
def histo(series, quantile=1):                
    '''Histogramm'''
    mask = (series <= series.quantile(quantile))   &   (series >= series.quantile(1-quantile)) 
    try:
        plt.figure(figsize=(16, 4))
        return seaborn.histplot(series[mask])
    except RuntimeError as re:
        if str(re).startswith("Selected KDE bandwidth is 0. Cannot estimate density."):
            return seaborn.histplot(series[mask], kde_kws={'bw': 0.1})
        else:
            raise re
    except ValueError as error:
        if str(error).startswith("could not convert string to float"):
            plt.figure(figsize=(0, 0))
            return countgrid(series, sort=True)    
        else:
            raise error                        

                
                
                
                
# ==============================================================================================
# analyse_freqs 
# war: top_beispiele
# ==============================================================================================

def analyse_freqs(dataframe, cols=None, limits=[], splits=[], sort_count=True ):
    """ Häufigkeitsanalyse, die eine untergeordnete Häufigkeitsanalyse beinhaltet.
        Liefert die wichtigsten Beispiele zu einer disjunkten Verteilung.
    
        * Für dataframe.cols wird eine Häufigkeitsverteilung erstellt 
        * mit splits kann man z.B. auch Sätze zu Wörtern exploden
        * mit split='' kann man den Zeichensatz bzw. die Zeichenhäufigkeiten ermitteln
    """
    
    # generelle Fehler abfangen
    if dataframe is None:
        return 'Nothing to analyze'
    
    if isinstance(dataframe, pd.Series):   
        dataframe = pd.DataFrame(dataframe)
        return analyse_freqs( dataframe, dataframe.columns[0], limits=limits, splits=splits, sort_count=sort_count)
    
    if dataframe.shape[0] == 0:
        return 'Keine Datensätze'
    df = dataframe.copy()
    
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
    
    # splits realisieren
    for i, col in enumerate(cols):
        
        # Sätze zu Wortlisten
        if (splits[i] or splits[i]=='') and col:
            df.loc[:,col] = df[col].str.split(splits[i])    
    
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


def val_first_valid(series):
    """ Liefert den ersten notna-Wert einer Series"""
    
    try:
        result = series.loc[series.first_valid_index()]
        if isinstance(result, pd.Series): # das liegt an nonunique Index
            return result.iloc[0]
        else:
            return result
    except:
        return None


def val_last_valid(series):
    """ Liefert den ersten notna-Wert einer Series"""
    try:
        result = series.loc[series.last_valid_index() ]
        if isinstance(result, pd.Series): # das liegt an nonunique Index
            return result.iloc[-1]
        else:
            return result        
    except:
        return None


    
def val_most(series):    
    """ Liefert den häufigsten Wert einer Series"""
    mask = ~(series.isna().fillna(False))
    ohnenull = series[mask]   
    return ohnenull.value_counts().idxmax()


# Sicher vor TypeError: unhashable type: 'list'
# wird von analyse_cols verwendet.
def nunique(series):
    """Liefert die Anzahl der unterschiedlichen Werte"""
    try:
        return series.nunique()
    except: 
        return series.apply(lambda x: str(x)).nunique()
    
    

def ntypes(series):
    '''
    Liefert die Anzahl der unterschiedlichen types. Untersucht dafür alle Werte der Series.
    NaN-Werte werden nicht mitgezählt.
    '''
    return series[series.notna()].map(type).nunique()





##############################################################################################
# ............................................................................................
# DataFrames 
# ............................................................................................
##############################################################################################



# sortiert die Spalten neu, vielfältigste Spalten zuerst 
def sort_cols_by_nunique(df):
    """Liefert das DataFrame mit umsortieren Spalten zurück.
       Sortiert wird nach nunique.
    """
    spalten = list(analyse_values(df, sort=True, with_index=False).col_name)
    df = df.reindex(spalten, axis=1)
    return df










# ==================================================================================================
# sample
# ==================================================================================================


def sample(df, anz=10):
    """ Liefert einige Beispielzeilen
        Immer den Anfang und das Ende, plus einige zufällige Zeilen in der Mitte
    """
    if df.shape[0] <= anz:
        return df
    anz = int(anz / 3)
    df1 =                df[  0   :  anz ]
    df2 = sample_notnull(df[ anz  : -anz   ], anz+1)  
    df3 =                df[ -anz :        ]  
    result = pd.concat( [df1, df2, df3] ).sort_index()
    #result = df1.append(df2).append(df3).sort_index()
    return result

sample_10     = partial(sample, anz=10)    
sample_20     = partial(sample, anz=20)   
sample_100    = partial(sample, anz=100) 
sample_1000   = partial(sample, anz=1000) 
sample_10000  = partial(sample, anz=10000) 
sample_100000 = partial(sample, anz=100000) 



def sample_notnull(df, anz=6):
    """ Liefert zufällige Beispielzeilen, bevorzugt dabei aber notnull-Zeilen
    """
    df1 = df.sample(anz*10, replace=True).dropna()
    df2 = df.sample(anz*10, replace=True).dropna(thresh=2)
    df3 = df.sample(anz,    replace=True)
    #result = df1.append(df2).append(df3)
    result = pd.concat( [df1, df2, df3] )  
    result = result[~result.index.duplicated(keep='first')]
    result = result.head(anz).sort_index()
    return result



       









# ==============================================================================================
# check_mask
# ==============================================================================================

def check_mask(df, mask, erwartungswert_min=-7777, erwartungswert_max=-7777, msg='', stop=True, verbose=None):
    """ Prüft, wieviele Ergebnisse eine Maske erzielt.
    
        Beispiele:
        ==========
        check_mask( df, mask )         # gibt nur die Anzahl aus    
        check_mask( df, mask, 0 )      # prüft auf genau 0 Datensätze    
        check_mask( df, mask, 2000 )   # prüft auf ungefähr 2000 Datensätze       
        check_mask( df, mask, 10, 50)  # prüft auf 10..50 Datensätze
 
        * erwartungswert_min muss nicht 0 sein, 0 geht immer, es sei denn erwartungswert_max ist 0
        * msg wird als Text zusätzlich ausgegeben
        
        Beispiel ohne stop:
        error = check_mask(df, mask, 214, stop=False)        
        grid(df, mask, error)        
        raise_if(error)
    """
    
    def print_red(msg):
        try:
            msg = colored(msg, 'red', attrs=['reverse','bold'])
        except:
            pass   
        print(msg)      
    
    if verbose is None:
        verbose = Config.get('VERBOSE')      
    
    error = ''
    
    if msg:
        msg = '{:<50s}'.format(msg)
    
    # erwartungswert_max fehlt
    if (erwartungswert_max == -7777)  and (erwartungswert_min != -7777):
        if erwartungswert_min == 0:
            e_min = 0
            e_max = 0
        else:
            e_min = int(erwartungswert_min * 0.5) # Verdoppelung oder Halbierung wird toleriert
            e_max = int(erwartungswert_min * 2.0) + 1        
        return check_mask(df, mask, e_min, e_max, msg=msg, stop=stop, verbose=verbose) 
    
    if type(mask) == pd.Series   or   type(mask) == np.ndarray:
        anz_ds = df[mask].shape[0]
    else:
        anz_ds = df.shape[0]
        
    # erwartungswert_min fehlt
    if erwartungswert_min == -7777:     
        print(msg, "{0} Datensätze".format(anz_ds).strip()  )   
        return
    
    # wurde vielleicht schon abgearbeitet?
    if (anz_ds == 0)   and   (erwartungswert_min > 0) and verbose:
        print_red(msg + " WARNUNG: {0} Datensätze, es sollten aber mindestens {1} sein".format(anz_ds, erwartungswert_min)  ) 
        return 
    
    if anz_ds > erwartungswert_max:
        error = " FEHLER: {0} Datensätze, es sollten aber maximal {1} sein".format(anz_ds, erwartungswert_max)  
    elif (anz_ds < erwartungswert_min):
        error = " FEHLER: {0} Datensätze, es sollten aber mindestens {1} sein".format(anz_ds, erwartungswert_min)     
    else:
        if verbose:
            print(msg, "{0} Datensätze".format(anz_ds).strip()  )  
        
    #raise_later  
    if (anz_ds > erwartungswert_max)  or (anz_ds < erwartungswert_min):
        if stop:
            raise Exception(  (msg + error).strip()  )
            
        else:
            print_red(   (msg + error).strip()  )
            return error.strip()
    




# ==================================================================================================
# analyse_groups
# ==================================================================================================


def analyse_groups(df, exclude=[], tiefe_max=3):
    """ Komplettanalyse eines DataFrame auf Eindeutigkeit und Redundanz """
    #return analyse_groups_worker(df, exclude, tiefe_max)
    try:
        return analyse_groups_worker(df, exclude, tiefe_max)
    except:
        # alles in String wandeln
        print('wandle in Strings...')
        df = df.applymap(lambda x: str(x))   
        return analyse_groups_worker(df, exclude, tiefe_max)


    

    
    
# ==================================================================================================
# Vergleich zweier DataFrames
# ==================================================================================================


def check_identical(df1, df2, verbose=None):
    '''
    Sind zwei DataFrames inhaltlich identisch?
    Dazu müssen sie:
    * die gleiche shape besitzen
    * die gleichen Spalten besitzen -- Reihenfolge ist aber egal
    * die gleichen Zeilen besitzen  -- Reihenfolge ist aber egal
    * Pro Spalte / Zeile den gleichen Inhalt besitzen
    * datatypes müssen nicht gleich sein.
    
    '''
    # Series
    if isinstance(df1, pd.Series):   
        df1 = pd.DataFrame(df1)  
        df1.columns = ['A']
        if not isinstance(df2, pd.Series):   
            if verbose:
                print('different datatype')
            return False
        else:
            df2 = pd.DataFrame(df2)   
            df2.columns = ['A']            
            return check_identical(df1, df2, verbose=verbose)
    
    if verbose is None:
        verbose = Config.get('VERBOSE')      
    
    # Unterschiedliche shape?
    if df1.shape != df2.shape:
        if verbose:
            print('different shape')
        return False
    
    # Unterschiedliche Spaltennamen?
    if set(df1.columns) != set(df2.columns):
        if verbose:
            print('different columns')        
        return False
    
    # Unterschiedliche Zeilenindizies?
    if set(df1.index) != set(df2.index):
        if verbose:
            print('different index')        
        return False    
    
    # Unterschiede in den Zellen finden
    diff = get_different_rows(df1, df2)
   
    return diff.shape[0] == 0
    


    
    
# same_but_different
# https://stackoverflow.com/questions/50583828/select-rows-with-same-id-but-different-values-in-pandas
#   
def same_but_different(df, same, different, sort=True, return_mask=False):
    """ Liefert die Zeilen eines DataFrame, die sich einerseits gleichen und andererseits verschieden sind:
        Sie gleichen sich in den in same genannten Feldern.
        Und sie unterscheiden sich im in different genannten Feld.
        * same:      Array von Spaltennamen. Hiernach wird gruppiert.
        * different: Einzelner Spaltenname.  Hier werden Unterschiede gesucht.
        Nützlich ist das zur Analyse, ob Felder 100%ig miteinander korrelieren oder aber eigenständig sind.
    """
    mask = df.groupby(same)[different].transform('nunique') > 1
    if return_mask:
        return mask
    if sort:
        return df[mask].sort_values(same)
    return df[mask]    
    
    
    
# 
# https://stackoverflow.com/questions/19917545/comparing-two-pandas-dataframes-for-differences
# Returns just the rows from the new dataframe that differ from the source dataframe
#
def get_different_rows(df1, df2, indicator=True):
    """ 
    Liefert die Zeilen zweier Dataframes, die sich unterscheiden. 
    Hilfreich für Kontrollzwecke.
    Bei float kann es zu Fehlern kommen.
    """
    
    # Spalten hashable machen
    df1 = df1.copy()
    df2    = df2.copy()  
    
    cols = col_names(df1, query='not is_hashable')
    if cols:
        df1[cols] = df1[cols].apply(lambda x: str(x))
        
    cols = col_names(df2, query='not is_hashable')  
    if cols:
        df2[cols]   = df2[cols].apply(lambda x: str(x))    
    
 
    merged_df = df1.merge(df2, indicator=True, how='outer')
    mask = (merged_df['_merge'] != 'both') 
    result = merged_df[mask] # alle geänderten Zeilen
    if not indicator:
        result = result.drop('_merge', axis=1)
    return result
    
    
    #except: # datatypes vorher angleichen
    #    df1 = change_datatype(df1, verbose=False)
    #    df2    = change_datatype(df2,    verbose=False)        
    #    merged_df = df1.merge(df2, indicator=True, how='outer')
    #    mask = (merged_df['_merge'] != 'both') 
    #    result = merged_df[mask] # alle geänderten Zeilen
    #    if not indicator:
    #        result = result.drop('_merge', axis=1)
    #    return result      
    #   
        












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
        warnings.warn('Keine Datensätze')
        return 
    result = pd.DataFrame(   series.value_counts()   ).reset_index()
    countname   = series.name + '_count'
    percentname = series.name + '_percent'
    result.columns = [series.name, countname]
    result[percentname] = scale(result[countname], typ='rel').round(3)*100
    result['graph'] = (result[percentname] * 0.5).round(0).astype(int) 
    result.graph = result.graph.apply( lambda x: x*'#')
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
    # https://qastack.com.de/programming/1482308/how-to-get-all-subsets-of-a-set-powerset
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
    for level in range(1,tiefe_max+1):
        
        if level > tiefe_max:
            break        
        
        # spaltenkombis
        spaltenkombis = powerlist(spalten,level,level)

        # Einzelne Zeile
        for sp in spaltenkombis:
            a = sp                                      # columns
            b = level                                   # level = group_size
            c = df.duplicated(subset=sp)                # c.sum() ist dups_abs
            d = hash(tuple(c))                          # hash
            anfügen = (a, b, c.sum(), d )
            #result_0 = pd.concat( [result_0, anfügen])  # alles anfügen
            result_0.append( (a, b, c.sum(), d ) )     # alles anfügen            
        
        # Ergebnis aufbereiten
        result_1 = pd.DataFrame.from_records(result_0)
        result_1.columns = ['columns','level','dups_abs','hash']
        
        # vorzeitigen Abbruch vormerken, sobald es spaltenkombis ohne dups gibt
        mask = (result_1.dups_abs == 0)
        if result_1[mask].shape[0] > 0  and fertig==0:
            tiefe_max = level +1 # Level noch zuende machen
            fertig = 1

    
    # Vollständiges Ergebnis endgültig aufbereiten
    result_1 = reset_index(result_1.sort_values(['dups_abs','level'], ascending=[True,True,]))
    result_1['dups_rel'] = result_1.dups_abs / df.shape[0] 
    
    # Sinnlose rauswerfen
    result_1['rank'] = result_1.groupby('hash')['level'].rank('dense', ascending=True)
    mask = (result_1['rank'] == 1)
    result_1 = result_1[mask].copy()

    result_1 = result_1.sort_values(['dups_abs'])
    result_1 = drop_cols(result_1,['hash', 'rank'])
    result_1 = reset_index(result_1)
    
    return result_1



    




