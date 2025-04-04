
import random, string
from collections import Counter

import numpy  as np
import pandas as pd
import polars as pl
import bpyth  as bpy

from .config   import Config
from .dataframe import dataframe
from .pandas   import reset_index, drop_cols, rename_col, repeat
from .analyse  import nunique, change_datatype, col_names
from .compare  import check_equal
from .scale    import scale

try:
    from perlin_noise import PerlinNoise
except:
    pass



#################################################################################
# ...............................................................................
# Excel
# ...............................................................................
#################################################################################


def dump_excel(df, filename, filetype='xlsx', tabcol='', index=False, changedatatype=True, check=True, verbose=None, framework=None):
    """
    Writes a DataFrame to an Excel file.

    Args:
        df (pd.DataFrame): The DataFrame to write.
        filename (str): The output filename.
        filetype (str, optional): The file type ('xlsx'). Defaults to 'xlsx'.
        tabcol (str, optional): Column name to split data into multiple tabs. Defaults to ''.
        index (bool, optional): Whether to write the index. Defaults to False.
        changedatatype (bool, optional): Whether to adjust data types automatically. Defaults to True.
        check (bool, optional): Whether to read the file back and check for identity. Defaults to True.
        verbose (bool, optional): Whether to print progress information. Defaults to None.
    """

    if verbose is None:
        verbose = Config.get('VERBOSE')

    if framework is None:
        framework = Config.get('FRAMEWORK')

    if framework not in ['pandas', 'polars']:
        raise ValueError(f"Framework '{framework}' is not supported. Only 'pandas' and 'polars' are allowed.")

        # changedatatype
    if changedatatype:
        df = change_datatype(df, verbose=False)

        # .xlsx
    filename = str(filename)  # Hinzugefügt: Konvertiere filename in einen String
    if not filename.endswith('.' + filetype):
        filename += '.' + filetype

    # Einzelner Tab
    if tabcol == '':
        with pd.ExcelWriter(filename) as writer:
            df.to_excel(writer, sheet_name='Pandas', index=index, inf_rep='__unendlich__', na_rep='__NaN__')

    # Viele Tabs
    else:
        # NaN-Werte in tabcol durch einen String ersetzen
        df[tabcol] = df[tabcol].fillna('__NaN__')
        alle_tabs = df[tabcol].unique().tolist()
        with pd.ExcelWriter(filename) as writer:
            for counter, tab in enumerate(alle_tabs):
                mask = df[tabcol] == tab
                tabname = str(tab)  # + '___' + str(counter)
                df[mask].drop(tabcol, axis=1).to_excel(writer, sheet_name=tabname, index=index)

    # check
    if check and not tabcol:
        df2 = load_excel(filename, tabcol=tabcol)
        assert check_equal(df, df2)

#dataframe_to_excel = dump_excel
  

                
                
# Die Umkehrung: Liefert den Dataframe zurück.   
# Wenn tabcol angegeben, wird eine zusätzliche Spalte angefügt, die den Namen des Tab enthält.
#
def load_excel(filename, filetype='xlsx', sheet_name=None, tabcol='', changedatatype=True, verbose=None, framework=None):
    '''
    Loads a DataFrame from an Excel file.

    Args:
        filename (str): The input filename.
        filetype (str, optional): The file type ('xlsx'). Defaults to 'xlsx'.
        sheet_name (str, optional): Name of the only sheet to read. Defaults to None (means all).
        tabcol (str, optional): Column name to use for tab names. Defaults to ''.
        changedatatype (bool, optional): Whether to optimize data types. Defaults to True.
        verbose (bool, optional): Whether to print progress information. Defaults to None.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    '''

    if verbose is None:
        verbose = Config.get('VERBOSE')

    if framework is None:
        framework = Config.get('FRAMEWORK')

    if framework not in ['pandas', 'polars']:
        raise ValueError(f"Framework '{framework}' is not supported. Only 'pandas' and 'polars' are allowed.")

        # .xlsx
    filename = str(filename)  # Hinzugefügt: Konvertiere filename in einen String
    if not filename.endswith('.' + filetype):
        filename += '.' + filetype

    # read
    sheets_dict = pd.read_excel(filename, sheet_name=sheet_name)

    if sheet_name is not None:
        result = sheets_dict
        if changedatatype:
            result = change_datatype(result, verbose=False)
        for col in col_names(result, 'str'):
            result[col] = result[col].str.strip()
        # result = result.replace('__NaN__', None)
        return result

    result = pd.DataFrame()

    for name, sheet in sheets_dict.items():
        if tabcol != '':
            sheet[tabcol] = name  # zusätzliche Spalte
        sheet = sheet.rename(columns=lambda x: x.split('\n')[-1])
        result = pd.concat([result, sheet])

    result.reset_index(inplace=True, drop=True)

    if changedatatype:
        result = change_datatype(result, verbose=False)
    for col in col_names(result, 'str'):
        result[col] = result[col].str.strip()
    # result = result.replace('__NaN__', None)

    if verbose:
        print(result.shape[0], 'rows loaded')

    return result

#excel_to_dataframe = load_excel





#################################################################################
# ...............................................................................
# Random data
# ...............................................................................
#################################################################################


def random_series(size, typ, **kwargs):
    ''' 
    Returns a pd.Series of random data.
    * size
    * typ: 'int', 'float', 'string', 'name', 'choice', 'list', 'time', mix',
           'ascending', 'descending', 'perlin' or 'errorprone'. Or the first letter of this.
           'name' generates random first names, 'list' generates lists of random first names.
           'mix' generates mixed datatypes. 
           'ascending', 'descending' and 'perlin' generates ordered random sequences.
           'errorprone' generates sequences of NaNs, 0, 1 with similar index. Useful for testing. 

    The other arguments are passed to the appropriate functions for the type of random data.
    General arguments are:
    * name
    * p_nan: value 0..1 specifies  how many NaNs are interspersed
    * p_dup: value 0..1 determines how many Dups are included.
    
    There are extra parameters for some types of random data:
    - int:    min, max: closed interval, min and max are both possible values in the generated data
    - float:  decimals: how many decimal places
    - string: len_min, len_max: controls the length of the generated character sequence
              mix: Explicit specification of the available character set. Overwrites p_dup. 
                   Example: mix='ABCabc'
    - list:   len_min, len_max: controls the length of the generated lists. 
    - time:   min, max
    - choice: choice: List or Series of elements to choose
    - perlin: freq: List of up to frequencies, see random_perlin for more details
              op: 'add' or 'mult', how the frequencies are linked together
              sc: scaling, default is 'max_abs', this scales -1..1, see function scale
    
    Examples:    
    =========
    random_series( 10, 'int')
    random_series( 10, 'string', len_min=1, len_max=2)
    '''
    if typ in   ['int','i']:
        myfunc = random_series_int
    elif typ in ['float','f']: 
        myfunc = random_series_float   
    elif typ in ['ascending','a']: 
        myfunc = random_series_ascending   
    elif typ in ['descending','d']: 
        myfunc = random_series_descending          
    elif typ in ['perlin','p']: 
        myfunc = random_series_perlin         
    elif typ in ['string','str','s']: 
        myfunc = random_series_string 
    elif typ in ['name','n']: 
        myfunc = random_series_name  
    elif typ in ['choice','c']: 
        myfunc = random_series_choice  
    elif typ in ['list','l']: 
        myfunc = random_series_list  
    elif typ in ['time','t','datetime']: 
        myfunc = random_series_datetime         
    elif typ in ['mix','m']: 
        myfunc = random_series_mix     
    elif typ in ['errorprone','e']: 
        myfunc = random_series_errorprone        
    
    p_dup = kwargs.get('p_dup',0)
    p_nan = kwargs.get('p_nan',0)    
    name  = kwargs.get('name',0)    
    assert p_dup >= 0
    assert p_dup <  1
    
    # unhashable type
    if typ in ['list','l','mix','m']:
        result = myfunc(size, **kwargs) 
        if p_nan>0:
            #print('p_nan',p_nan)
            result = result.apply(decorate, p=p_nan) # mit nan dekorieren
        return result
    
    # Keine Dups wählbar 
    if typ in ['ascending','a','descending','d','perlin','p','errorprone','e']: 
        result = myfunc(size, **kwargs)
        if p_nan>0:
            result = result.apply(decorate, p=p_nan) # mit nan dekorieren
        return result
    
    # Keine Dups gewünscht    
    elif p_dup == 0:
        result = myfunc(size*2, **kwargs).drop_duplicates().head(size)
        fortschritt = nunique(result)
        for i in range(1,15):
            #result = result.append(myfunc(size*2, **kwargs)).drop_duplicates().head(size)
            result = pd.concat( [result, myfunc(size*2, **kwargs)] )
            result = result.drop_duplicates().head(size)
            fortschritt_neu = nunique(result)
            if (fortschritt_neu == size) or (fortschritt_neu == fortschritt):
                break # Keine Dups mehr drin, oder kein weiterer Fortschritt
            fortschritt = fortschritt_neu
        # Ende
        result = pd.concat( [result, myfunc(size, **kwargs)] )  # noch ein letztes Mal auffüllen
        result = result.reset_index(drop=True).head(size)       # beschneiden und reindexieren
        if p_nan>0:
            result = result.apply(decorate, p=p_nan) # mit nan dekorieren
            if typ in  ['int','i']:
                result = result.astype('Int64')
        if typ in  ['string','str','s','name','n']:
            result = result.astype('string')
        return result


    # Dups gewünscht
    else:
        basis = myfunc(size*10, **kwargs).drop_duplicates().head(  int(size*(1-p_dup)  )  )      # dupfrei
        result = random_series( size, 'choice', choice=basis, p_nan=p_nan, p_dup=0, name=basis.name)
        if p_nan>0:
            result = result.apply(decorate, p=p_nan) # mit nan dekorieren         
            if typ in  ['int','i']: 
                result = result.astype('Int64')  
        if typ in  ['string','str','s','name','n']: 
            result = result.astype('string')  
        result = result.reset_index(drop=True)
        return result
    
    # Ende wird nie erreicht
    assert False
    
    
    

def decorate(skalar, p=0.2, special=np.nan):
    """ 
    Decorates a series with specials (e.g. NaNs), is applied with apply
    e.g. result = result.apply(decorate, p=0.1)               # decorate with 10% nan  
    e.g. result = result.apply(decorate, p=0.1, special='a')  # decorate with 10% 'a'.              
    """
    if p <= 0 or random.random() > p:
        return skalar
    else:
        if random.random() > 0.5:
            try:
                return skalar + [None]
            except:
                return special
        else:
            return special
        


    


# Fertige gemischte Zufallsdaten
def people(size=100, seed=None, framework=None):
    """
    Generates a DataFrame with random people data for testing.

    Args:
        size (int, optional): Number of people (rows). Defaults to 100.
        seed (int, optional): Seed for the random generator. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame with columns:
            - first_name (str): Random first names.
            - age (int): Random age between 20 and 42.
            - age_class (int): Age class (rounded to 10s).
            - postal_code (int): Random postal codes.
            - birthplace (str): Random birthplaces (Bremen or Berlin).
            - secret (str): Random strings.
            - features (set): Sets of random strings.
            - history (list): Lists of random strings.
    """

    if framework is None:
        framework = Config.get('FRAMEWORK')

    if framework not in ['pandas', 'polars']:
        raise ValueError(f"Framework '{framework}' is not supported. Only 'pandas' and 'polars' are allowed.")

    if size < 2:
        if size == 0 and framework == 'polars':
            return pl.DataFrame()
        result = people(size=3, seed=seed, framework=framework)
        return result.head(size)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    a = random_series( size, 'name',                                                 p_dup=0.3,  name='first_name',    )
    b = random_series( size, 'int',    min=20, max=30) + random_series( size, 'int', min=0, max=12)
    b.name = 'age'
    bb = b.apply(lambda x: int(x / 10) * 10 if not pd.isna(x) else np.nan)
    bb.name = 'age_class'    
    c = random_series( size, 'int',    min=10000, max=99999,            p_nan=0.02,  p_dup=0.3,  name='postal_code',  )
    d = random_series( size, 'choice', choice=['Bremen','Berlin'],      p_nan=0.3,   p_dup=0,    name='birthplace',   )
    e = random_series( size, 'string', len_min=5, len_max=10,           p_nan=0,     p_dup=0,    name='secret',       )
    f = random_series( size, 'string', len_min=0, len_max=5,            p_nan=0,     p_dup=0.2,  name='features',     ).apply(set)
    g = random_series( size, 'choice', choice=['ABC','ABCC','','abc','cba','Ax','AAA','ACCB','bbab'],  name='history',).apply(list)
    result = dataframe( (a,b,bb,c,d,e,f,g), verbose=False, framework='pandas' )
    result = change_datatype(result,        verbose=False)
    if framework == 'pandas':
        return result
    elif framework == 'polars':
        return pl.from_pandas(result)



# Fertige gemischte Zufallsdaten
def random_numbers(size=1000, seed=None, framework=None):
    """
    Returns a DataFrame with random numbers for testing.

    Args:
        size (int, optional): Number of rows. Defaults to 1000.
        seed (int, optional): Seed for the random generator. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame with columns 'A', 'B' (int), 'C', 'D', 'E' (float).
    """

    if framework is None:
        framework = Config.get('FRAMEWORK')

    if framework not in ['pandas', 'polars']:
        raise ValueError(f"Framework '{framework}' is not supported. Only 'pandas' and 'polars' are allowed.")

    if size < 2:
        result = random_numbers(size=3, seed=seed)
        return result.head(size)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    a = random_series( size, 'int',    min=20, max=30) + random_series( size, 'int', min=0, max=12)
    b = random_series( size, 'int',    min=10000, max=99999,  p_dup=0.3,)
    c = random_series( size, 'float',  decimals=3,            p_dup=0.3,)
    d = random_series( size, 'float',  decimals=3,            p_dup=0.3,) * 10
    e = random_series( size, 'ascending',                               )
    result = dataframe( (a,b,c,d,e),   verbose=False, framework='pandas' )
    result = change_datatype(result, verbose=False)
    result.columns = ['A', 'B', 'C', 'D', 'E']
    if framework == 'pandas':
        return result
    elif framework == 'polars':
        return pl.from_pandas(result)

    
def random_perlin( shape=(100,5), freq=[3,6,12,24], op='add', framework=None):
    '''
    Returns a Dataframe with Perlin Noise:
    - Every column looks like a random walk.
    - The columns correlate (the noise is 2-dim)
    * shape: The shape of the result
    * freq:  Up to 4 frequencies for the random walk.
             The frequencies are added or multipied with descending factors 1, 1/2, 1/4 and 1/16.
             A frequency of 1 means 1 maximum and 1 minimum.
             A frequency of 2 means 2 maxima  and 2 minima.                
             A frequency of 0 means, that nothing is added for the corresponding factor.
             E.g. freq=[1,0,0,100] gives a very low-frequency random walk 
             with a very slight admixture of high-frequency components. 
    * op:    'add' or 'mult', how the frequencies are linked together
                
    '''
    try:
        from perlin_noise import PerlinNoise
    except ImportError:
        print("The 'perlin_noise' package is required for the 'random_perlin' function. Please install it with 'pip install perlin_noise'.")
        return pd.DataFrame()

    if framework is None:
        framework = Config.get('FRAMEWORK')

    if framework not in ['pandas', 'polars']:
        raise ValueError(f"Framework '{framework}' is not supported. Only 'pandas' and 'polars' are allowed.")

    if isinstance(freq,int):
        freq = [freq]
    freq = list(freq) + [1, 1, 1, 1]
    freq = freq[:4]
    freq_nn = [ o if o>0  else 1 for o in freq ]
    noise1 = PerlinNoise(octaves=freq_nn[0])
    noise2 = PerlinNoise(octaves=freq_nn[1])
    noise3 = PerlinNoise(octaves=freq_nn[2])
    noise4 = PerlinNoise(octaves=freq_nn[3])

    xpix, ypix = shape
    pic = []
    for i in range(xpix):
        row = []
        for j in range(ypix):
            noise_val = noise1([i/xpix, j/ypix])
            if freq_nn[1] > 0:
                nv2 = 0.5 * noise2([i/xpix, j/ypix])
                if op=='add':
                    noise_val += nv2
                elif op=='mult':
                    noise_val *= nv2                   
            if freq_nn[2] > 0:
                nv2 = 0.25 * noise3([i/xpix, j/ypix])
                if op=='add':
                    noise_val += nv2
                elif op=='mult':
                    noise_val *= nv2  
            if freq_nn[3] > 0:                
                nv2 = 0.0625 * noise4([i/xpix, j/ypix])
                if op=='add':
                    noise_val += nv2
                elif op=='mult':
                    noise_val *= nv2                  

            row.append(noise_val)
        pic.append(row)
    result = dataframe(pic, verbose=False, framework='pandas')
    if framework == 'pandas':
        return result
    elif framework == 'polars':
        return pl.from_pandas(result)





# ==================================================================================================
# Interne Funktionen für zufällige Testdaten
# ==================================================================================================

def random_series_datetime(size, min='1900-01-01', max='2023-12-31', name='rnd_time', index=None, p_nan=0, p_dup=0):
    '''
    Returns a series of random datetime between min and max.
    '''
    min_ts = pd.Timestamp(min)
    max_ts = pd.Timestamp(max)

    # Calculate the range in nanoseconds
    range_ns = (max_ts - min_ts).value

    # Generate random numbers between 0 and 1
    random_numbers = np.random.rand(size)

    # Scale and shift the random numbers to the desired range
    random_ns = (random_numbers * range_ns) + min_ts.value

    # Convert to datetime
    random_dates = pd.to_datetime(random_ns, unit='ns')

    result = pd.Series(random_dates)
    result = result.rename(name)
    return result

    

def random_series_int( size, min=0, max=1000, name='rnd_int', index=None, p_nan=0, p_dup=0):
    '''
    Returns a series of random integers between min and max. min and max are both possible.
    '''
    result = pd.Series(np.random.randint(min, max+1, size))
    result = result.rename(name)        
    return result



def random_series_float(size, decimals=3, name='rnd_float', index=None, p_nan=0, p_dup=0):
    '''
    Returns a series of random floats between 0 and 1 with specified decimal places. 
    '''
    f1 = 10**decimals
    result = pd.Series(np.random.randint(0, f1, size) / f1)  
    result = result.rename(name)     
    return result



def random_series_perlin(size, name='rnd_perlin', freq=[3,6,12,24], op='add', sc='max_abs', index=None, p_nan=0, p_dup=0):
    '''
    Returns a series of random floats between -1 and 1 
    '''
    result = random_perlin( shape=(size,1), freq=freq, op=op, framework='pandas' ).A
    if sc:
        result = scale( result, sc)
    result = result.rename(name)     
    return result



# Liefert eine Series zufälliger Floats zwischen 0 und 1 
def random_series_monotonic(size, name='rnd_monotonic', ascending=True, index=None, p_nan=0, p_dup=0):
    '''
    Returns an monotonic, crooked Series of random floats
    '''
    # freq=[ 0.1, 0.2, 3,0]
    f1 = random.uniform( 0.01, 0.2 )
    f2 = random.uniform( 0.05, 0.2 )  
    f3 = random.uniform( 2.0,  4.0 )  
    freq = [f1,f2,f3,0]
    result = random_series_perlin( size, freq=freq, op='mult', sc='min_max')
    result = result.sort_values(ascending=ascending).reset_index(drop=True)
    result = result.rename(name)    
    return result



def random_series_ascending(size, name='rnd_ascending', index=None, p_nan=0, p_dup=0):
    '''
    Returns a series of random floats between 0 and 1 
    '''
    result = random_series_monotonic(size=size, name=name, ascending=True)
    return result



def random_series_descending(size, name='rnd_descending', index=None, p_nan=0, p_dup=0):
    '''
    Returns a series of random floats between 0 and 1 
    '''
    result = random_series_monotonic(size=size, name=name, ascending=False)
    return result



# .
def random_series_string(size, len_min=4, len_max=7, name='rnd_string', index=None, p_nan=0, p_dup=0, mix=None):
    '''
    Returns a series of random strings.
    * size: Length of the series
    * len_min: Minimum length of the random strings 
    * len_max: Maximum length of the random strings  
    * name: Name of the series
    * p_nan: Dead parameter
    * p_dup: 0..1, influences the width of the character set and thus the dup probability
    * mix: Explicit specification of the available character set. Overwrites p_dup. Example: mix='ABCabc'
    '''
    if not mix:
        mix = string.ascii_letters + string.digits + 'ÄÖÜäöüaeiou'
        ziel = int((1-p_dup)*len(mix))
        mix = mix[:ziel]        
    result = pd.Series([ bpy.random_str(size_min=len_min, size_max=len_max, mix=mix) for i in range(size) ])  
    result = result.rename(name) 
    result = result.astype('string')
    return result
   
    

def random_series_list(size, len_min=2, len_max=10, name='rnd_list', index=None, p_nan=0, p_dup=0):
    '''
    Returns a series of random lists.
    '''
    #länge = random.randint(len_min,len_max)
    #c = list(random_series_name(länge))
    result = pd.Series([ list(random_series_name(random.randint(len_min,len_max))) for i in range(0,size)])
    
    
   #result = pd.Series(pd.util.testing.rands_array(len_max, size))    
   #result = result.str.replace('[aeiouAEIOU]',' ', regex=True)
   #
   #def bearbeite_element( skalar ):
   #    return skalar.split() 

   #result = result.apply(bearbeite_element)      
    result = result.rename(name) 
    return result    



def random_series_choice(size, choice=[], name='rnd_choice', index=None, p_nan=0, p_dup=0):
    '''
    Returns a series of random selections
    '''
    if type(choice) is pd.core.series.Series:
        choice = list(choice)
    elif choice == []: 
        choice = list('abcde')
    result = pd.Series(np.random.choice(choice, size=size))      
    result = result.rename(name)     
    return result   
 


def random_series_name(size, name='rnd_name', index=None, p_nan=0, p_dup=0):
    '''
    Returns a series of random first names
    '''
    vornamen = ['Anna','Anna','Anna','Anna','Anna','Anna','Anna','Anna','Anna','Anna','Anna','Anna','Anna','Anna','Anna','Anna','Anna','Anna','Anna','Anna','Anna','Anna', 'Maria', 'Karl', 'Hans', 'Carl', 'Michael', 'Paul', 'Jan', 'Elisabeth', 'Alexander', 'Peter', 'Andre', 'Christian', 'Johanna', 'Marie', 'Thomas', 'Andreas', 'Walter', 'Johannes', 'Max', 'Werner', 'Matthias', 'Julia', 'Katharina', 'Martin', 'Daniel', 'Klaus', 'Stephan', 'Stefan', 'Claus', 'Emma', 'Christina', 'Tobias', 'Hermann', 'Wolfgang', 'Günter', 'Jürgen', 'Helmut', 'Ursula', 'Günther', 'Wilhelm', 'Heinrich', 'Tim', 'Kurt', 'Gerhard', 'Robert', 'Oliver', 'Nicole', 'Lisa', 'Heinz', 'Florian', 'Sebastian', 'Martha', 'Otto', 'Philipp', 'Eva', 'Mark', 'Horst', 'Helga', 'Sarah', 'Sven', 'Ernst', 'Markus', 'Georg', 'Erika', 'Uwe', 'Charlotte', 'Nils', 'Karin', 'Friedrich', 'Herbert', 'Lars', 'Frieda', 'Ingrid', 'Stefanie', 'Rolf', 'Nina', 'Sabine', 'Katrin', 'Susanne', 'Monika', 'Renate', 'Dennis', 'Patrick', 'Jens', 'Mathias', 'Gisela', 'Gertrud', 'Frank', 'Manfred', 'Franz', 'Marc', 'Christoph', 'Marcel', 'Bernd', 'Anja', 'Rudolf', 'Alfred', 'Jasmin', 'Lena', 'Felix', 'Alexandra', 'Harald', 'Petra', 'Willi', 'Sandra', 'Lukas', 'Melanie', 'Annika', 'Dieter', 'Claudia', 'Elke', 'Marion', 'Laura', 'Jana', 'Fritz', 'Brigitte', 'Simon', 'Christine', 'Heike', 'Barbara', 'Andrea', 'Mike', 'Julian', 'Marko', 'Gerda', 'Fabian', 'Maik', 'Joachim', 'Birgit', 'Caroline', 'Benjamin', 'Sara', 'Hanna', 'Margarete', 'Sonja', 'Johann', 'Marco', 'Rainer', 'Jessica', 'Margarethe', 'Ute', 'Richard', 'Jörg', 'Lea', 'Kai', 'Jonas', 'Ilse', 'Dominik', 'Tom', 'Timo', 'Helene', 'Hildegard', 'Jutta', 'Vanessa', 'Luise', 'Moritz', 'Lara', 'Erna', 'Niklas', 'Christa', 'Nico', 'Klara', 'Holger', 'David', 'Meik', 'Anne', 'Kerstin', 'Franziska', 'Ralf', 'Martina', 'Torsten', 'Kevin', 'Niels', 'Hannah', 'Ingeborg', 'Gerd', 'Gabriele', 'Berndt', 'Angelika', 'Lucas', 'Jannik', 'Dirk', 'Silvia', 'Rene', 'Philip', 'Phillip', 'Louise', 'Paula', 'Erich', 'Sophie', 'Marianne', 'Clara', 'Edith', 'Alina', 'Irmgard', 'Leon', 'Miriam', 'Carolin', 'Volker', 'Karsten', 'Maximilian', 'Willy', 'Leonie', 'Isabell', 'Maike', 'Lina', 'Eric', 'Tanja', 'Hannelore', 'Niclas', 'Inge', 'Christiane', 'Erik', 'Thorsten', 'Pia', 'Henry', 'Michelle', 'Walther', 'Kay', 'Yannik', 'Yannick', 'Yannic', 'Josef', 'Frida', 'Ole', 'Daniela', 'Manuela', 'Björn', 'Isabel', 'Luisa', 'Marcus', 'Ben', 'Anke', 'Jennifer', 'Finn', 'Natalie', 'Fynn', 'Sofie', 'Sascha', 'Isabelle', 'Sylvia', 'Robin', 'Emil', 'Jakob', 'Vincent', 'Kim', 'Nele', 'Elena', 'Carla', 'Bärbel', 'Gustav', 'Christel', 'Pascal', 'Karoline', 'Niko', 'Carsten', 'Malte', 'Viktoria', 'Lilly', 'Maja', 'Erwin', 'Nadine', 'Karla', 'Lilli', 'Sophia', 'Norbert', 'Antonia', 'Ruth', 'Jacob', 'Sina', 'Luka', 'Emilie', 'Ella', 'Bettina', 'Neele', 'Lennart', 'Lennard', 'Manuel', 'Victoria', 'Louisa', 'Anton', 'Katja', 'Käthe', 'Ida', 'Waltraud', 'Angela', 'Louis', 'Luca', 'Heidi', 'Anni', 'Emily', 'Celina', 'Eileen', 'Nathalie', 'Elfriede', 'Bruno', 'Marvin', 'Sabrina', 'Mia', 'Maya', 'Hendrik', 'Amelie', 'Silke', 'Karina', 'Sofia', 'Timm', 'Marina', 'Meike', 'Regina', 'Carina', 'Arthur', 'Rita', 'Melina', 'Melissa', 'Hannes', 'Sigrid', 'Bernhard', 'Hertha', 'Jonathan', 'Herta', 'Linda', 'Marlene', 'Olaf', 'Ulrich', 'Marta', 'Torben', 'Janina', 'Diana', 'Zoe', 'Astrid', 'Nick', 'Ulrike', 'Josephine', 'Swen', 'Albert', 'Leah', 'Josefine', 'Ingo', 'Elli', 'Lieselotte', 'Larissa', 'Luis', 'Annica', 'Arne', 'Jule', 'Aileen', 'Ayleen', 'Britta', 'Stephanie', 'Yvonne', 'Kathrin', 'Noah', 'Jacqueline', 'Bastian', 'Cornelia', 'Michaela', 'Joshua', 'Heiko', 'Lothar', 'Curt', 'Svenja', 'Jannick', 'Yasmin', 'Rebecca', 'Christopher', 'Chiara', 'Beate', 'Doris', 'Toni', 'Lucy', 'Antje', 'Theresa', 'Steffen', 'Leo', 'Kirsten', 'Simone', 'Mara', 'Adrian', 'Nora', 'Leoni', 'Nicolas', 'Nikolas', 'Mario', 'Catharina', 'Till', 'Axel', 'John', 'Detlef', 'Elias', 'Konstantin', 'Ina', 'Kira', 'Julius', 'Raphael', 'Merle', 'Bianca', 'Siegfried', 'Dominic', 'Jannis', 'Pauline', 'Lasse', 'Linus', 'Reinhard', 'Henri', 'Denis', 'Rosemarie', 'Benedikt', 'Celine', 'Rudolph', 'Helena', 'Annette', 'Stella', 'Evelyn', 'Tina', 'Vivien', 'Else', 'Dagmar', 'Mohammed', 'Anneliese', 'Adolf', 'Artur', 'Ali', 'Milena', 'Oskar', 'Maurice', 'Isabella', 'Phil', 'Elly', 'Mika', 'Marius', 'Fiona', 'Hugo', 'Theo', 'Anette','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom']
    result = pd.Series(np.random.choice(vornamen, size=size))    
    result = result.astype('string') 
    result = result.rename(name)      
    return result    




def random_series_mix(size, name='rnd_mix', p_nan=0, p_dup=0):
    '''
    Returns a series with various datatypes
    '''
    anz = int(size / 5) + 1
    
    b = random_series( anz, 'int',    min=-66666, max=66666,           )
    c = random_series( anz, 'float',  decimals=4,                      )
    d = random_series( anz, 'string', len_min=2, len_max=20,           )
    e = random_series( anz, 'name',                                    )
    f = random_series( anz, 'choice', choice=['Bremen','Bremerhaven'], )
    g = random_series( anz, 'list',                                    )
    h = random_series( anz, 'time',                                    )
      
    result = pd.concat( [b, c, d, e, f, g, h] )
    result = result.sample(frac=1).reset_index(drop=True).head(size)
    result = result.rename(name)    
    result.iloc[0] = {0}
    
    return result



def random_series_errorprone(size, name=None, values=None, index=None, p_nan=0, p_dup=0):
    '''
    Provides a random series of known issues
    '''
    if values is None:
        values = random.randint(0, 9)
    if index is None:
        index = random.randint(-1, 9)

    if values == 0:
        result =  repeat( [0], size )     
    elif values == 1:
        result =  repeat( [1], size )
    elif values == 2:
        result =  repeat( [-1], size )         
    elif values == 3:
        result =  repeat( [None], size )  
    elif values == 4:
        result =  repeat( [np.nan], size )  
    elif values == 5:
        result =  repeat( [-1,1], size )   
    elif values == 6:
        result =  repeat( [0,-1], size )
    elif values == 7:
        result =  repeat( [0,1], size )  
    elif values == 8:
        result =  repeat( [0,np.nan], size )    
    elif values == 9:
        result =  repeat( [42], size )              
    else:
        result =  repeat( [42], size ) 

    # name    
    if name is None:
        result.name = 'rnd_errorprone' + '_' + str(values)
    else:
        result.name = name

    # index
    if index == -1:
        return result 
    else:
        result.index = random_series_errorprone(size,values=index, index=-1)  
        result.index.name = None
        if result.name != name:
            result.name  += 'i' + str(index)              
        return result
            





