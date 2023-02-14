import warnings

import pandas     as pd 
import numpy      as np
   
try:
    from termcolor import colored    # um Fehlermeldungen rot auszugeben
except ImportError:
    pass

try:
    import dtale
except ImportError:
    warnings.warn('dtale not present. Think about pip install dtale')    

from .config   import Config
from .analyse  import col_names    
    
    
    
# ==============================================================================================
# dtale Settings
# ==============================================================================================

try:
    import dtale.global_state as global_state
except:
    pass


def set_grid(**kwargs):
    ''' 
    Sets settings for dtale.
    Ex:
    set_grid(max_column_width=1000)
    '''
    #print(kwargs)
    try:
        global_state.set_app_settings( kwargs) 
    except:
        pass        
    
    
    
def reset_grid():
    '''
    Resets settings for dtale to default values.
    '''
    try:
        set_grid( max_column_width=300, 
                  theme='light')
    except:
        pass
    
    
try:    
    reset_grid()   
except:
    pass    
    
    
    
    
# ==============================================================================================
# grid
# ==============================================================================================

def grid( df, mask=None, error='€€€', color='blue', **kwargs ):
    '''
    Visualization of a DataFrame using dtale.
    * df:     DataFrame to show
    * mask:   Binary mask, function or Searchstring to reduce the number of rows
    * error:  Error message
    * color:  Color of the error message.
    * kwargs: Options for dtale. See https://github.com/man-group/dtale#instance-settings
              Caution: This will affect all grids in this notebook...
    dtale may not work in a multiscreen setting on windows.
    
    === Examples ===
    grid(df)                        # show all rows
    grid(df,mask)                   # show with binary mask    
    grid(df,sample)                 # show with functionally mask

    === For error indication after check_mask ===
    error = check_mask(df, mask, 900, stop=False)
    grid(df, mask, error) 
    and later: raise_if(error)
    '''
    
    def print_color(msg, color):
        if color:
            try:
                msg = colored(msg, color, attrs=['reverse','bold'])
            except:
                pass   
        print(msg)  
    
    
    if df is None:
        print_color('Nothing to show', 'red')        
        return 
    
    if df.shape[0] == 0:
        print_color('No rows', color)   
        return
    
    # verschiedene mask behandeln
    if isinstance(mask, pd.Series)  or  isinstance(mask, np.ndarray):
        df_show = df[mask]
    elif callable(mask):   
        df_show = mask(df)
    elif isinstance(mask, str):
        df_show = search_str(df, mask)
    else:
        df_show = df
        
    if df_show.shape[0] == 0:
        print_color('No rows, mask filters them all away', color)
        return

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)   
        try:
            widget = dtale.show(df_show,**kwargs) # funktioniert nicht für alle options. Z.B. nicht für max_column_width
        except:
            widget = None

    # Anzahl printen
    if error: # gemeint ist, wenn error irgendwas enthält, z.B. den Standardwert'€€€'
        if (df_show.shape[0] != df.shape[0])  and  (df_show.shape[0] > 0):
            print( df_show.shape[0], 'rows out of', df.shape[0] )     
        else:
            print( df_show.shape[0], 'rows' )
                
    # grid wird zur Anzeige potentieller Fehler genutzt, aber es gibt keine Fehler.
    # Dann wird auch kein Widget ausgegeben.
    else:       
        return 
        
    # Widget ausgeben
    if df_show.shape[0] > 0:
        if widget is not None:
            return widget
        else:
            return df_show


    
    
# ==============================================================================================
# search_str
# ==============================================================================================    
    
def search_str(df, find, without=[]):
    """ 
    Searches all str columns of a dataframe.
    Useful for development and debugging.
    * find:    What is to be found?                  String or list of strings.
    * without: Which columns should not be searched? String or list of strings.    
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
        
    
    
# ==============================================================================================
# check_mask
# ==============================================================================================

def check_mask(df, mask, expectation_min=None, expectation_max=None, msg='', stop=True, verbose=None):
    """ 
    Count rows filtered by a binary mask.
    Raises an error, if the number is unexpected.

    Examples:
    ==========
    check_mask( df, mask )         # just show the number of rows   
    check_mask( df, mask, 2000 )   # checks for about 2000 rows (if not 0: not more than double, not less than half)
    check_mask( df, mask, 0 )      # checks for exactly 0 rows    
    check_mask( df, mask, 10, 50)  # checks for 10..50 rows

    Example with later raise:
    =========================
    error = check_mask(df, mask, 214, stop=False)        
    grid(df, mask, error)        
    bpy.raise_if(error)
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
    
    # expectation_max fehlt
    if not (expectation_min is None) and (expectation_max is None):
        if expectation_min == 0:
            e_min = 0
            e_max = 0
        else:
            e_min = int(expectation_min * 0.5) # Verdoppelung oder Halbierung wird toleriert
            e_max = int(expectation_min * 2.0) + 1        
        return check_mask(df, mask, e_min, e_max, msg=msg, stop=stop, verbose=verbose) 
    
    if type(mask) == pd.Series   or   type(mask) == np.ndarray:
        anz_ds = df[mask].shape[0]
    else:
        anz_ds = df.shape[0]
        
    # expectation_min fehlt
    if expectation_min is None:     
        print('check_mask:', msg, "{0} rows".format(anz_ds).strip()  )   
        return
    
    # wurde vielleicht schon abgearbeitet?
    #if (anz_ds == 0)   and   (expectation_min > 0) and verbose:
    #    print_red(msg + " WARNING: {0} rows, but it should be at least {1}".format(anz_ds, expectation_min)  ) 
    #    return 
    
    if anz_ds > expectation_max:
        error = " ERROR: {0} rows, but it should be a maximum of {1}".format(anz_ds, expectation_max)  
    elif (anz_ds < expectation_min):
        error = " ERROR: {0} rows, but it should be at least {1}".format(anz_ds, expectation_min)     
    else:
        if verbose:
            print('check_mask:',msg, "{0} rows".format(anz_ds).strip()  )  
        
    #raise_later  
    if (anz_ds > expectation_max)  or (anz_ds < expectation_min):
        if stop:
            raise Exception(  (msg + error).strip()  )
            
        else:
            print_red(   (msg + error).strip()  )
            return error.strip()
    
    
