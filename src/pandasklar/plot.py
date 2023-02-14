
import pandas as pd
import matplotlib.pyplot as plt   
import warnings, logging

from .config   import Config
from .analyse  import col_names
from .pandas   import dataframe
from .subsets  import sample




def plot( data, x=None, secondary_y=False, ylabel=None, subplots=False, figsize=None, inaccurate_limit=10000, **kwargs ):
    """
    Plots data. All parameters are passed to pandas.DataFrame.plot, but 
    * data:             The data to plot. DataFrame, or list of Series, or any other data 
                        which can be converted to DataFrame by the pandasklar dataframe function.
                        Non-numeric columns are ignored (even for column positions). 
    * x:                Which column to be used as x-axis. Column name or column position.
                        x=None -> The index is used as x.   
    * secondary_y:      Which columns to plot on the secondary y-axis. 
                        Column name, column position or list of column names.
    * figsize:          Size of a figure object. Default is (16,3) or (16,4), depending on the data.
    * inaccurate_limit: From what size should the data be thinned randomly.
                        Uses pandasklars sample function, so minimums and maximums are kept.     

    """
    
    logging.getLogger('matplotlib.font_manager').disabled = True
    
    # DataFrame erzwingen
    if not isinstance(data, pd.DataFrame):
        data = dataframe(data, verbose=False)
    
    # nur numerische Spalten
    cols = col_names(data, query='is_numeric==True')
    data = data[cols]
    
    # ausdünnen, falls nötig
    data = sample(data,inaccurate_limit)    
    
    # Parameter x
    if x is not None:
        if isinstance(x, (int, float, complex)) and not isinstance(x, bool):
            x = data.columns[x]
        data = data.set_index(x)
    
    # Parameter secondary_y
    if type(secondary_y) is str:
        secondary_y = [secondary_y] #let the command take a string or list    
    elif isinstance(secondary_y, (int, float, complex)) and not isinstance(secondary_y, bool):
        secondary_y = [data.columns[secondary_y]]
        
    # Parameter ylabel
    if ylabel is None and secondary_y != False:
        ylabel = ',  '.join(secondary_y) 
        
    # Parameter figsize   
    if figsize is None:
        if secondary_y  or  data.shape[1] > 2  or  subplots:
            figsize = (16,4)
        else:
            figsize = (16,3)        
    
    return data.plot( secondary_y=secondary_y, ylabel=ylabel, subplots=subplots, figsize=figsize, **kwargs)

    

        
    

        

    











