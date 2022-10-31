
import numpy  as np
import pandas as pd

try:
    import seaborn
except ImportError:
    pass
    #print('no seaborn, plot will not work')  
##seaborn.set()
    
import matplotlib.pyplot as plt   
import warnings

from .config   import Config
from .pandas   import rename_col

    

    
# ==============================================================================================
# plot
# ==============================================================================================

# Todo: automatisch ausdünnen, wenn zuviele Datenpunkte
#
def plot(   df1, df2=None, x='--', size=(16, 4), palette=('rainbow','tab10'), line=(1,1)   ):
    ''' 
    Plots DataFrames or Series.
    * df1, df2: The first two parameters are DataFrames or Series. 
                If there are two, they get separate y-axes.
    * x:        Which column contains the x-axis? 
                x=='index' -> The index is used as x.    
                If no x is given, x is tried to be guessed. 
                If no suitable column is found, the index is used as x.
                A column is considered suitable if it is called 'x', 'X' or 'index'.
    size:       Width and height of the plot as tuples
    palette:    The two palettes as tuple or sting
    line:       The line thickness as tuple or number    
    '''   
    
    # filter findfont warnings
    warnings.filterwarnings( 'ignore', module = 'findfont' )    
    
    if isinstance(df1, pd.Series):
        df1 = pd.DataFrame(df1).reset_index()
        x='index'
        
    if isinstance(df2, pd.Series):      
        df2 = pd.DataFrame(df2).reset_index() 
      
    
    # Einspaltiges DataFrame?
    #if df1.shape[1] == 1:       
    #    df1 = df1.reset_index()
    #    col0 = list(df1.columns)[0]
    #    df1 = rename_col(df1, col0, 'x')   
    #    x = 'x'
    #
    #if not df2 is None:
    #    if df2.shape[1] == 1:               
    #        df2 = df2.reset_index()
    #        col0 = list(df2.columns)[0]
    #        df2 = rename_col(df2, col0, 'x')   
    #        #x = 'x2'
    
    # Kein x festgelegt aber es gibt eine Spalte namens x, X oder index
    schnittmenge = set(df1.columns)  &  {'x','X','index'}
    if x=='--':
        schnittmenge = set(df1.columns)  &  {'x','X','index'}        
        if len(schnittmenge)==1:
            x = next(iter(schnittmenge))  # x festlegen
    
    # Erste Spalte untersuchen
    if x=='--':
        col0 = list(df1.columns)[0]
        if list(df1[col0]) == list(df1.index):
            x = col0
        else:
            x = 'index'
      
    # x=='index'
    if x=='index' and not 'index' in df1.columns:
        df1 = df1.reset_index()            
        x='index'
        if not df2 is None:
            if not 'index' in df2.columns:
                df2 = df2.reset_index()              
   
    #print('x =',x)        
        
    # palette1 und palette2  festlegen    
    if isinstance(palette, tuple):
        palette1 = palette[0]
        palette2 = palette[1]
    else:
        palette1 = palette
        
    # line1 und line2  festlegen    
    if isinstance(line, tuple):
        line1 = line[0]
        line2 = line[1]
    else:
        line1 = line      
    
    # Erste y-Achse  

    plt.figure(figsize=size)
    seaborn.lineplot(x=x, 
                 y='value', 
                 hue='variable', 
                 palette=palette1,
                 linewidth=line1,
                 data=pd.melt(df1, [x])
                )       
    plt.legend(loc='upper left')
    plt.ylabel( ' '.join([str(c) for c in df1.columns[1:]]) )
    
    if df2 is None:
        return
    
    # Zweite y-Achse
    ax2 = plt.twinx()
    seaborn.lineplot(x=x, 
                 y='value', 
                 hue='variable', 
                 palette=palette2,
                 linewidth=line2,                 
                 data=pd.melt(df2, [x]),
                 ax=ax2
                )     

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)  
    plt.ylabel( ' '.join([str(c) for c in df2.columns[1:]]) )
    
                



