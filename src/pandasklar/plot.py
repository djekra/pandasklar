
import numpy  as np
import pandas as pd

try:
    import seaborn
except ImportError:
    print('seaborn nicht importierbar')  
##seaborn.set()
    
import matplotlib.pyplot as plt    

from pandasklar.config   import Config
from pandasklar.pandas   import rename_col

    

    
# ==============================================================================================
# plot
# ==============================================================================================

# Todo: automatisch ausdünnen, wenn zuviele Datenpunkte
#
def plot(   df1, df2=None, x='--', size=(16, 4), palette=('rainbow','tab10'), line=(1,1)   ):
    """ Plottet das Dataframe oder eine Series.
    df1, df2:      Die ersten beiden Parameter sind Dataframes oder Series. Sind es zwei, bekommen sie separate y-Achsen.
    x:             Welche Spalte enthält die x-Achse? Wenn kein x angegeben ist, wird die erste Spalte als x angenommen.
    size:          Breite und Höhe des Plots als Tupel
    palette:       die beiden Paletten als Tuple oder Sting
    line:          die Liniendicke als Tuple oder Zahl
    
    """    
    
    if isinstance(df1, pd.Series):
        df1 = pd.DataFrame(df1)
        
    if isinstance(df2, pd.Series):      
        df2 = pd.DataFrame(df2)        
    
    # Einspaltiges DataFrame?
    if df1.shape[1] == 1:       
        df1 = df1.reset_index()
        col0 = list(df1.columns)[0]
        df1 = rename_col(df1, col0, 'x')      
    
    if not df2 is None:
        if df2.shape[1] == 1:               
            df2 = df2.reset_index()
            col0 = list(df2.columns)[0]
            df2 = rename_col(df2, col0, 'x')   
        
    # x festlegen
    if x=='--':
        x = str(df1.columns[0])
        
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
    
                



