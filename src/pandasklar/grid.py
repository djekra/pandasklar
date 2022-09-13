import warnings

import pandas     as pd 
import numpy      as np

try:
    import qgrid
except ImportError:
    pass
    
try:
    import dtale
except ImportError:
    warnings.warn('dtale nicht importierbar')    
    
try:
    from termcolor import colored    
except ImportError:
    pass
    

# dtale Settings
import dtale.global_state as global_state
dtale_settings = {'theme':'light', 'max_column_width':400, }
global_state.set_app_settings( dtale_settings)



def set_grid(**kwargs):
    ''' Legt Settings für dtale fest.
    Bsp:
    set_grid(max_column_width=1000)
    '''
    global_state.set_app_settings( kwargs) 
    


# ==============================================================================================
# grid
# ==============================================================================================

def grid( df, mask=None, error='€€€', color='red', **kwargs ):
    '''
    Darstellung eines DataFrame mittels dtale.
    * df:     DataFrame zum Anzeigen
    * mask:   Maske oder Funktion, die vor der Anzeige angewendet wird
    * error:  Eine Fehlermeldung
    * kwargs: Argumente für dtale. Siehe https://github.com/man-group/dtale#instance-settings
              Die zuletzt gesetzten kwargs werden immer auch für alle folgenden grids verwendet.
    
    === Aufruf Beispiele ===
    grid(df,mask)                   # mask ist binäre Maske    
    grid(df,sample)                 # mask ist eine Funktion, die z.B. einen Teil der DS zurückgibt
    grid(df, max_column_width=200)  # Spaltenbreite begrenzen  

    === Zur Fehleranzeige nach einer Zählung ===
    error = check_mask(df, mask, 900, stop=False)
    grid(df, mask, error) 
    und später: raise_if(error)
    '''
    
    def print_color(msg, color):
        if color:
            try:
                msg = colored(msg, color, attrs=['reverse','bold'])
            except:
                pass   
        print(msg)  
    
    
    if df is None:
        print_color('Nothing to analyse', color)        
        return 
    
    if df.shape[0] == 0:
        print_color('No records', color)   
        return
    
    # verschiedene mask behandeln
    if type(mask) == pd.Series   or   type(mask) == np.ndarray:
        df_show = df[mask]
    elif callable(mask):   
        df_show = mask(df)
    else:
        df_show = df
        
    if df_show.shape[0] == 0:
        print_color('No records, mask filters them all away', color)
        return


    # widget erzeugen
    #widget = dtale.show(df_show,**kwargs) # funktioniert nicht  
    #widget.update_settings({'theme':'light', 'max_column_width':150}) # funktioniert auch nicht
    global_state.set_app_settings( kwargs)    
    widget = dtale.show(df_show)   

    # Anzahl printen
    if error: # gemeint ist, wenn error irgendwas enthält, z.B. den Standardwert'€€€'
        if (df_show.shape[0] != df.shape[0])  and  (df_show.shape[0] > 0):
            print( df_show.shape[0], 'Datensätze von', df.shape[0] )     
        else:
            print( df_show.shape[0], 'Datensätze' )
                
    # grid wird zur Anzeige potentieller Fehler genutzt, aber es gibt keine Fehler.
    # Dann wird auch kein Widget ausgegeben.
    else:       
        return 
        
    # Widget ausgeben
    if df_show.shape[0] > 0:
        return widget



    
    # ==============================================================================================
# grid
# ==============================================================================================

## grid(plan_merkmal, width=150, rows=9, column_definitions={'collect': {'maxWidth': 50,} }    )
#
#def grid( df, mask=None, error='€€€', **options ):
##def grid( df, mask=None, error='€€€', engine='dtale', width=100, rows=9, options=None, column_definitions={  'index': {'maxWidth': 50,}   } ):
#    """
#    Aufruf Beispiele:
#    grid(df,mask)        # mask ist binäre Maske    
#    grid(df,sample)      # mask ist eine Funktion, die z.B. einen Teil der DS zurückgibt
#
#    
#    Oder zur Fehleranzeige nach einer Zählung:
#    error = check_mask(df, mask, 900, stop=False)
#    grid(df, mask, error) 
#    und später: raise_if(error)
#    """
#    
#    if df is None:
#        return 'Nichts zum Analysieren'
#    if df.shape[0] == 0:
#        return 'Keine Datensätze'    
#    
#
#    
#    # verschiedene mask behandeln
#    if type(mask) == pd.Series   or   type(mask) == np.ndarray:
#        df_show = df[mask]
#    elif callable(mask):   
#        df_show = mask(df)
#    else:
#        df_show = df
#        
#    # verschiedene Engines
#    if engine == 'dtale':
#        widget = dtale.show(df_show, options)
#    elif engine == 'qgrid':
#        qgrid_default_options = {'forceFitColumns': False,  'defaultColumnWidth': width, 'enableColumnReorder': True, 'minVisibleRows': 3, 'maxVisibleRows': rows, 'filterable': False,}
#        qgrid_options = {**qgrid_default_options, **options}        
#        widget = qgrid.show_grid( df_show, grid_options=qgrid_options, column_definitions=column_definitions )     
#    else:
#        widget = df_show
#        
#    if df_show.shape[0] == 0:
#        return 'mask filtert alle Datensätze weg'         
#        
#    if error: # gemeint ist, wenn error irgendwas enthält, z.B. den Standardwert'€€€'
#        if (df_show.shape[0] != df.shape[0])  and  (df_show.shape[0] > 0):
#            print( df_show.shape[0], 'Datensätze von', df.shape[0] )     
#        else:
#            print( df_show.shape[0], 'Datensätze' )
#                
#    # Fehleranzeige, aber kein Fehler
#    if not error:       
#        return 
#        
#    # alles andere
#    if df_show.shape[0] > 0:
#        return widget
#
#