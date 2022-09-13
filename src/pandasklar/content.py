
import random, string
from collections import Counter

import numpy  as np
import pandas as pd 
import bpyth  as bpy

from pandasklar.config   import Config
from pandasklar.pandas   import reset_index, dataframe
from pandasklar.analyse  import get_different_rows, nunique, change_datatype


#from pandasklarbj_helper.python          import bpy.random_str
#from pandas._testing import rands_array # Zufallsstrings https://stackoverflow.com/questions/40461072/generate-random-strings-in-pandas
#
#import locale
#locale.setlocale(locale.LC_ALL, '') 
#

#################################################################################
# ...............................................................................
# Excel
# ...............................................................................
#################################################################################


#
def dataframe_to_excel(df, filename, tabspalte='', index=False, kontrolle=True):
    """ Schreibt einen Dataframe in eine Excel-Tabelle.
    
        Wenn tabspalte angegeben, wird das Dataframe anhand dieser Spalte auf 
        verschiedene TABs aufgesplittet.
        Kontrolliert durch wiedereinlesen und liefert die fehlerhaften Zeilen zurück
    """
    
    # Einzelner Tab
    if tabspalte == '':
        with pd.ExcelWriter(filename) as writer: 
            df.to_excel(writer, sheet_name='Pandas', index=index, inf_rep='__unendlich__', na_rep='__NaN__' )
            
    # Viele Tabs        
    else:
        alle_tabs = df[tabspalte].unique().tolist()    
        with pd.ExcelWriter(filename) as writer:  
            for counter, tab in enumerate(alle_tabs):
                mask = df[tabspalte] == tab
                tabname = tab #+ '___' + str(counter)
                df[mask].drop(tabspalte, axis=1).to_excel(writer, sheet_name=tabname, index=index)
                
    # Kontrolle
    if kontrolle:
        kontrolle = excel_to_dataframe(filename)
        diff = get_different_rows( df, kontrolle )
        #assert  diff.shape[0] == 0  
        return diff #.shape[0]    
  

                
                
# Die Umkehrung: Liefert den Dataframe zurück.   
# Wenn tabspalte angegeben, wird eine zusätzliche Spalte angefügt, die den Namen des Tab enthält.
# #https://stackoverflow.com/questions/44549110/python-loop-through-excel-sheets-place-into-one-df
#
def excel_to_dataframe(filename, tabspalte=''):
    
    # alle Tabs in einem Dictionary
    sheets_dict = pd.read_excel(filename, sheet_name=None ) 
    
    full_table = pd.DataFrame()
    
    for name, sheet in sheets_dict.items():
        if tabspalte != '':
            sheet[tabspalte] = name # zusätzliche Spalte
        sheet = sheet.rename(columns=lambda x: x.split('\n')[-1])
        #full_table = full_table.append(sheet)
        full_table = pd.concat( [full_table, sheet] )

    full_table.reset_index(inplace=True, drop=True)  
    return full_table







#################################################################################
# ...............................................................................
# Zufällige Testdaten
# ...............................................................................
#################################################################################


def random_series(size, typ, **kwargs):
    """ Liefert eine Series von Zufallsdaten.

        Die Argumente werden dabei an die zuständigen Funktionen für Series weitergereicht.
        Generelle Argumente sind:
        * name
        * p_nan: Wert 0..1 legt fest, wieviele NaNs eingestreut werden
        * p_dup: Wert 0..1 legt fest, wieviele Dups enthalten sind.
        Beispiele:
        random_series( 10, 'int')
        random_series( 10, 'string', len_min=1, len_max=2)
    """

    if typ in  ['int','i']: 
        myfunc = random_series_int
    elif typ in ['float','f']: 
        myfunc = random_series_float   
    elif typ in ['string','str','s']: 
        myfunc = random_series_string 
    elif typ in ['name','n']: 
        myfunc = random_series_name  
    elif typ in ['choice','c']: 
        myfunc = random_series_choice  
    elif typ in ['list','l']: 
        myfunc = random_series_list     
    elif typ in ['mix','m']: 
        myfunc = random_series_mix           
    
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
    
    # Keine Dups gewünscht
    if p_dup == 0:
        result = myfunc(size*2, **kwargs).drop_duplicates().head(size) 
        fortschritt = nunique(result)
        for i in range(1,15):
            #result = result.append(myfunc(size*2, **kwargs)).drop_duplicates().head(size)
            result = pd.concat( [result, myfunc(size*2, **kwargs)] )
            result = result.drop_duplicates().head(size)
                
            fortschritt_neu = nunique(result)
            #print(size, i, fortschritt, fortschritt_neu)
            if (fortschritt_neu == size) or (fortschritt_neu == fortschritt):
                break # Keine Dups mehr drin, oder kein weiterer Fortschritt
            fortschritt = fortschritt_neu
        # Ende    
        #result = result.append(myfunc(size, **kwargs))         # noch ein letztes Mal auffüllen        
        result = pd.concat( [result, myfunc(size, **kwargs)] )  # noch ein letztes Mal auffüllen
        result = result.reset_index(drop=True).head(size)       # beschneiden und reindexieren
        if p_nan>0:
            #print('p_nan',p_nan)
            result = result.apply(decorate, p=p_nan) # mit nan dekorieren
            if typ in  ['int','i']: 
                result = result.astype('Int64')
        if typ in  ['string','str','s']: 
            result = result.astype('string')                    
        return result
    
    # Dups gewünscht
    else:
        basis = myfunc(size*10, **kwargs).drop_duplicates().head(  int(size*(1-p_dup)  )  )      # dupfrei
        result = random_series( size, 'choice', choice=basis, p_nan=p_nan, p_dup=0, name=name)
        if p_nan>0:
            result = result.apply(decorate, p=p_nan) # mit nan dekorieren         
            if typ in  ['int','i']: 
                result = result.astype('Int64')  
        if typ in  ['string','str','s','name','n']: 
            result = result.astype('string')                   
        return result.reset_index(drop=True)
    
    # Ende wird nie erreicht
    assert False
    
    
    

def decorate(skalar, p=0.2, special=np.nan):
    """ Garniert eine Series mit Specials (z.B. NaNs), wird mit apply angewendet
        z.B. result = result.apply(decorate, p=0.1)              # mit 10% nan dekorieren  
        z.B. result = result.apply(decorate, p=0.1, special='a') # mit 10% 'a' dekorieren          
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
def leute(anz=100):
    """ Liefert ein DataFrame zu Testzwecken """
    a = random_series( anz, 'name',                                                 p_dup=0.3,  name='vorname' )
    b = random_series( anz, 'int',    min=20, max=30) + random_series( anz, 'int', min=0, max=12) 
    b.name = 'alter'
    bb = b.apply(lambda x: int(x/10)*10 )  
    bb.name = 'altersklasse'    
    c = random_series( anz, 'int',    min=10000, max=99999,            p_nan=0.02,  p_dup=0.3,  name='plz')    
    d = random_series( anz, 'choice', choice=['Bremen','Berlin'],      p_nan=0.3,   p_dup=0,    name='geburtsstadt')
    e = random_series( anz, 'string', len_min=5, len_max=10,           p_nan=0,     p_dup=0,    name='passwort')
    f = random_series( anz, 'string', len_min=0, len_max=5,            p_nan=0,     p_dup=0.2,  name='merkmale').apply(set)    
    g = random_series( anz, 'choice', choice=['ABC','ABCC','','abc','cba','Ax','AAA','ACCB','bbab'],  name='history').apply(list)  
    result = dataframe( (a,b,bb,c,d,e,f,g) )  
    result = change_datatype(result, verbose=False)
    return result       



# Fertige gemischte Zufallsdaten
def zufallsdaten(anz=1000):
    """ Liefert ein DataFrame zu Testzwecken """
    a = random_series( anz, 'int',    min=20, max=30) + random_series( anz, 'int', min=0, max=12) 
    b = random_series( anz, 'int',    min=10000, max=99999,  p_dup=0.3)
    c = random_series( anz, 'float',  decimals=3, p_dup=0.3)    
    d = random_series( anz, 'float',  decimals=3, p_dup=0.3) * 10   
    result = dataframe( (a,b,c,d) ) 
    result = change_datatype(result, verbose=False)    
    return result  
        

    
    





# ==================================================================================================
# Interne Funktionen für zufällige Testdaten
# ==================================================================================================
    

# Liefert eine Series zufälliger Integerzahlen zwischen min und max. min und max sind beide möglich.
def random_series_int( size, min=0, max=1000, name='random', p_nan=0, p_dup=0):
    result = pd.Series(np.random.randint(min, max+1, size))
    result = result.rename(name)        
    return result



# Liefert eine Series zufälliger Floats zwischen 0 und 1 mit festgelegten Dezimalstellen 
def random_series_float(size, decimals=3, name='random', p_nan=0, p_dup=0):
    f1 = 10**decimals
    result = pd.Series(np.random.randint(0, f1, size) / f1)  
    result = result.rename(name)     
    return result



# .
def random_series_string(size, len_min=4, len_max=7, name='random', p_nan=0, p_dup=0, mix=None):
    '''
    Liefert eine Series zufälliger Strings.
    * size: Länge der Series
    * len_min: Mindestlänge der Zufallsstrings 
    * len_max: Maximallänge der Zufallsstrings  
    * name: Name der Series
    * p_nan: Toter Parameter
    * p_dup: 0..1, beeinflusst die Breite des Zeichensatzes und damit die Dup-Wahrscheinlichkeit
    * mix: Explizite Angabe des verfügbaren Zeichensatzes. Überschreibt p_dup. Beispiel: mix='ABCabc'
    '''
    if not mix:
        mix = string.ascii_letters + string.digits + 'ÄÖÜäöüaeiou'
        ziel = int((1-p_dup)*len(mix))
        mix = mix[:ziel]        
    result = pd.Series([ bpy.random_str(size_min=len_min, size_max=len_max, mix=mix) for i in range(size) ])  
    result = result.rename(name) 
    result = result.astype('string')
    return result
   
    
# Liefert eine Series zufälliger Listen.
def random_series_list(size, len_min=2, len_max=10, name='random', p_nan=0, p_dup=0):
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


# Liefert eine Series zufälliger Auswahlen
def random_series_choice(size, choice=[], name='random', p_nan=0, p_dup=0):
    if type(choice) is pd.core.series.Series:
        choice = list(choice)
    elif choice == []: 
        choice = list('abcde')
    result = pd.Series(np.random.choice(choice, size=size))      
    result = result.rename(name)     
    return result   
 

# Liefert eine Series zufälliger Vornamen    
def random_series_name(size, name='random', p_nan=0, p_dup=0):
    vornamen = ['Anna','Anna','Anna','Anna','Anna','Anna','Anna','Anna','Anna','Anna','Anna','Anna','Anna','Anna','Anna','Anna','Anna','Anna','Anna','Anna','Anna','Anna', 'Maria', 'Karl', 'Hans', 'Carl', 'Michael', 'Paul', 'Jan', 'Elisabeth', 'Alexander', 'Peter', 'Andre', 'Christian', 'Johanna', 'Marie', 'Thomas', 'Andreas', 'Walter', 'Johannes', 'Max', 'Werner', 'Matthias', 'Julia', 'Katharina', 'Martin', 'Daniel', 'Klaus', 'Stephan', 'Stefan', 'Claus', 'Emma', 'Christina', 'Tobias', 'Hermann', 'Wolfgang', 'Günter', 'Jürgen', 'Helmut', 'Ursula', 'Günther', 'Wilhelm', 'Heinrich', 'Tim', 'Kurt', 'Gerhard', 'Robert', 'Oliver', 'Nicole', 'Lisa', 'Heinz', 'Florian', 'Sebastian', 'Martha', 'Otto', 'Philipp', 'Eva', 'Mark', 'Horst', 'Helga', 'Sarah', 'Sven', 'Ernst', 'Markus', 'Georg', 'Erika', 'Uwe', 'Charlotte', 'Nils', 'Karin', 'Friedrich', 'Herbert', 'Lars', 'Frieda', 'Ingrid', 'Stefanie', 'Rolf', 'Nina', 'Sabine', 'Katrin', 'Susanne', 'Monika', 'Renate', 'Dennis', 'Patrick', 'Jens', 'Mathias', 'Gisela', 'Gertrud', 'Frank', 'Manfred', 'Franz', 'Marc', 'Christoph', 'Marcel', 'Bernd', 'Anja', 'Rudolf', 'Alfred', 'Jasmin', 'Lena', 'Felix', 'Alexandra', 'Harald', 'Petra', 'Willi', 'Sandra', 'Lukas', 'Melanie', 'Annika', 'Dieter', 'Claudia', 'Elke', 'Marion', 'Laura', 'Jana', 'Fritz', 'Brigitte', 'Simon', 'Christine', 'Heike', 'Barbara', 'Andrea', 'Mike', 'Julian', 'Marko', 'Gerda', 'Fabian', 'Maik', 'Joachim', 'Birgit', 'Caroline', 'Benjamin', 'Sara', 'Hanna', 'Margarete', 'Sonja', 'Johann', 'Marco', 'Rainer', 'Jessica', 'Margarethe', 'Ute', 'Richard', 'Jörg', 'Lea', 'Kai', 'Jonas', 'Ilse', 'Dominik', 'Tom', 'Timo', 'Helene', 'Hildegard', 'Jutta', 'Vanessa', 'Luise', 'Moritz', 'Lara', 'Erna', 'Niklas', 'Christa', 'Nico', 'Klara', 'Holger', 'David', 'Meik', 'Anne', 'Kerstin', 'Franziska', 'Ralf', 'Martina', 'Torsten', 'Kevin', 'Niels', 'Hannah', 'Ingeborg', 'Gerd', 'Gabriele', 'Berndt', 'Angelika', 'Lucas', 'Jannik', 'Dirk', 'Silvia', 'Rene', 'Philip', 'Phillip', 'Louise', 'Paula', 'Erich', 'Sophie', 'Marianne', 'Clara', 'Edith', 'Alina', 'Irmgard', 'Leon', 'Miriam', 'Carolin', 'Volker', 'Karsten', 'Maximilian', 'Willy', 'Leonie', 'Isabell', 'Maike', 'Lina', 'Eric', 'Tanja', 'Hannelore', 'Niclas', 'Inge', 'Christiane', 'Erik', 'Thorsten', 'Pia', 'Henry', 'Michelle', 'Walther', 'Kay', 'Yannik', 'Yannick', 'Yannic', 'Josef', 'Frida', 'Ole', 'Daniela', 'Manuela', 'Björn', 'Isabel', 'Luisa', 'Marcus', 'Ben', 'Anke', 'Jennifer', 'Finn', 'Natalie', 'Fynn', 'Sofie', 'Sascha', 'Isabelle', 'Sylvia', 'Robin', 'Emil', 'Jakob', 'Vincent', 'Kim', 'Nele', 'Elena', 'Carla', 'Bärbel', 'Gustav', 'Christel', 'Pascal', 'Karoline', 'Niko', 'Carsten', 'Malte', 'Viktoria', 'Lilly', 'Maja', 'Erwin', 'Nadine', 'Karla', 'Lilli', 'Sophia', 'Norbert', 'Antonia', 'Ruth', 'Jacob', 'Sina', 'Luka', 'Emilie', 'Ella', 'Bettina', 'Neele', 'Lennart', 'Lennard', 'Manuel', 'Victoria', 'Louisa', 'Anton', 'Katja', 'Käthe', 'Ida', 'Waltraud', 'Angela', 'Louis', 'Luca', 'Heidi', 'Anni', 'Emily', 'Celina', 'Eileen', 'Nathalie', 'Elfriede', 'Bruno', 'Marvin', 'Sabrina', 'Mia', 'Maya', 'Hendrik', 'Amelie', 'Silke', 'Karina', 'Sofia', 'Timm', 'Marina', 'Meike', 'Regina', 'Carina', 'Arthur', 'Rita', 'Melina', 'Melissa', 'Hannes', 'Sigrid', 'Bernhard', 'Hertha', 'Jonathan', 'Herta', 'Linda', 'Marlene', 'Olaf', 'Ulrich', 'Marta', 'Torben', 'Janina', 'Diana', 'Zoe', 'Astrid', 'Nick', 'Ulrike', 'Josephine', 'Swen', 'Albert', 'Leah', 'Josefine', 'Ingo', 'Elli', 'Lieselotte', 'Larissa', 'Luis', 'Annica', 'Arne', 'Jule', 'Aileen', 'Ayleen', 'Britta', 'Stephanie', 'Yvonne', 'Kathrin', 'Noah', 'Jacqueline', 'Bastian', 'Cornelia', 'Michaela', 'Joshua', 'Heiko', 'Lothar', 'Curt', 'Svenja', 'Jannick', 'Yasmin', 'Rebecca', 'Christopher', 'Chiara', 'Beate', 'Doris', 'Toni', 'Lucy', 'Antje', 'Theresa', 'Steffen', 'Leo', 'Kirsten', 'Simone', 'Mara', 'Adrian', 'Nora', 'Leoni', 'Nicolas', 'Nikolas', 'Mario', 'Catharina', 'Till', 'Axel', 'John', 'Detlef', 'Elias', 'Konstantin', 'Ina', 'Kira', 'Julius', 'Raphael', 'Merle', 'Bianca', 'Siegfried', 'Dominic', 'Jannis', 'Pauline', 'Lasse', 'Linus', 'Reinhard', 'Henri', 'Denis', 'Rosemarie', 'Benedikt', 'Celine', 'Rudolph', 'Helena', 'Annette', 'Stella', 'Evelyn', 'Tina', 'Vivien', 'Else', 'Dagmar', 'Mohammed', 'Anneliese', 'Adolf', 'Artur', 'Ali', 'Milena', 'Oskar', 'Maurice', 'Isabella', 'Phil', 'Elly', 'Mika', 'Marius', 'Fiona', 'Hugo', 'Theo', 'Anette','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Tanja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Anja','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom','Tom']  
    result = pd.Series(np.random.choice(vornamen, size=size))    
    result = result.rename(name)    
    result = result.astype('string')    
    return result    




# Liefert eine Series zufälliger Listen.
def random_series_mix(size, name='random', p_nan=0, p_dup=0):
    
    anz = int(size / 5) + 1
    
    b = random_series( anz, 'int',    min=-66666, max=66666,          )    
    c = random_series( anz, 'float',  decimals=4,                     )
    d = random_series( anz, 'string', len_min=2, len_max=20,          )
    e = random_series( anz, 'name',                                   )
    f = random_series( anz, 'choice', choice=['Bremen','Bremerhaven'] )
    g = random_series( anz, 'list',                                   )    
      
    #result = b.append(c).append(d).append(e).append(f).append(g)
    result = pd.concat( [b, c, d, e, f, g] )
    result = result.sample(frac=1).reset_index(drop=True).head(size)
    result = result.rename(name)    
    result.iloc[0] = {0}
    
    return result

