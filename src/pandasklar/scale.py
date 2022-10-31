    
import random    
import numpy  as np
from .config   import Config
from .pandas  import repeat
    
# ==================================================================================================
# scale
# ==================================================================================================
     
    
def scale(series, method, powerfactor=1, almostzero=0.00000001, verbose=None ):
    '''
    Scales all values of a numeric series to a defined value range.
    * method must be 'max_abs','min_max','min_max_robust',
      'rel','mean','median','compare_median','rank' or 'random'
    * powerfactor is an additional parameter for scaling 'rank'

    ## scale method='max_abs'
    * scales every value with a fixed factor
    * one finds: Every scaled value is somewhere in the range -1..1
    * series_scaled.max() is often 1
    * series_scaled.min() can be anything -1..0.999    

    ## scale method='min_max'
    * forces all scaled values to fit the full range 0..1 (closed interval)
    * one finds: series_scaled.min() == 0
    * one finds: series_scaled.max() == 1 if there are more than 2 different values

    ## scale method='min_max_robust'
    * like min_max, but robust against outlier values. 
      Works with quantile(0.01) and quantile(0.99) instead of min() and max().
      The scaling is therefore not determined by the outliers.
    * scales 98% of the values to fit the range 0..1
    * one finds: series_scaled.min() <= 0 but not much lower  than -0.02 normally
    * one finds: series_scaled.max() >= 1 but not much higher than  1.02 normally

    ## scale method='rel'
    * scales every value with a fixed factor so that
    * series_scaled.sum() == 1  or  series_scaled.sum() == -1 
    * therefore, this scaling is well suited for frequencies, 
      the numerical values are then the relative frequencies
    * series_scaled.min()  and  series_scaled.max() are numbers near 0

    ## scale method='mean'
    * also called z-score
    * one finds: series_scaled.mean() == 0 
    * one finds: series_scaled.std()  == 1 if there are more than 2 different values
    * a typical range for the scaled values is -1.7..1.7

    ## scale method='median'
    * like mean, but more robust against outliner values.
    * one finds: series_scaled.median() == 0 
    * a typical range for the scaled values is -1..1, 
      but the range can be much wider than with mean

    ## scale method='compare_median'
    * scales 0..1 below median and 1.. above median 
    * So you can multipy the scaled values by any factor, keeping the comparison to the median.
    * one finds: all values < median are 0..1
    * one finds: all values == median are 1
    * one finds: all values > median are > 1
    * series_scaled.median() will be very near 1.
    * unlike most other scalings, the plots do not lie on top of each other, 
      but have a different shape

    ## scale method='rank'
    * scales 0..1 (open interval)
    * scales by rank
    * distributes evenly over the interval, the original shape is destroyed. 
      The plots do not lie on top of each other.
    * if powerfactor == 1 one finds: series_scaled.median() == 0.5 
    * additional parameter powerfactor deforms the scale, see example

    ## scale method='random'
    * scales randomly
    * generates an ugly, krank scaling for testings
    
    '''
    if verbose is None:
        verbose = Config.get('VERBOSE') 
        
    if method == 'rel': 
        summe = series.sum() 
        if summe > almostzero:
            return series / summe
        elif summe < -almostzero:
            return -series / summe
        else:
            return  repeat( [0], series.shape[0] )  # only zeros
        
    elif method == 'max_abs':
        nenner = series.abs().max()
        if nenner > almostzero:
            return series  / nenner
        else:
            return  repeat( [0], series.shape[0] )  # only zeros
        
    elif method == 'min_max':
        nenner = (series.max() - series.min())
        if nenner > almostzero:
            return (series - series.min())  /  nenner
        else: 
            return  repeat( [0], series.shape[0] )  # only zeros
    
    
    elif method == 'min_max_robust':
        nenner = (series.quantile(0.99) - series.quantile(0.01)) 
        if nenner > almostzero:
            return (series - series.quantile(0.01))  /  nenner
        else:
            return  repeat( [0], series.shape[0] )  # only zeros
    
    # --------------------------------------------------------
    elif method == 'mean':
        nenner = series.std() 
        if nenner > almostzero:
            result = (series - series.mean()) / nenner
        else:
            return  repeat( [0], series.shape[0] )  # only zeros
        
        if abs(result.mean()) <=  almostzero:
            return result # alles ok
        if almostzero > 0.01  or  result.max() - result.min() < almostzero:
            #print('sinnlos', almostzero)            
            return result
        else:
            #print('next try', almostzero)
            return scale(result, method, almostzero=almostzero*10) 
        
    elif method == 'median': # ist im median 0
        nenner = (series.quantile(0.75) - series.quantile(0.25)) 
        if nenner > almostzero:
            return (series - series.median())  / nenner 
        else:
            return  repeat( [0], series.shape[0] )  # only zeros
        
    elif method == 'rank':
        if powerfactor == 1:
            rang = series.rank(method='dense')
        else:
            rang = np.power(series.rank(method='dense'), powerfactor)
        maximum = rang.max()
        result = rang / maximum
        abziehen = result.min() / 2
        return result - abziehen     
    
    elif method == 'compare_median':
        median = series.quantile(0.5)        
        result = series.copy()
        result = 1 + scale( series, method='median'  )  # erst mal für alle, ist im median 1          
        mask = (series < median)           
        result.loc[mask] = scale( series[mask], method='min_max' )   # kleiner median: 0..1 
        return result
    
    elif method == 'random':
        result = series.copy()
        
        # mult
        würfel = random.randint(1, 7)
        if würfel == 1:
            result = result * random.uniform(-1, 1)           
        elif würfel == 2:
            result = result * random.uniform(-10, 10)   
        elif würfel == 3:
            result = result * random.uniform(-100, 100)    
        elif würfel == 4:
            result = result * random.uniform(-1000, 1000) 
        elif würfel == 5:
            result = result * random.uniform(-0.1, 0.1)   
        elif würfel == 6:
            result = result * random.uniform(-0.01, 0.01) 
        elif würfel == 7:
            result = result * random.uniform(-0.001, 0.001)  
            
        if result.min() == 0  or  result.max() == 0: 
            result = result + random.uniform(-10, 10) 
            
        diff = result.max() - result.min()
        würfel = random.randint(1, 1)
        
        if diff < 0.001  and  würfel==1:
            result = result ** random.uniform( 1, 4) 
        elif diff < 0.01  and  würfel==1:
            result = result ** random.uniform( 1, 2)  
        elif diff < 0.1  and  würfel==1:
            result = result ** random.uniform( 1, 1.5)      
            
        if result.min() == 0  or  result.max() == 0: 
            result = result + random.uniform(-10, 10)             
                
        # again?
        #if np.isnan( result.min() ):
        #    return scale(series,'random')
        #if np.isnan( result.max() ):
        #    return scale(series,'random')        
        return result




# normiert eine Series auf einen Wert 0..1
# wahrscheinlich ohne die 0 und ohne die 1
# Der powerfactor verzerrt die Ergebnisse, so dass die Verteilung nicht mehr linear ist
#def normiere_rang(s, powerfactor=1):
#    if powerfactor == 1:
#        rang = s.rank(method='dense')
#    else:
#        rang = np.power(s.rank(method='dense'), powerfactor)
#    maximum = rang.max()
#    result = rang / maximum
#    abziehen = result.min() / 2
#    return result - abziehen



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



