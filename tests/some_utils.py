import pandas as pd
import numpy as np
import pandasklar.content as pak
import random


def generate_random_data_tcdt(seed=42, anz=1000):
    """
    Generiert zufällige Testdaten für change_datatype mit einem festen Seed.

    Args:
        seed (int): Der Seed für den Zufallsgenerator.

    Returns:
        pd.DataFrame: Ein DataFrame mit zufälligen Testdaten.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Generate random data
    a = pak.random_series(anz, 'int', min=-500, max=100)
    b = pak.random_series(anz, 'int', min=-127, max=127, p_dup=0)  # keine Dups erlaubt
    c = a + b + 0.0001
    a = a % 10 * 10
    v = pak.random_series(anz, 'name', p_nan=0)
    w = v.str[:1]
    s = pak.random_series(anz, 'string', p_nan=0)
    t = pak.random_series(anz, 'string', p_nan=0.1)
    m = pak.random_series(anz, 'int', min=0, max=127, p_nan=0.1)
    n = pak.random_series(anz, 'float', decimals=4, p_nan=0.2)  # * 70000
    o = pak.random_series(anz, 'choice', choice=['Bremen', 'Bremerhaven'], p_nan=0.3, p_dup=0)
    p = pak.random_series(anz, 'list', p_nan=0.1, p_dup=0.5)
    q = pak.random_series(anz, 'time', p_nan=0.1, p_dup=0.5)
    z = pak.random_series(anz, 'mix', p_nan=0.01, p_dup=0)

    df = pak.dataframe([a, b, c, v, w, s, t, m, n, o, p, q, z], verbose=False)
    df.columns = ['int_grob', 'int_fein', 'float_summe', 'first_name', 'Letter1', 'string_nonan', 'string_nan',
                  'int_nan', 'float_nan', 'City', 'List', 'time', 'Mix']
    df.float_summe = df.float_summe.astype('float')
    return df