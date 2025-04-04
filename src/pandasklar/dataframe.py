
# 0.5

import warnings, copy

import pandas as pd
import polars as pl
import numpy as np
import bpyth as bpy

#from functools import partial
from collections import Counter, defaultdict

from .config import Config



# ==================================================================================================
# Create
# ==================================================================================================
#
#
def dataframe(inp, verbose=None, framework=None):
    """
    Converts various multidimensional objects into DataFrames (Pandas or Polars).

    This function intelligently transforms a wide range of input data structures into either a
    Pandas DataFrame or a Polars DataFrame, based on the specified `framework`. It automatically
    infers the intended structure (row-wise or column-wise) and handles various data types.

    **Input Handling:**

    The function can handle the following input types:

    - **Scalars:** Single values (int, float, str, bool) are converted into a 1x1 DataFrame.
    - **Lists:**
        - One-dimensional lists are interpreted as a single row.
        - Multidimensional lists (list of lists, list of tuples) are interpreted row-wise,
          where each inner list/tuple represents a row.
        - A list of Series is interpreted column-wise.
    - **Tuples:**
        - One-dimensional tuples are interpreted as a single row.
        - Multidimensional tuples (tuple of lists, tuple of tuples) are interpreted column-wise,
          where each inner list/tuple represents a column.
        - A tuple of Series is interpreted column-wise.
    - **Dictionaries:**
        - One-dimensional dictionaries are interpreted as a single row.
        - Multidimensional dictionaries are interpreted column-wise, where keys become column names.
        - Lists of dictionaries are interpreted row-wise.
    - **Pandas Series** (pd.Series): Interpreted as a single column.
    - **Polars Series** (pl.Series): Interpreted as a single column.
    - **NumPy ndarrays** (np.ndarray): Interpreted as a single column.
    - **Lists/Tuples of Series/ndarrays:** Interpreted column-wise, where each Series/ndarray represents a column.

    **Key Features:**
    - **Automatic Structure Inference:** The function automatically determines whether the input data
      should be interpreted row-wise or column-wise, based on the input type.
    - **Column Name Handling:** Sensible column names are automatically assigned if not provided.
      Duplicate column names or numeric column names are replaced with letters (A, B, C, ...).
    - **Framework Flexibility:** Supports both Pandas and Polars DataFrames.
    - **Series/ndarray Handling:** Correctly handles Pandas Series, Polars Series, and NumPy ndarrays as columns.
    - **Robustness:** Handles various edge cases, including empty inputs and mixed data types.

    **Args:**
    - `inp`: The input object to be converted into a DataFrame. Can be any of the types listed above.
    - `verbose` (bool, optional): If True, prints detailed information about the input object and the
            conversion process. Defaults to the value of `Config.get('VERBOSE')`.
    - `framework` (str, optional): Specifies the desired DataFrame framework. Must be either 'pandas'
            or 'polars'. Defaults to the value of `Config.get('FRAMEWORK')`.

    **Returns:**
    - pandas.DataFrame or polars.DataFrame: The resulting DataFrame, based on the specified `framework`.
    """

    # Vorbereitung -----------------------------------------------------------------------------------------------------

    if verbose is None:
        verbose = Config.get('VERBOSE')

    if framework is None:
        framework = Config.get('FRAMEWORK')

    if framework not in ['pandas', 'polars']:
        raise ValueError(f"Framework '{framework}' is not supported. Only 'pandas' and 'polars' are allowed.")

    result = None

    # already DataFrame ------------------------------------------------------------------------------------------------

    if isinstance(inp, pd.DataFrame):
        if framework == 'pandas':
            return inp  # Kein Konvertierung nötig
        elif framework == 'polars':
            return pl.from_pandas(inp)  # Konvertiere zu Polars
    elif isinstance(inp, pl.DataFrame):
        if framework == 'polars':
            return inp  # Keine Konvertierung nötig
        elif framework == 'pandas':
            return inp.to_pandas()  # Konvertiere zu Pandas


    # Analyse ----------------------------------------------------------------------------------------------------------

    inp_rtype = bpy.rtype(inp)
    try:
        inp_shape = bpy.shape(inp)
    except:
        inp_shape = (-77, -77)
    if verbose:
        print('dataframe: Input rtype=' + str(inp_rtype), 'shape=' + str(inp_shape))

    # Interne Methoden -------------------------------------------------------------------------------------------------

    # dups vermeiden
    def remove_duplicate_names(iter_of_series):
        #return iter_of_series
        name_counts = Counter()
        result0 = []
        for s in iter_of_series:
            if s.name is None or s.name == '':
                result0.append(s)
            else:
                name_counts[s.name] += 1
                new_name = f"{s.name}_{name_counts[s.name] - 1}" if name_counts[s.name] > 1 else s.name
                result0.append(s.rename(new_name))
        return result0


    # Spaltennamen
    def cols_benennen_pandas(df: pd.DataFrame) -> pd.DataFrame:
        if df.shape[1] > 52:  # zu breit, kann man nicht umbenennen
            return df
        # gibt es Duplikate in den Spaltennamen?                       oder sind die Spalten rein numerisch?
        if (len(list(df.columns)) != len(set(df.columns))) or isinstance(df.columns, pd.RangeIndex):
            df.columns = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')[0:df.shape[1]]
        return df

    def cols_benennen_polars(df: pl.DataFrame) -> pl.DataFrame:
        if df.width > 52:  # zu viele Spalten, nicht umbenennen
            return df

        col_names = df.columns
        # generische Spaltennamen?
        if all(name.startswith("column_") and name[7:].isdigit() for name in df.columns):
            new_cols = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')[:df.width]
            df.columns = new_cols
        return df

    # no shape ---------------------------------------------------------------------------------------------------------

    if bpy.has_no_content(inp):
        # if verbose:
        #     print('dataframe: no content')
        if framework == 'pandas':
            return pd.DataFrame()
        elif framework == 'polars':
            return pl.DataFrame()

    if not bpy.has_shape(inp):
        # if verbose:
        #     print('dataframe: no shape')
        if framework == 'pandas':
            result = cols_benennen_pandas(pd.DataFrame([inp]))
        elif framework == 'polars':
            result = cols_benennen_polars(pl.DataFrame([inp]))

    # beenden -----------------------------------------------------------------------------------------------
    if result is not None:
        return result


    # 1 Dimensionen: ---------------------------------------------------------------------------------------------------

    if len(inp_shape) == 1:
        # inp ist eine einzelne Serie -> 1 Spalte
        if isinstance(inp, (pd.Series, pl.Series, np.ndarray)):
            # if verbose:
            #     print('dataframe: Series')
            if framework == 'pandas':
                if not isinstance(inp, np.ndarray)  and inp.name:
                    result = pd.DataFrame(list(inp))
                    result.columns = [inp.name]
                else:
                    result = cols_benennen_pandas(pd.DataFrame(list(inp)))

            elif framework == 'polars':
                if not isinstance(inp, np.ndarray)  and inp.name:
                    result = pl.DataFrame(list(inp))
                    result.columns = [str(inp.name)]
                else:
                    result = cols_benennen_polars(pl.DataFrame(list(inp)))


        # inp 1dim dict
        elif isinstance(inp, dict):
            # if verbose:
            #     print('dataframe: 1dim-dict')
            if framework == 'pandas':
                result = cols_benennen_pandas(pd.DataFrame([inp]))
            elif framework == 'polars':
                result = cols_benennen_polars(pl.DataFrame([inp], orient='row'))

        # 1dim alles andere außer dict
        elif not isinstance(inp, dict):
            # if verbose:
            #     print('dataframe: 1dim-anything')
            if framework == 'pandas':
                result = cols_benennen_pandas(pd.DataFrame([inp]))
            elif framework == 'polars':
                result = cols_benennen_polars(pl.DataFrame([inp], orient='row'))

    # ------------------------------------------------------------------------------------------------------------------

    if result is None:

        if isinstance(inp, dict):
            if verbose:
                print('dataframe: dict')
            if framework == 'pandas':
                result = cols_benennen_pandas(pd.DataFrame(inp))
            elif framework == 'polars':
                result = pl.from_pandas(cols_benennen_pandas(pd.DataFrame(inp)))


        elif ( inp_rtype[:2] == ('list', 'Series') or
               inp_rtype[:2] == ('tuple', 'Series') or
               inp_rtype[:2] == ('list', 'ndarray') or
               inp_rtype[:2] == ('tuple', 'ndarray') ):

            if verbose:
                print('dataframe: list or tuple of ndarray or Series')
            if framework == 'pandas':
                inp = [s.to_pandas() if isinstance(s, pl.Series) else s for s in inp]
                inp = remove_duplicate_names(inp)
                result = cols_benennen_pandas(pd.concat(inp, axis=1))
            elif framework == 'polars':
                # ggf. Spaltennamen vergeben
                inp = [s.rename(f"column_{i}") if isinstance(s, (pl.Series, pd.Series)) and s.name is None else s for i, s in enumerate(inp)]
                inp = [s.rename(str(s.name))   if isinstance(s, (pl.Series, pd.Series)) and not isinstance(s.name, str) else s for s in inp]
                inp = remove_duplicate_names(inp)
                result = cols_benennen_polars(pl.DataFrame(inp))


        elif ( inp_rtype[:2] == ('list', 'list') or
               inp_rtype[:2] == ('list', 'tuple') ):

            # if verbose:
            #     print('dataframe: list of lists or tuples')
            if framework == 'pandas':
                result = cols_benennen_pandas(pd.DataFrame(inp))
            elif framework == 'polars':
                result = cols_benennen_polars(pl.DataFrame(inp, orient='row'))


        elif ( inp_rtype[:2] == ('tuple', 'list') or
               inp_rtype[:2] == ('tuple', 'tuple') ):
            # if verbose:
            #     print('dataframe: tuple of lists or tuples')
            if framework == 'pandas':
                result = cols_benennen_pandas(pd.DataFrame(zip(*inp)))
            elif framework == 'polars':
                result = cols_benennen_polars(pl.DataFrame(inp, orient='col'))

        else:
            # if verbose:
            #     print('dataframe: REST')
            if framework == 'pandas':
                result = cols_benennen_pandas(pd.DataFrame(list(inp)))
                result = (result)
            elif framework == 'polars':
                inp = [{str(k): v for k, v in d.items()} if isinstance(d, dict) else d for d in inp]
                result = cols_benennen_polars(pl.DataFrame(inp))

    return result



