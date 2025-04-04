
import pytest
import pandas as pd
import numpy as np
import bpyth as bpy
from pandasklar.analyse import col_names, nnan, any_nan, nan_rows
from pandasklar.analyse import val_most, nunique, ntypes, analyse_freqs, sort_cols_by_nunique
from pandasklar.analyse import same_but_different
from pandasklar.content import people, random_numbers
from pandasklar.dataframe import dataframe


#@pytest.mark.jetzt
class TestAnalyse:

    ######################################################################
    # col_names
    ######################################################################

    def test_col_names_only(self):
        # Teste, ob col_names mit only korrekt funktioniert
        df = people(size=10, seed=84)
        result = set(col_names(df, only='str'))
        assert result ==  {'first_name', 'birthplace', 'secret'}

    def test_col_names_without(self):
        # Teste, ob col_names mit without korrekt funktioniert
        df = people(size=10, seed=84)
        result = set(col_names(df, without='str'))
        assert result == {'age', 'age_class', 'features', 'history', 'postal_code'}

    def test_col_names_query(self):
        # Teste, ob col_names mit query korrekt funktioniert
        df = people(size=10, seed=84)
        result = set(col_names(df, only='int', query='nnan > 0'))
        assert result == {'postal_code'}

    def test_col_names_as_list(self):
        # Teste, ob col_names mit as_list korrekt funktioniert
        df = people(size=10, seed=84)
        result = col_names(df, as_list=True)
        assert isinstance(result, list)

    def test_col_names_not_as_list(self):
        # Teste, ob col_names mit as_list=False korrekt funktioniert
        df = people(size=10, seed=84)
        result = col_names(df, as_list=False)
        assert isinstance(result, pd.DataFrame)

    def test_col_names_sort(self):
        # Teste, ob col_names mit sort korrekt funktioniert
        df = people(size=10, seed=84)
        result = col_names(df, sort=True)
        assert isinstance(result, list)


    ######################################################################
    # sort_cols_by_nunique
    ######################################################################

    def test_sort_cols_by_nunique_small(self):
        # Teste, ob sort_cols_by_nunique mit einem kleinen DataFrame korrekt funktioniert
        df = pd.DataFrame({'a': [1, 1, 1, 2, 2, 3],
                           'b': [1, 1, 1, 1, 1, 1],
                           'c': [1, 2, 3, 4, 5, 6]})
        result = sort_cols_by_nunique(df)
        assert result.columns.tolist() == ['c', 'a', 'b']

    def test_sort_cols_by_nunique_large(self):
        # Teste, ob sort_cols_by_nunique mit einem großen DataFrame korrekt funktioniert
        df = pd.DataFrame({'a': np.random.randint(0, 100, size=10000),
                           'b': np.random.randint(0, 10, size=10000),
                           'c': np.random.randint(0, 1000, size=10000)})
        result = sort_cols_by_nunique(df)
        assert result.columns.tolist() == ['c', 'a', 'b']

    def test_sort_cols_by_nunique_empty(self):
        # Teste, ob sort_cols_by_nunique mit einem leeren DataFrame korrekt funktioniert
        df = pd.DataFrame()
        result = sort_cols_by_nunique(df)
        assert result.shape == (0, 0)


    ######################################################################
    # nnan
    ######################################################################

    def test_nnan_series(self):
        # Teste, ob nnan mit einer Series korrekt funktioniert
        s = pd.Series([1, 2, np.nan, 4])
        result = nnan(s)
        assert result == 1

    def test_nnan_dataframe(self):
        # Teste, ob nnan mit einem DataFrame korrekt funktioniert
        df = pd.DataFrame({'a': [1, 2, np.nan, 4], 'b': [np.nan, 2, 3, 4]})
        result = nnan(df)
        assert result.to_dict() == {'a': 1, 'b': 1}

    def test_nnan_dataframe_all(self):
        # Teste, ob nnan mit einem DataFrame und all=True korrekt funktioniert
        df = pd.DataFrame({'a': [1, 2, np.nan, 4], 'b': [np.nan, 2, 3, 4]})
        result = nnan(df, all=True)
        assert result.to_dict() == {'a': 1, 'b': 1}

    ######################################################################
    # any_nan
    ######################################################################

    def test_any_nan_series(self):
        # Teste, ob any_nan mit einer Series korrekt funktioniert
        s = pd.Series([1, 2, np.nan, 4])
        result = any_nan(s)
        assert result == True

    def test_any_nan_dataframe(self):
        # Teste, ob any_nan mit einem DataFrame korrekt funktioniert
        df = pd.DataFrame({'a': [1, 2, np.nan, 4], 'b': [np.nan, 2, 3, 4]})
        result = any_nan(df)
        assert result == True

    def test_any_nan_dataframe_without(self):
        # Teste, ob any_nan mit einem DataFrame und without korrekt funktioniert
        df = pd.DataFrame({'a': [1, 2, np.nan, 4], 'b': [1, 2, 3, 4]})
        result = any_nan(df, without='a')
        assert result == False

    def test_any_nan_dataframe_without_list(self):
        # Teste, ob any_nan mit einem DataFrame und without als Liste korrekt funktioniert
        df = pd.DataFrame({'a': [1, 2, np.nan, 4], 'b': [1, 2, 3, 4]})
        result = any_nan(df, without=['a'])
        assert result == False

    ######################################################################
    # nan_rows
    ######################################################################

    def test_nan_rows_dataframe(self):
        # Teste, ob nan_rows mit einem DataFrame korrekt funktioniert
        df = pd.DataFrame({'a': [1, 2, np.nan, 4], 'b': [np.nan, 2, 3, 4]})
        result = nan_rows(df)
        assert result.shape == (2, 2)

    def test_nan_rows_dataframe_col(self):
        # Teste, ob nan_rows mit einem DataFrame und col korrekt funktioniert
        df = pd.DataFrame({'a': [1, 2, np.nan, 4], 'b': [np.nan, 2, 3, 4]})
        result = nan_rows(df, col='a')
        assert result.shape == (1, 2)

    def test_nan_rows_dataframe_col_0(self):
        # Teste, ob nan_rows mit einem DataFrame und col=0 korrekt funktioniert
        df = pd.DataFrame({'a': [1, 2, np.nan, 4], 'b': [np.nan, 2, 3, 4]})
        result = nan_rows(df, col=0)
        assert result.shape == (1, 2)


    ######################################################################
    # val_most
    ######################################################################

    def test_val_most_series(self):
        # Teste, ob val_most mit einer Series korrekt funktioniert
        s = pd.Series([1, 2, 2, 3, 3, 3])
        result = val_most(s)
        assert result == 3

    def test_val_most_series_nan(self):
        # Teste, ob val_most mit einer Series und NaN korrekt funktioniert
        s = pd.Series([1, 2, 2, np.nan, 3, 3, 3])
        result = val_most(s)
        assert result == 3

    ######################################################################
    # nunique
    ######################################################################

    def test_nunique_series(self):
        # Teste, ob nunique mit einer Series korrekt funktioniert
        s = pd.Series([1, 2, 2, 3, 3, 3])
        result = nunique(s)
        assert result == 3

    def test_nunique_series_list(self):
        # Teste, ob nunique mit einer Series und einer Liste korrekt funktioniert
        s = pd.Series([[1, 2], [1, 2], [3, 4]])
        result = nunique(s)
        assert result == 2

    ######################################################################
    # ntypes
    ######################################################################

    def test_ntypes_series(self):
        # Teste, ob ntypes mit einer Series korrekt funktioniert
        s = pd.Series([1, 2.0, 'a', 'b'])
        result = ntypes(s)
        assert result == 3

    def test_ntypes_series_nan(self):
        # Teste, ob ntypes mit einer Series und NaN korrekt funktioniert
        s = pd.Series([1, 2.0, np.nan, 'a', 'b'])
        result = ntypes(s)
        assert result == 3


