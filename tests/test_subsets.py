import pytest
import pandas as pd
import numpy as np
import polars as pl
from pandasklar.subsets import specials, sample
from pandasklar.content import people, random_numbers
from bpyth import rtype

#@pytest.mark.jetzt # pytest -m jetzt -x
class TestSubsetsSpecials:

    ###################################################################
    # specials leer
    ###################################################################

    def test_specials_empty_dataframe(self):
        # Teste, ob specials mit einem leeren DataFrame korrekt funktioniert
        df = pd.DataFrame()
        result = specials(df)
        assert result.empty

    def test_specials_dataframe_with_only_nan(self):
        # Teste, ob specials mit einem DataFrame, der nur NaN-Werte enthält, korrekt funktioniert
        df = pd.DataFrame({'col1': [np.nan, np.nan], 'col2': [np.nan, np.nan]})
        result = specials(df, indicator='note')
        assert result.shape == (2, 3)

    ###################################################################
    # specials Generell
    ###################################################################

    def test_specials_different_datatypes(self):
        # Teste, ob specials mit einem DataFrame, der unterschiedliche Datentypen enthält, korrekt funktioniert
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c'], 'col3': [1.1, 2.2, 3.3], 'col4': [True, False, True]})
        result = specials(df, indicator='note')
        assert result.shape[1] == df.shape[1] + 1
        assert result.shape[0] == 3

    def test_specials_with_duplicates(self):
        # Teste, ob specials mit einem DataFrame, der Duplikate enthält, korrekt funktioniert
        df = pd.DataFrame({'col1': [1, 1, 2, 2, 3], 'col2': ['a', 'a', 'b', 'b', 'c']})
        result = specials(df, indicator='note')
        assert result.shape[1] == df.shape[1] + 1
        assert result.shape[0] == 2


    ###################################################################
    # specials Einzelne Features
    ###################################################################

    def test_specials_head(self):
        # Teste, ob specials mit find=['head'] korrekt funktioniert
        df = people(size=10, seed=84)
        result = specials(df, find=['head'], indicator='note')
        assert result.shape[1] == df.shape[1] + 1
        assert result.shape[0] == 1
        assert result.index[0] == 0
        assert 'note' in result.columns

    def test_specials_tail(self):
        # Teste, ob specials mit find=['tail'] korrekt funktioniert
        df = people(size=10, seed=84)
        result = specials(df, find=['tail'], indicator='note')
        assert result.shape[1] == df.shape[1] + 1
        assert result.shape[0] == 1
        assert result.index[0] == 9

    def test_specials_first(self):
        # Teste, ob specials mit find=['first'] korrekt funktioniert
        df = people(size=10, seed=84)
        result = specials(df, find=['first'], indicator='note')
        assert result.shape[1] == df.shape[1] + 1
        assert result.shape[0] == 1
        assert result.index[0] == 0

    def test_specials_first(self):
        # Teste, ob specials mit find=['first'] korrekt funktioniert
        df = people(size=10, seed=3)
        result = specials(df, find=['first'], indicator='note')
        assert result.shape[1] == df.shape[1] + 1
        assert result.shape[0] == 2 # das liegt daran, dass birthplace anfangs leer ist
        assert result.index[0] == 0

    def test_specials_last(self):
        # Teste, ob specials mit find=['last'] korrekt funktioniert
        df = people(size=10, seed=3)
        result = specials(df, find=['last'], indicator='note')
        assert result.shape[1] == df.shape[1] + 1
        assert result.shape[0] == 1
        assert result.index[0] == 9

    def test_specials_last(self):
        # Teste, ob specials mit find=['last'] korrekt funktioniert
        df = people(size=10, seed=42)
        result = specials(df, find=['last'], indicator='note')
        assert result.shape[1] == df.shape[1] + 1
        assert result.shape[0] == 2 # das liegt daran, dass postal code anfangs leer ist


    def test_specials_min(self):
        # Teste, ob specials mit find=['min'] korrekt funktioniert
        df = people(size=10, seed=17)
        result = specials(df, find=['min'], indicator='note')
        assert result.shape[1] == df.shape[1] + 1
        assert result.shape[0] == 4
        assert result['age'].iloc[0] == df['age'].min()

    def test_specials_max(self):
        # Teste, ob specials mit find=['max'] korrekt funktioniert
        df = people(size=10, seed=1)
        result = specials(df, find=['max'], indicator='note')
        assert result.shape[1] == df.shape[1] + 1
        assert result.shape[0] == 4
        assert result['age'].iloc[0] == df['age'].max()

    def test_specials_most(self):
        # Teste, ob specials mit find=['most'] korrekt funktioniert
        df = people(size=10, seed=1)
        result = specials(df, find=['most'], indicator='note')
        assert result.shape[1] == df.shape[1] + 1
        assert result.shape[0] == 2
        assert result['first_name'].iloc[0] == df['first_name'].mode()[0]

    def test_specials_nan(self):
        # Teste, ob specials mit find=['nan'] korrekt funktioniert
        df = people(size=10, seed=13)
        df.loc[3, 'first_name'] = np.nan
        result = specials(df, find=['nan'], indicator='note')
        assert result.shape[1] == df.shape[1] + 1
        assert result.shape[0] == 3
        assert pd.isna(result['first_name'].iloc[0])


    ###################################################################
    # specials Div
    ###################################################################

    def test_specials_all(self):
        # Teste, ob specials mit allen find-Optionen korrekt funktioniert
        df = people(size=10, seed=13)
        df.loc[3, 'first_name'] = np.nan
        result = specials(df, indicator='note')
        assert result.shape[1] == df.shape[1] + 1
        assert result.shape[0] == 7



    def test_sample_without_parameters(self):
        # Teste, ob sample ohne Parameter korrekt funktioniert und min und max enthält
        df_pandas = people(size=100, seed=84, framework='pandas')
        result_pandas = sample(df_pandas)
        assert result_pandas.shape[0] == specials(df_pandas).shape[0]
        for col in ['age', 'postal_code']:
            assert df_pandas[col].min() in result_pandas[col].values
            assert df_pandas[col].max() in result_pandas[col].values
        assert isinstance(result_pandas, pd.DataFrame)
        df_polars = people(size=100, seed=84, framework='polars')
        result_polars = sample(df_polars)
        assert result_polars.shape[0] == specials(df_polars).shape[0]
        for col in ['age', 'postal_code']:
            assert df_polars[col].min() in result_polars[col].to_list()
            assert df_polars[col].max() in result_polars[col].to_list()
        assert isinstance(result_polars, pl.DataFrame)


###################################################################
# sample
###################################################################

#@pytest.mark.jetzt # pytest -m jetzt -x
class TestSubsetsSample:
    ###################################################################
    # sample
    ###################################################################

    def test_sample_empty_dataframe(self):
        # Teste, ob sample mit einem leeren DataFrame korrekt funktioniert
        df = pd.DataFrame()
        result_pandas = sample(df)
        assert result_pandas.empty
        assert isinstance(result_pandas, pd.DataFrame)
        result_polars = sample(pl.DataFrame())
        assert result_polars.shape == (0,0)
        assert isinstance(result_polars, pl.DataFrame)

    def test_sample_only_nans(self):
        df = pd.DataFrame({'col1': [np.nan, np.nan, np.nan, np.nan,], 'col2': [np.nan, np.nan, np.nan, np.nan]})
        result_pandas = sample(df)
        assert result_pandas.shape == (2, 2)
        assert isinstance(result_pandas, pd.DataFrame)
        result_polars = sample(pl.DataFrame({'col1': [np.nan, np.nan, np.nan, np.nan,], 'col2': [np.nan, np.nan, np.nan, np.nan]}))
        assert result_polars.shape == (2, 2)
        assert isinstance(result_polars, pl.DataFrame)

    def test_sample_zero_row(self):
        # Teste, ob sample mit einem DataFrame, der nur eine Zeile enthält, korrekt funktioniert
        df_pandas = people(size=0, seed=84, framework='pandas')
        result_pandas = sample(df_pandas, size=10)
        assert result_pandas.shape[0] == 0
        assert isinstance(result_pandas, pd.DataFrame)
        df_polars = people(size=0, seed=84, framework='polars')
        result_polars = sample(df_polars, size=10)
        assert result_polars.shape[0] == 0
        assert isinstance(result_polars, pl.DataFrame)

    def test_sample_none_size(self):
        # Teste, ob sample mit size=None korrekt funktioniert
        df_pandas = people(size=50, seed=84, framework='pandas')
        result_pandas = sample(df_pandas, size=None)
        assert result_pandas.shape[0] == specials(df_pandas).shape[0]
        assert isinstance(result_pandas, pd.DataFrame)
        df_polars = people(size=50, seed=84, framework='polars')
        result_polars = sample(df_polars, size=None)
        assert result_polars.shape[0] == specials(df_polars).shape[0]
        assert isinstance(result_polars, pl.DataFrame)

    def test_sample_zero_size(self):
        # Teste, ob sample mit size=0 korrekt funktioniert
        df_pandas = people(size=10, seed=84, framework='pandas')
        result_pandas = sample(df_pandas, size=0)
        assert result_pandas.shape[0] == 0
        assert isinstance(result_pandas, pd.DataFrame)
        df_polars = people(size=10, seed=84, framework='polars')
        result_polars = sample(df_polars, size=0)
        assert result_polars.shape[0] == 0
        assert isinstance(result_polars, pl.DataFrame)

    def test_sample_smaller_than_size(self):
        # Teste, ob sample korrekt funktioniert, wenn size größer ist als die Anzahl der Zeilen
        df_pandas = people(size=5, seed=84, framework='pandas')
        result_pandas = sample(df_pandas, size=10)
        assert result_pandas.shape[0] == 5
        assert isinstance(result_pandas, pd.DataFrame)
        df_polars = people(size=5, seed=84, framework='polars')
        result_polars = sample(df_polars, size=10)
        assert result_polars.shape[0] == 5
        assert isinstance(result_polars, pl.DataFrame)

    def test_sample_correct_size(self):
        # Teste, ob sample die korrekte Anzahl von Zeilen zurückgibt
        df_pandas = people(size=100, seed=84, framework='pandas')
        result_pandas = sample(df_pandas, size=20)
        assert result_pandas.shape[0] == 20
        assert isinstance(result_pandas, pd.DataFrame)
        df_polars = people(size=100, seed=84, framework='polars')
        result_polars = sample(df_polars, size=20)
        assert result_polars.shape[0] == 20
        assert isinstance(result_polars, pl.DataFrame)

    def test_sample_contains_head_and_tail(self):
        # Teste, ob sample die erste und letzte Zeile enthält
        df_pandas = people(size=100, seed=84, framework='pandas')
        result_pandas = sample(df_pandas, size=20)
        assert 0 in result_pandas.index
        assert 99 in result_pandas.index
        assert isinstance(result_pandas, pd.DataFrame)
        df_polars = people(size=100, seed=84, framework='polars')
        result_polars = sample(df_polars, size=20)
        assert isinstance(result_polars, pl.DataFrame)

    def test_sample_one_row(self):
        # Teste, ob sample mit einem DataFrame, der nur eine Zeile enthält, korrekt funktioniert
        df_pandas = people(size=1, seed=84, framework='pandas')
        result_pandas = sample(df_pandas, size=10)
        assert result_pandas.shape[0] == 1
        assert isinstance(result_pandas, pd.DataFrame)
        df_polars = people(size=1, seed=84, framework='polars')
        result_polars = sample(df_polars, size=10)
        assert result_polars.shape[0] == 1
        assert isinstance(result_polars, pl.DataFrame)

    def test_sample_with_duplicates(self):
        # Teste, ob sample mit einem DataFrame, der Duplikate enthält, korrekt funktioniert
        df = pd.DataFrame({'col1': [1, 1, 2, 2, 3], 'col2': ['a', 'a', 'b', 'b', 'c']})
        result_pandas = sample(df)
        assert result_pandas.shape[0] <= 5
        assert isinstance(result_pandas, pd.DataFrame)
        result_polars = sample(pl.DataFrame({'col1': [1, 1, 2, 2, 3], 'col2': ['a', 'a', 'b', 'b', 'c']}))
        assert result_polars.shape[0] <= 5
        assert isinstance(result_polars, pl.DataFrame)

    def test_sample_min_max(self):
        # Teste, ob sample ohne Parameter korrekt funktioniert und min und max enthält
        df_pandas = people(size=100, seed=84, framework='pandas')
        result_pandas = sample(df_pandas)
        for col in ['age', 'postal_code']:
            assert df_pandas[col].min() in result_pandas[col].values
            assert df_pandas[col].max() in result_pandas[col].values
        assert isinstance(result_pandas, pd.DataFrame)
        df_polars = people(size=100, seed=84, framework='polars')
        result_polars = sample(df_polars)
        for col in ['age', 'postal_code']:
            assert df_polars[col].min() in result_polars[col].to_list()
            assert df_polars[col].max() in result_polars[col].to_list()
        assert isinstance(result_polars, pl.DataFrame)