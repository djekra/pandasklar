
import pytest
import pandas as pd
import numpy as np
from pandasklar.analyse import analyse_freqs
from pandasklar.content import people, random_numbers
from pandasklar.dataframe import dataframe
from pandasklar.aggregate import agg_numbers


#@pytest.mark.jetzt
class TestAnalyseFreqs:

    ##############################################################
    # Nothing to analyze
    ##############################################################

    def test_analyse_freqs_empty_dataframe(self):
        # Teste, ob analyse_freqs mit einem leeren DataFrame korrekt funktioniert
        df = pd.DataFrame()
        result = analyse_freqs(df, 'first_name')
        assert result == 'Nothing to analyze'

    def test_analyse_freqs_empty_series(self):
        # Teste, ob analyse_freqs mit einer leeren Series korrekt funktioniert
        s = pd.Series([], dtype=object)
        result = analyse_freqs(s)
        assert result == 'Nothing to analyze'

    def test_analyse_freqs_none(self):
        # Teste, ob analyse_freqs mit None korrekt funktioniert
        result = analyse_freqs(None)
        assert result == 'Nothing to analyze'

    ##############################################################
    # Äußere Form
    ##############################################################

    def test_analyse_freqs_single_column(self):
        # Teste, ob analyse_freqs mit einer einzelnen Spalte korrekt funktioniert
        df = people(size=100, seed=84)
        result = analyse_freqs(df, 'first_name')
        assert result.shape[0] <= 100
        assert result.shape[1] == 4
        assert 'first_name' in result.columns
        assert 'first_name_count' in result.columns
        assert 'first_name_percent' in result.columns

    def test_analyse_freqs_multiple_columns(self):
        # Teste, ob analyse_freqs mit mehreren Spalten korrekt funktioniert
        df = people(size=100, seed=84)
        result = analyse_freqs(df, ['first_name', 'birthplace'])
        assert result.shape[0] <= 100
        assert result.shape[1] == 5
        assert 'first_name' in result.columns
        assert 'first_name_count' in result.columns
        assert 'first_name_percent' in result.columns
        assert 'birthplace' in result.columns
        assert 'birthplace_count' in result.columns

    def test_analyse_freqs_limits(self):
        # Teste, ob analyse_freqs mit limits korrekt funktioniert
        df = people(size=100, seed=84)
        result = analyse_freqs(df, ['first_name', 'birthplace'], limits=[5, 2])
        assert result.shape[0] <= 5
        assert result.shape[1] == 5
        assert 'first_name' in result.columns
        assert 'first_name_count' in result.columns
        assert 'first_name_percent' in result.columns
        assert 'birthplace' in result.columns
        assert 'birthplace_count' in result.columns

    def test_analyse_freqs_splits(self):
        # Teste, ob analyse_freqs mit splits korrekt funktioniert
        df = people(size=100, seed=84)
        df['secret'] = df['secret'].str.replace(' ', ',')
        result = analyse_freqs(df, ['first_name', 'secret'], splits=[None, ','])
        assert result.shape[0] <= 100
        assert result.shape[1] == 5
        assert 'first_name' in result.columns
        assert 'first_name_count' in result.columns
        assert 'first_name_percent' in result.columns
        assert 'secret' in result.columns
        assert 'secret_count' in result.columns

    def test_analyse_freqs_sort_count(self):
        # Teste, ob analyse_freqs mit sort_count korrekt funktioniert
        df = people(size=100, seed=84)
        result_sorted = analyse_freqs(df, 'first_name', sort_count=True)
        result_unsorted = analyse_freqs(df, 'first_name', sort_count=False)
        assert result_sorted['first_name_count'].iloc[0] >= result_sorted['first_name_count'].iloc[-1]
        assert result_unsorted['first_name'].iloc[0] <= result_unsorted['first_name'].iloc[-1]

    def test_analyse_freqs_different_datatypes(self):
        # Teste, ob analyse_freqs mit verschiedenen Datentypen korrekt funktioniert
        df = people(size=100, seed=84)
        df['age_float'] = df['age'].astype(float)
        df['age_bool'] = df['age'] > 25
        df['age_list'] = df['age'].apply(lambda x: [x])
        result = analyse_freqs(df, ['age', 'age_float', 'age_bool', 'age_list'])
        assert result.shape[0] <= 100
        assert result.shape[1] == 9
        assert 'age' in result.columns
        assert 'age_count' in result.columns
        assert 'age_float' in result.columns
        assert 'age_float_count' in result.columns
        assert 'age_bool' in result.columns
        assert 'age_bool_count' in result.columns
        assert 'age_list' in result.columns
        assert 'age_list_count' in result.columns

    def test_analyse_freqs_series(self):
        # Teste, ob analyse_freqs mit einer Series korrekt funktioniert
        s = people(size=100, seed=84)['first_name']
        result = analyse_freqs(s)
        assert result.shape[0] <= 100
        assert result.shape[1] == 4
        assert 'item' in result.columns
        assert 'item_count' in result.columns
        assert 'item_percent' in result.columns

    def test_analyse_freqs_dataframe(self):
        # Teste, ob analyse_freqs mit einem DataFrame korrekt funktioniert
        df = people(size=100, seed=84)
        result = analyse_freqs(df, ['first_name', 'birthplace'])
        assert result.shape[0] <= 100
        assert result.shape[1] == 5
        assert 'first_name' in result.columns
        assert 'first_name_count' in result.columns
        assert 'first_name_percent' in result.columns
        assert 'birthplace' in result.columns
        assert 'birthplace_count' in result.columns

    ##############################################################
    # Inhaltlich
    ##############################################################

    def test_analyse_freqs_single_column_content(self):
        # Teste die inhaltliche Korrektheit der Häufigkeitsanalyse einer einzelnen Spalte
        df = dataframe({'col1': ['a', 'a', 'a', 'b', 'b', 'c', 'd', 'd', 'd', 'd']})
        result = analyse_freqs(df, 'col1')
        assert result['col1_count'].sum() == 10
        assert np.isclose(result['col1_percent'].sum(), 100.0)
        assert result['col1_count'].iloc[0] == 4
        assert result['col1_count'].iloc[1] == 3
        assert result['col1_count'].iloc[2] == 2
        assert result['col1_count'].iloc[3] == 1
        assert result['col1'].iloc[0] == 'd'
        assert result['col1'].iloc[1] == 'a'
        assert result['col1'].iloc[2] == 'b'
        assert result['col1'].iloc[3] == 'c'


    def test_analyse_freqs_multiple_columns_content(self):

        # Teste die inhaltliche Korrektheit der Häufigkeitsanalyse mehrerer Spalten
        df = dataframe({'col1': ['a', 'a', 'a', 'b', 'b', 'c', 'd', 'd', 'd', 'd'],
                        'col2': ['x', 'x', 'y', 'y', 'z', 'z', 'x', 'y', 'z', 'z']})
        result = analyse_freqs(df, ['col1', 'col2'])
        assert result['col1_count'].sum() == 10
        assert result['col1_count'].iloc[0] == 4
        assert result['col1_count'].iloc[1] == 3
        assert result['col1_count'].iloc[2] == 2
        assert result['col1_count'].iloc[3] == 1
        assert result['col1'].iloc[0] == 'd'
        assert result['col1'].iloc[1] == 'a'
        assert result['col1'].iloc[2] == 'b'
        assert result['col1'].iloc[3] == 'c'
        assert np.isclose(result['col1_percent'].sum(), 100.0)
        assert agg_numbers(result['col2_count']) == 10




