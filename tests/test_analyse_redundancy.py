
import pytest
import pandas as pd
import numpy as np
from pandasklar.analyse import same_but_different
import pandasklar.content as pak


#@pytest.mark.jetzt # pytest -m jetzt -x
class TestAnalyseRedundancy:

    ######################################################################
    # same_but_different
    ######################################################################

    def test_same_but_different_people(self):
        # Teste, ob same_but_different mit people korrekt funktioniert
        df = pak.people(20, seed=42)
        result = same_but_different(df, same=['first_name', 'age_class'], different='birthplace')
        assert result.shape == (2, 8)
        assert result['first_name'].tolist() == ['Kathrin', 'Kathrin']
        assert result['age'].tolist() == [22, 24]
        assert result['age_class'].tolist() == [20, 20]
        assert result['birthplace'].tolist() == ['Bremen', 'Berlin']

    def test_same_but_different_no_difference(self):
        # Teste, ob same_but_different korrekt funktioniert, wenn es keine Unterschiede gibt
        df = pd.DataFrame({'a': [1, 1, 2, 2],
                           'b': [1, 1, 1, 1],
                           'c': [1, 1, 1, 1]})
        result = same_but_different(df, same=['a'], different='b')
        assert result.shape == (0, 3)

    def test_same_but_different_return_mask(self):
        # Teste, ob same_but_different mit return_mask korrekt funktioniert
        df = pd.DataFrame({'a': [1, 1, 2, 2],
                           'b': [1, 2, 1, 2],
                           'c': [1, 1, 1, 1]})
        result = same_but_different(df, same=['a'], different='b', return_mask=True)
        assert result.tolist() == [True, True, True, True]

    def test_same_but_different_different_datatypes(self):
        # Teste, ob same_but_different mit verschiedenen Datentypen korrekt funktioniert
        df = pd.DataFrame({'a': [1, 1, 2, 2],
                           'b': [1.0, 1.0, 1.0, 1.0],
                           'c': [3.0, 3.0, 4.0, 3.0]})
        result = same_but_different(df, same=['a', 'b'], different='c')
        assert result.shape == (2, 3)
        assert result['a'].tolist() == [2, 2]
        assert result['b'].tolist() == [1.0, 1.0]
        assert result['c'].tolist() == [4.0, 3.0]

    def test_same_but_different_empty(self):
        # Teste, ob same_but_different mit einem leeren DataFrame korrekt funktioniert
        df = pd.DataFrame()
        result = same_but_different(df, same=['a'], different='b')
        assert result.shape == (0, 0)

    def test_same_but_different_return_mask(self):
        # Teste, ob same_but_different mit return_mask korrekt funktioniert
        df = pd.DataFrame({'a': [1, 1, 2, 2],
                           'b': [1, 2, 1, 2],
                           'c': [1, 1, 1, 1]})
        result = same_but_different(df, same=['a'], different='b', return_mask=True)
        assert result.tolist() == [True, True, True, True]
        assert result.index.tolist() == [0,1,2,3]

    def test_same_but_different_missing_column(self):
        # Teste, ob same_but_different mit fehlenden Spalten korrekt funktioniert
        df = pd.DataFrame({'a': [1, 1, 2, 2],
                           'b': [1.0, 2.0, 1.0, 2.0],
                           'c': ['a', 'a', 'a', 'a']})
        result = same_but_different(df, same=['a', 'd'], different='b')
        assert result.shape == (0, 0)