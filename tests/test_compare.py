import pytest
import pandas as pd
from pandasklar.compare import fillna, check_equal, compare_col_dtype
from pandasklar.content import people, random_numbers

#@pytest.mark.jetzt # pytest -m jetzt -x
class TestCompare:

    ######################################################################
    # fillna
    ######################################################################

    def test_fillna_zero_int(self):
        # Teste, ob fillna mit Methode 'zero' und int korrekt funktioniert
        s = pd.Series([1, 2, None, 4], dtype='Int64')
        result = fillna(s, method='zero')
        assert result.tolist() == [1, 2, 0, 4]
        assert result.dtype == 'Int64'


    def test_fillna_zero_float(self):
        # Teste, ob fillna mit Methode 'zero' und float korrekt funktioniert
        s = pd.Series([1.1, 2.2, None, 4.4])
        result = fillna(s, method='zero')
        assert result.tolist() == [1.1, 2.2, 0.0, 4.4]
        assert result.dtype == 'float64'


    def test_fillna_zero_string(self):
        # Teste, ob fillna mit Methode 'zero' und string korrekt funktioniert
        s = pd.Series(['a', 'b', None, 'd'], dtype='string')
        result = fillna(s, method='zero')
        assert result.tolist() == ['a', 'b', '', 'd']
        assert result.dtype == 'string'


    def test_fillna_special_int(self):
        # Teste, ob fillna mit Methode 'special' und int korrekt funktioniert
        s = pd.Series([1, 2, None, 4], dtype='Int64')
        result = fillna(s, method='special')
        assert result.tolist() == [1, 2, -777, 4]
        assert result.dtype == 'Int64'


    def test_fillna_special_float(self):
        # Teste, ob fillna mit Methode 'special' und float korrekt funktioniert
        s = pd.Series([1.1, 2.2, None, 4.4])
        result = fillna(s, method='special')
        assert result.tolist() == [1.1, 2.2, -77.77, 4.4]
        assert result.dtype == 'float64'


    def test_fillna_special_string(self):
        # Teste, ob fillna mit Methode 'special' und string korrekt funktioniert
        s = pd.Series(['a', 'b', None, 'd'], dtype='string')
        result = fillna(s, method='special')
        assert result.tolist() == ['a', 'b', '∅', 'd']
        assert result.dtype == 'string'


    def test_fillna_invalid_method(self):
        # Teste, ob fillna mit ungültiger Methode eine Exception wirft
        s = pd.Series([1, 2, None, 4], dtype='Int64')
        with pytest.raises(ValueError):
            fillna(s, method='invalid')


    ######################################################################
    # check_equal
    ######################################################################


    def test_check_equal_identical(self):
        # Teste, ob check_equal mit identischen DataFrames korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84)
        assert check_equal(df1, df2) == True

    def test_check_equal_different_content(self):
        # Teste, ob check_equal mit unterschiedlichem Inhalt korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84)
        df2.loc[0, 'first_name'] = 'Test'
        assert check_equal(df1, df2) == False

    def test_check_equal_different_columns(self):
        # Teste, ob check_equal mit unterschiedlichen Spalten korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84).drop('first_name', axis=1)
        assert check_equal(df1, df2) == False

    def test_check_equal_different_shape(self):
        # Teste, ob check_equal mit unterschiedlicher Form korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=5, seed=84)
        assert check_equal(df1, df2) == False

    def test_check_equal_different_index(self):
        # Teste, ob check_equal mit unterschiedlichen Indexen korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84).set_index('first_name')
        assert check_equal(df1, df2) == False

    def test_check_equal_different_sort(self):
        # Teste, ob check_equal mit unterschiedlicher Sortierung korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84).sort_values('first_name', ascending=False)
        assert check_equal(df1, df2) == True



    ######################################################################
    # compare_col_dtype
    ######################################################################

    def test_compare_col_dtype_identical(self):
        # Teste, ob compare_col_dtype mit identischen DataFrames korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84)
        assert compare_col_dtype(df1, df2) == []

    def test_compare_col_dtype_different(self):
        # Teste, ob compare_col_dtype mit unterschiedlichen Datentypen korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84).astype({'age': 'float64'})
        assert compare_col_dtype(df1, df2) == ['age']

    def test_compare_col_dtype_different_columns(self):
        # Teste, ob compare_col_dtype mit unterschiedlichen Spalten korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84).drop('first_name', axis=1)
        assert compare_col_dtype(df1, df2) == ['first_name']



















