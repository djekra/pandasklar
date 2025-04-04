import pytest
import pandas as pd
import numpy as np
import polars as pl
from pandasklar.content import random_series, decorate, people, random_numbers, dump_excel, load_excel, dataframe, random_perlin
from pandasklar.compare import check_equal, get_different_rows
from pandasklar.analyse import change_datatype, dump_pickle, load_pickle
from pandasklar.pandas import reset_index, move_cols
from bpyth import rtype
import random


def excel_data(size=100, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    a = random_series(size, 'name', p_dup=0.3, name='first_name')
    b = random_series(size, 'int', min=20, max=30) + random_series(size, 'int', min=0, max=12)
    b.name = 'age'
    c = random_series(size, 'int', min=10000, max=99999, p_nan=0, p_dup=0.3, name='postal_code')
    d = random_series(size, 'choice', choice=['Bremen', 'Berlin'], p_nan=0, p_dup=0, name='birthplace')
    e = random_series(size, 'string', len_min=5, len_max=10, p_nan=0, p_dup=0, name='secret')

    result = dataframe((a, b, c, d, e), verbose=False)
    result = change_datatype(result, verbose=False)
    return result


#@pytest.mark.jetzt # pytest -m jetzt -x
class TestRandomSeries:

    ######################################################################
    # random_series
    ######################################################################

    def test_random_series_int(self):
        # Teste, ob random_series mit Typ 'int' korrekt funktioniert
        result = random_series(10, 'int', min=0, max=100)
        assert rtype(result) == ('Series', 'int')
        assert result.shape == (10,)
        assert result.min() >= 0
        assert result.max() <= 100

    def test_random_series_float(self):
        # Teste, ob random_series mit Typ 'float' korrekt funktioniert
        result = random_series(10, 'float', decimals=2)
        assert rtype(result) == ('Series', 'float')
        assert result.shape == (10,)
        assert result.min() >= 0
        assert result.max() <= 1

    def test_random_series_ascending(self):
        # Teste, ob random_series mit Typ 'ascending' korrekt funktioniert
        result = random_series(100, 'ascending')
        assert rtype(result) == ('Series', 'float')
        assert result.shape == (100,)
        assert result.is_monotonic_increasing

    def test_random_series_descending(self):
        # Teste, ob random_series mit Typ 'descending' korrekt funktioniert
        result = random_series(100, 'descending')
       # assert rtype(result) == ('Series', 'float')
        assert result.shape == (100,)
        assert result.is_monotonic_decreasing

    def test_random_series_perlin(self):
        # Teste, ob random_series mit Typ 'perlin' korrekt funktioniert
        result = random_series(100, 'perlin', freq=[1, 2, 3])
        assert rtype(result) == ('Series', 'float')
        assert result.shape == (100,)

    def test_random_series_string(self):
        # Teste, ob random_series mit Typ 'string' korrekt funktioniert
        result = random_series(10, 'string', len_min=2, len_max=5)
        assert rtype(result) == ('Series', 'str')
        assert result.shape == (10,)
        assert result.str.len().min() >= 2
        assert result.str.len().max() <= 5

    def test_random_series_name(self):
        # Teste, ob random_series mit Typ 'name' korrekt funktioniert
        result = random_series(10, 'name')
        assert rtype(result) == ('Series', 'str')
        assert result.shape == (10,)

    def test_random_series_choice(self):
        # Teste, ob random_series mit Typ 'choice' korrekt funktioniert
        result = random_series(10, 'choice', choice=['a', 'b', 'c'])
        assert rtype(result) == ('Series', 'str')
        assert result.shape == (10,)
        assert result.isin(['a', 'b', 'c']).all()

    def test_random_series_list(self):
        # Teste, ob random_series mit Typ 'list' korrekt funktioniert
        result = random_series(10, 'list', len_min=1, len_max=3)
        assert rtype(result) == ('Series', 'list', 'str')
        assert result.shape == (10,)
        assert result.apply(len).min() >= 1
        assert result.apply(len).max() <= 3

    def test_random_series_time(self):
        # Teste, ob random_series mit Typ 'time' korrekt funktioniert
        result = random_series(10, 'time', min='2023-01-01', max='2023-12-31')
        assert rtype(result) == ('Series', 'Timestamp')
        assert result.shape == (10,)
        assert result.min() >= pd.Timestamp('2023-01-01')
        assert result.max() <= pd.Timestamp('2023-12-31')

    def test_random_series_mix(self):
        # Teste, ob random_series mit Typ 'mix' korrekt funktioniert
        result = random_series(10, 'mix')
        assert rtype(result)[0] == 'Series'
        assert result.shape == (10,)



######################################################################
# decorate
######################################################################

#@pytest.mark.jetzt  # pytest -m jetzt -x
class TestDecorate:

    def test_decorate_with_nan(self):
        # Teste, ob decorate mit np.nan korrekt funktioniert
        result = pd.Series([1, 2, 3]).apply(decorate, p=1.0)
        assert result.isnull().all()

    def test_decorate_with_string(self):
        # Teste, ob decorate mit einem String korrekt funktioniert
        result = pd.Series([1, 2, 3]).apply(decorate, p=1.0, special='a')
        assert (result == 'a').all()

    def test_decorate_with_list(self):
        # Teste, ob decorate mit einer Liste korrekt funktioniert
        result = pd.Series([1, 2, 3]).apply(decorate, p=1.0, special=[None])
        assert (result.apply(lambda x: x == [None]).all()) or (result.isnull().all())

    def test_decorate_with_p_zero(self):
        # Teste, ob decorate mit p=0 korrekt funktioniert
        result = pd.Series([1, 2, 3]).apply(decorate, p=0.0)
        assert (result == pd.Series([1, 2, 3])).all()

    def test_decorate_with_p_half(self):
        # Teste, ob decorate mit p=0.5 korrekt funktioniert
        result = pd.Series([1, 2, 3]).apply(decorate, p=0.5)
        assert result.shape == (3,)




######################################################################
# people
######################################################################

# @pytest.mark.jetzt  # pytest -m jetzt -x
class TestPeople:

    def test_people_default_size(self):
        # Teste, ob people mit der Standardgröße korrekt funktioniert
        result = people()
        assert rtype(result) == ('DataFrame', 'Series', 'str')
        assert result.shape == (100, 8)
        assert result.index.name is None
        assert result.columns.tolist() == ['first_name', 'age', 'age_class', 'postal_code', 'birthplace', 'secret',
                                           'features', 'history']

    def test_people_custom_size(self):
        # Teste, ob people mit einer benutzerdefinierten Größe korrekt funktioniert
        result = people(size=50)
        assert rtype(result) == ('DataFrame', 'Series', 'str')
        assert result.shape == (50, 8)
        assert result.index.name is None
        assert result.columns.tolist() == ['first_name', 'age', 'age_class', 'postal_code', 'birthplace', 'secret',
                                           'features', 'history']

    def test_people_size_zero(self):
        # Teste, ob people mit size=0 korrekt funktioniert
        result = people(size=0)
        assert result.shape == (0, 8)
        assert result.index.name is None
        assert result.columns.tolist() == ['first_name', 'age', 'age_class', 'postal_code', 'birthplace', 'secret',
                                           'features', 'history']

    def test_people_size_one(self):
        # Teste, ob people mit size=1 korrekt funktioniert
        result = people(size=1)
        assert rtype(result) == ('DataFrame', 'Series', 'str')
        assert result.shape == (1, 8)
        assert result.index.name is None
        assert result.columns.tolist() == ['first_name', 'age', 'age_class', 'postal_code', 'birthplace', 'secret',
                                           'features', 'history']

    def test_people_size_two(self):
        # Teste, ob people mit size=2 korrekt funktioniert
        result = people(size=2)
        assert rtype(result) == ('DataFrame', 'Series', 'str')
        assert result.shape == (2, 8)
        assert result.index.name is None
        assert result.columns.tolist() == ['first_name', 'age', 'age_class', 'postal_code', 'birthplace', 'secret',
                                           'features', 'history']

    def test_people_column_types(self):
        # Teste, ob people die korrekten Spaltentypen hat
        result = people()
        assert rtype(result['first_name']) == ('Series', 'str')
        assert rtype(result['age']) == ('Series', 'int')
        assert rtype(result['age_class']) == ('Series', 'int')
        assert rtype(result['postal_code']) == ('Series', 'int')
        assert rtype(result['secret']) == ('Series', 'str')
        #assert rtype(result['features']) == ('Series', 'set', 'str')
        #assert rtype(result['history']) == ('Series', 'list', 'str')



######################################################################
# random_numbers
######################################################################

# @pytest.mark.jetzt  # pytest -m jetzt -x
class TestRandomNumbers:

    def test_random_numbers_default_size(self):
        # Teste, ob random_numbers mit der Standardgröße korrekt funktioniert
        result = random_numbers()
        assert rtype(result) == ('DataFrame', 'Series', 'int')
        assert result.shape == (1000, 5)
        assert result.index.name is None
        assert result.columns.tolist() == ['A', 'B', 'C', 'D', 'E']

    def test_random_numbers_custom_size(self):
        # Teste, ob random_numbers mit einer benutzerdefinierten Größe korrekt funktioniert
        result = random_numbers(size=50)
        assert rtype(result) == ('DataFrame', 'Series', 'int')
        assert result.shape == (50, 5)
        assert result.index.name is None
        assert result.columns.tolist() == ['A', 'B', 'C', 'D', 'E']

    def test_random_numbers_size_zero(self):
        # Teste, ob random_numbers mit size=0 korrekt funktioniert
        result = random_numbers(size=0)
        assert result.shape == (0, 5)
        assert result.index.name is None
        assert result.columns.tolist() == ['A', 'B', 'C', 'D', 'E']

    def test_random_numbers_size_one(self):
        # Teste, ob random_numbers mit size=1 korrekt funktioniert
        result = random_numbers(size=1)
        assert rtype(result) == ('DataFrame', 'Series', 'int')
        assert result.shape == (1, 5)
        assert result.index.name is None
        assert result.columns.tolist() == ['A', 'B', 'C', 'D', 'E']

    def test_random_numbers_size_two(self):
        # Teste, ob random_numbers mit size=2 korrekt funktioniert
        result = random_numbers(size=2)
        assert rtype(result) == ('DataFrame', 'Series', 'int')
        assert result.shape == (2, 5)
        assert result.index.name is None
        assert result.columns.tolist() == ['A', 'B', 'C', 'D', 'E']

    def test_random_numbers_column_types(self):
        # Teste, ob random_numbers die korrekten Spaltentypen hat
        result = random_numbers()
        assert rtype(result.iloc[:, 0]) == ('Series', 'int')
        assert rtype(result.iloc[:, 1]) == ('Series', 'int')
        assert rtype(result.iloc[:, 2]) == ('Series', 'float')
        assert rtype(result.iloc[:, 3]) == ('Series', 'float')
        # assert rtype(result.iloc[:, 4]) == ('Series', 'float') # das ist mal int mal float


######################################################################
# random_perlin
######################################################################

#@pytest.mark.jetzt  # pytest -m jetzt -x
class TestRandomPerlin:

    def test_random_perlin_default(self):
        # Teste, ob random_perlin mit Standardparametern korrekt funktioniert
        result = random_perlin(framework='pandas')
        assert isinstance(result, pd.DataFrame)
        assert rtype(result) == ('DataFrame', 'Series', 'float')
        assert result.shape == (100, 5)
        assert result.index.name is None
        for col in result.columns:
            assert result[col].min() >= -1
            assert result[col].max() <= 1

        result = random_perlin(framework='polars')
        assert isinstance(result, pl.DataFrame)
        assert rtype(result) == ('DataFrame', 'Series', 'float')
        assert result.shape == (100, 5)
        for col in result.columns:
            assert result[col].min() >= -1
            assert result[col].max() <= 1


    def test_random_perlin_custom_shape(self):
        # Teste, ob random_perlin mit benutzerdefinierter Form korrekt funktioniert
        result = random_perlin(shape=(50, 3), framework='pandas')
        assert isinstance(result, pd.DataFrame)
        assert rtype(result) == ('DataFrame', 'Series', 'float')
        assert result.shape == (50, 3)
        assert result.index.name is None
        for col in result.columns:
            assert result[col].min() >= -1
            assert result[col].max() <= 1

        result = random_perlin(shape=(50, 3), framework='polars')
        assert isinstance(result, pl.DataFrame)
        assert rtype(result) == ('DataFrame', 'Series', 'float')
        assert result.shape == (50, 3)
        for col in result.columns:
            assert result[col].min() >= -1
            assert result[col].max() <= 1


    def test_random_perlin_shape1(self):
        # Teste, ob random_perlin mit benutzerdefinierter Form korrekt funktioniert
        result = random_perlin(shape=(50, 1), framework='pandas')
        assert isinstance(result, pd.DataFrame)
        assert rtype(result) == ('DataFrame', 'Series', 'float')
        assert result.shape == (50, 1)
        assert result.index.name is None
        for col in result.columns:
            assert result[col].min() >= -1
            assert result[col].max() <= 1

        result = random_perlin(shape=(50, 1), framework='polars')
        assert isinstance(result, pl.DataFrame)
        assert rtype(result) == ('DataFrame', 'Series', 'float')
        assert result.shape == (50, 1)
        for col in result.columns:
            assert result[col].min() >= -1
            assert result[col].max() <= 1


    def test_random_perlin_custom_freq(self):
        # Teste, ob random_perlin mit benutzerdefinierten Frequenzen korrekt funktioniert
        result = random_perlin(freq=[1, 2, 3, 4], framework='pandas')
        assert isinstance(result, pd.DataFrame)
        assert rtype(result) == ('DataFrame', 'Series', 'float')
        assert result.shape == (100, 5)
        assert result.index.name is None
        for col in result.columns:
            assert result[col].min() >= -1
            assert result[col].max() <= 1

        result = random_perlin(freq=[1, 2, 3, 4], framework='polars')
        assert isinstance(result, pl.DataFrame)
        assert rtype(result) == ('DataFrame', 'Series', 'float')
        assert result.shape == (100, 5)
        for col in result.columns:
            assert result[col].min() >= -1
            assert result[col].max() <= 1

    def test_random_perlin_custom_op_add(self):
        # Teste, ob random_perlin mit op='add' korrekt funktioniert
        result = random_perlin(op='add', framework='pandas')
        assert rtype(result) == ('DataFrame', 'Series', 'float')
        assert result.shape == (100, 5)
        assert result.index.name is None
        for col in result.columns:
            assert result[col].min() >= -1
            assert result[col].max() <= 1

    def test_random_perlin_custom_op_mult(self):
        # Teste, ob random_perlin mit op='mult' korrekt funktioniert
        result = random_perlin(op='mult', framework='pandas')
        assert isinstance(result, pd.DataFrame)
        assert rtype(result) == ('DataFrame', 'Series', 'float')
        assert result.shape == (100, 5)
        assert result.index.name is None
        for col in result.columns:
            assert result[col].min() >= -1
            assert result[col].max() <= 1

    def test_random_perlin_custom_freq_int(self):
        # Teste, ob random_perlin mit benutzerdefinierten Frequenzen korrekt funktioniert
        result = random_perlin(freq=1, framework='pandas')
        assert isinstance(result, pd.DataFrame)
        assert rtype(result) == ('DataFrame', 'Series', 'float')
        assert result.shape == (100, 5)
        assert result.index.name is None
        for col in result.columns:
            assert result[col].min() >= -1
            assert result[col].max() <= 1






######################################################################
# dump_load_excel
######################################################################

# @pytest.mark.jetzt  # pytest -m jetzt -x
class TestDumpLoadExcel:

    def test_dump_load_excel_with_people(self, tmp_path):
        # Teste, ob dump_excel und load_excel mit people korrekt funktionieren
        df = excel_data(size=200, seed=21)
        filename = tmp_path / "test_people.xlsx"
        dump_excel(df, filename, check=False)
        df_loaded = load_excel(filename)
        assert check_equal(df, df_loaded)

    def test_dump_load_excel_with_tabcol(self, tmp_path):
        # Teste, ob dump_excel und load_excel mit tabcol korrekt funktionieren
        df = excel_data(size=200, seed=21)
        filename = tmp_path / "test_tabcol.xlsx"
        dump_excel(df, filename, tabcol='birthplace', check=False)
        df_loaded = load_excel(filename, tabcol='birthplace')
        df_loaded = move_cols(df_loaded, 'secret', -1) # Spaltenreihenfolge wiederherstellen
        df_loaded = reset_index(df_loaded, sort=True)
        df = reset_index(df, sort=True)
        assert check_equal(df, df_loaded)

    def test_dump_load_excel_with_random_numbers(self, tmp_path):
        # Teste, ob dump_excel und load_excel mit random_numbers korrekt funktionieren
        df = random_numbers(size=200)
        filename = tmp_path / "test_random_numbers.xlsx"
        dump_excel(df, filename, check=False)
        df_loaded = load_excel(filename)
        assert check_equal(df, df_loaded)


######################################################################
# dump_load_pickle
######################################################################

# @pytest.mark.jetzt  # pytest -m jetzt -x
class TestDumpLoadPickle:

    def test_dump_load_pickle_with_people(self, tmp_path):
        # Teste, ob dump_pickle und load_pickle mit people korrekt funktionieren
        df = people(size=200, seed=21)
        filename = tmp_path / "test_people.pickle"
        dump_pickle(df, filename)
        df_loaded = load_pickle(filename)
        assert check_equal(df, df_loaded)

    def test_dump_load_pickle_with_changedatatype(self, tmp_path):
        # Teste, ob dump_pickle und load_pickle mit changedatatype korrekt funktionieren
        df = people(size=200, seed=21)
        filename = tmp_path / "test_changedatatype.pickle"
        dump_pickle(df, filename, changedatatype=True)
        df_loaded = load_pickle(filename, changedatatype=True)
        assert check_equal(df, df_loaded)

    def test_dump_load_pickle_with_reset_index(self, tmp_path):
        # Teste, ob dump_pickle und load_pickle mit reset_index korrekt funktionieren
        df = people(size=200, seed=21)
        df = df.set_index('first_name')
        filename = tmp_path / "test_reset_index.pickle"
        dump_pickle(df, filename)
        df_loaded = load_pickle(filename, reset_index=True)
        assert check_equal(df.reset_index(), df_loaded)

    def test_dump_load_pickle_empty_dataframe(self, tmp_path):
        # Teste, ob dump_pickle und load_pickle mit einem leeren DataFrame korrekt funktionieren
        df = pd.DataFrame()
        filename = tmp_path / "test_empty.pickle"
        dump_pickle(df, filename)
        df_loaded = load_pickle(filename)
        assert check_equal(df, df_loaded)

    def test_dump_load_pickle_index(self, tmp_path):
        # Teste, ob dump_pickle und load_pickle mit einem DataFrame mit MultiIndex korrekt funktionieren
        df = people(size=200, seed=21)
        df = df.set_index('secret')
        filename = tmp_path / "test_multiindex.pickle"
        dump_pickle(df, filename)
        df_loaded = load_pickle(filename, reset_index=False)
        assert check_equal(df, df_loaded)

    def test_dump_load_pickle_multiindex(self, tmp_path):
        # Teste, ob dump_pickle und load_pickle mit einem DataFrame mit MultiIndex korrekt funktionieren
        df = people(size=200, seed=21)
        df = df.set_index(['secret','age'])
        filename = tmp_path / "test_multiindex.pickle"
        dump_pickle(df, filename)
        df_loaded = load_pickle(filename, reset_index=False)
        assert check_equal(df, df_loaded)

