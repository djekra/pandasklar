import pytest
import pandas as pd
import numpy as np
from pandasklar.analyse import change_datatype
from pandasklar.compare import compare_dataframes
from bpyth import rtype
import pandasklar.content as pak
import random
from .some_utils import generate_random_data_tcdt



#@pytest.mark.jetzt # pytest -m jetzt -x
class TestChangeDatatype:

    def test_change_datatype_automatic(self):

        df = generate_random_data_tcdt()
        assert rtype(df['int_grob']) == ('Series', 'int')

        # int_grob             --> pd.Int8
        result = change_datatype(df['int_grob'])
        assert result.dtype == pd.Int8Dtype()

        # int_fein             --> pd.Int8
        result = change_datatype(df['int_fein'])
        assert result.dtype == pd.Int8Dtype()

        # float_summe          --> np.float32
        result = change_datatype(df['float_summe'])
        assert result.dtype == np.float32

        # int_nan              --> pd.Int8
        result = change_datatype(df['int_nan'])
        assert result.dtype == pd.Int8Dtype()

        # float_nan            --> np.float32
        result = change_datatype(df['float_nan'])
        assert result.dtype == np.float32

        # City                 --> pd.string
        result = change_datatype(df['City'])
        assert result.dtype == pd.StringDtype()



    def test_change_datatype_with_specified_type(self):
        # Teste, ob change_datatype einen vorgegebenen Datentyp korrekt anwendet
        df = generate_random_data_tcdt()
        result = change_datatype(df['int_grob'], search='Int16')
        assert result.dtype == pd.Int16Dtype()

    def test_change_datatype_mixed_types(self):
        # Teste, ob change_datatype Spalten mit gemischten Typen korrekt in object konvertiert
        df = pd.DataFrame({'mixed': [1, 'a', 2.2, True]})
        result = change_datatype(df['mixed'])
        assert result.dtype == object


    def test_change_datatype_large_numbers(self):
        # Teste, ob change_datatype mit großen Zahlen korrekt umgeht
        df = pd.DataFrame({'large_num': [1000000000000, 2000000000000, 3000000000000]})
        result = change_datatype(df['large_num'])
        assert result.dtype == np.int64

    def test_change_datatype_verbose(self, capsys):
        # Teste, ob change_datatype mit verbose korrekt umgeht
        df = generate_random_data_tcdt()
        change_datatype(df, verbose=True)
        captured = capsys.readouterr()
        assert "change_datatype" in captured.out

    def test_change_datatype_category_maxsize(self):
        # Teste, ob change_datatype mit category_maxsize korrekt umgeht
        df = generate_random_data_tcdt()
        result = change_datatype(df['City'], category_maxsize=1)
        assert result.dtype == pd.StringDtype()

    def test_change_datatype_series(self):
        # Teste, ob change_datatype mit Series korrekt umgeht
        df = generate_random_data_tcdt()
        result = change_datatype(df['int_grob'])
        assert result.dtype == pd.Int8Dtype()

    def test_change_datatype_dataframe(self):
        # Teste, ob change_datatype mit einem kompletten DataFrame korrekt umgeht
        df = generate_random_data_tcdt()
        result = change_datatype(df)

        assert result['int_fein'].dtype == pd.Int8Dtype()
        assert result['float_summe'].dtype == np.float32
        assert result['int_nan'].dtype == pd.Int8Dtype()
        assert result['float_nan'].dtype == np.float32
        assert result['City'].dtype == pd.StringDtype()
        assert result['first_name'].dtype == pd.StringDtype()
        assert result['Letter1'].dtype == pd.StringDtype()
        assert result['string_nonan'].dtype == pd.StringDtype()
        assert result['string_nan'].dtype == pd.StringDtype()
        assert result['List'].dtype == object
        assert result['time'].dtype == 'datetime64[ns]'
        assert result['Mix'].dtype == object

    def test_change_datatype_content(self):
        # Teste, ob change_datatype den Inhalt des DataFrames nicht verändert
        df = generate_random_data_tcdt()
        df_copy = df.copy()
        result = change_datatype(df)
        comparison = compare_dataframes(df_copy, result)
        assert comparison.loc['(Total)', 'content'] == False

        comparison = compare_dataframes(df_copy, result, decimals=2)
        assert comparison.loc['(Total)', 'content'] == True

















