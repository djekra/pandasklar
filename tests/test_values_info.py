import pytest
import pandas as pd
import numpy as np
from pandasklar.values_info import values_info


#@pytest.mark.jetzt # pytest -m jetzt -x
class TestValuesInfo:

    def test_values_info_numeric_series(self):
        # Teste, ob values_info mit einer numerischen Series korrekt funktioniert
        s = pd.Series([1, 2, 3, 4, 5])
        vi = values_info(s)
        assert vi.n == 5
        assert vi.nnan == 0
        assert vi.ntypes == 1
        assert vi.nunique == 5
        assert vi.ndups == 0
        assert vi.vmin == 1
        assert vi.vmax == 5
        assert vi.vmean == 3
        assert vi.vmedian == 3
        assert vi.vsum == 15

    def test_values_info_string_series(self):
        # Teste, ob values_info mit einer String-Series korrekt funktioniert
        s = pd.Series(['a', 'b', 'c', 'd', 'e'])
        vi = values_info(s)
        assert vi.n == 5
        assert vi.nnan == 0
        assert vi.ntypes == 1
        assert vi.nunique == 5
        assert vi.ndups == 0
        assert vi.vmin == 'a'
        assert vi.vmax == 'e'
        assert np.isnan(vi.vmean)
        assert np.isnan(vi.vmedian)
        assert np.isnan(vi.vsum)

    def test_values_info_with_nans(self):
        # Teste, ob values_info mit NaNs korrekt funktioniert
        s = pd.Series([1, 2, np.nan, 4, np.nan])
        vi = values_info(s)
        assert vi.n == 5
        assert vi.nnan == 2
        assert vi.ntypes == 1
        assert vi.nunique == 3
        assert vi.ndups == 0
        assert vi.vmin == 1
        assert vi.vmax == 4
        assert vi.vmean == 2.33
        assert vi.vmedian == 2
        assert vi.vsum == 7

    def test_values_info_with_duplicates(self):
        # Teste, ob values_info mit Duplikaten korrekt funktioniert
        s = pd.Series([1, 2, 2, 3, 3])
        vi = values_info(s)
        assert vi.n == 5
        assert vi.nnan == 0
        assert vi.ntypes == 1
        assert vi.nunique == 3
        assert vi.ndups == 2
        assert vi.vmin == 1
        assert vi.vmax == 3
        assert vi.vmean == 2.2
        assert vi.vmedian == 2
        assert vi.vsum == 11

    def test_values_info_mixed_datatypes(self):
        # Teste, ob values_info mit gemischten Datentypen korrekt funktioniert
        s = pd.Series([1, 'a', 2, 'b', 3])
        vi = values_info(s)
        assert vi.n == 5
        assert vi.nnan == 0
        assert vi.ntypes == 2
        assert vi.nunique == 5
        assert vi.ndups == 0
        assert pd.isna(vi.vmin)
        assert pd.isna(vi.vmax)
        assert np.isnan(vi.vmean)
        assert np.isnan(vi.vmedian)
        assert np.isnan(vi.vsum)

    def test_values_info_empty_series(self):
        # Teste, ob values_info mit einer leeren Series korrekt funktioniert
        s = pd.Series([], dtype=object)
        vi = values_info(s)
        assert vi.n == 0
        assert vi.nnan == 0
        assert vi.ntypes == 0
        assert vi.nunique == 0
        assert vi.ndups == 0
        assert np.isnan(vi.vmin)
        assert np.isnan(vi.vmax)
        assert np.isnan(vi.vmean)
        assert np.isnan(vi.vmedian)
        assert np.isnan(vi.vsum)

    def test_values_info_different_datatypes(self):
        # Teste, ob values_info mit verschiedenen Datentypen korrekt funktioniert
        vi_int = values_info(pd.Series([1, 2, 3]))
        assert vi_int.datatype_identified == 'int'
        vi_float = values_info(pd.Series([1.1, 2.2, 3.3]))
        assert vi_float.datatype_identified == 'float'
        vi_string = values_info(pd.Series(['a', 'b', 'c']))
        assert vi_string.datatype_identified == 'string'
        vi_bool = values_info(pd.Series([True, False, True]))
        assert vi_bool.datatype_identified == 'bool'
        vi_datetime = values_info(pd.Series(pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])))
        assert vi_datetime.datatype_identified == 'datetime'
        vi_object = values_info(pd.Series([1, 'a', True]))
        assert vi_object.datatype_identified == ''

    def test_values_info_datatype_suggest(self):
        # Teste, ob datatype_suggest korrekt funktioniert
        vi_int = values_info(pd.Series([1, 2, 3]))
        assert vi_int.datatype_suggest == 'pd.Int8'
        vi_float = values_info(pd.Series([1.1, 2.2, 3.3]))
        assert vi_float.datatype_suggest == 'np.float32'
        vi_string = values_info(pd.Series(['a', 'b', 'c']))
        assert vi_string.datatype_suggest == 'pd.string'
        vi_category = values_info(pd.Series(['a', 'b', 'a']))
        assert vi_category.datatype_suggest == 'pd.string'
        vi_int_with_nan = values_info(pd.Series([1, 2, np.nan, 3]))
        assert vi_int_with_nan.datatype_suggest == 'pd.Int8'
        vi_float_with_nan = values_info(pd.Series([1.1, 2.2, np.nan, 3.3]))
        assert vi_float_with_nan.datatype_suggest == 'np.float32'
        vi_int_with_nan_nanless = values_info(pd.Series([1, 2, np.nan, 3]), nanless_ints=True)
        assert vi_int_with_nan_nanless.datatype_suggest == 'pd.Int8'
        vi_int_with_nan_nanless_int = values_info(pd.Series([1, 2, 3]), nanless_ints=True)
        assert vi_int_with_nan_nanless_int.datatype_suggest == 'np.int8'
        vi_float_with_nan_nanless = values_info(pd.Series([1.1, 2.2, np.nan, 3.3]), nanless_ints=True)
        assert vi_float_with_nan_nanless.datatype_suggest == 'np.float32'
        vi_float_with_nan_nanless_float = values_info(pd.Series([1.1, 2.2, 3.3]), nanless_ints=True)
        assert vi_float_with_nan_nanless_float.datatype_suggest == 'np.float32'

    def test_values_info_datatype_identified(self):
        # Teste, ob datatype_identified korrekt funktioniert
        vi_int = values_info(pd.Series([1, 2, 3]))
        assert vi_int.datatype_identified == 'int'
        vi_float = values_info(pd.Series([1.1, 2.2, 3.3]))
        assert vi_float.datatype_identified == 'float'
        vi_string = values_info(pd.Series(['a', 'b', 'c']))
        assert vi_string.datatype_identified == 'string'
        vi_bool = values_info(pd.Series([True, False, True]))
        assert vi_bool.datatype_identified == 'bool'
        vi_datetime = values_info(pd.Series(pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])))
        assert vi_datetime.datatype_identified == 'datetime'
        vi_object = values_info(pd.Series([1, 'a', True]))
        assert vi_object.datatype_identified == ''
        vi_list = values_info(pd.Series(['a,b', 'c,d', 'e,f']))
        assert vi_list.datatype_identified == 'list'

    def test_values_info_category_maxsize(self):
        # Teste, ob category_maxsize korrekt funktioniert
        vi_category = values_info(pd.Series(['a', 'b', 'a']), category_maxsize=2)
        assert vi_category.datatype_suggest == 'pd.string'
        vi_string = values_info(pd.Series(['a', 'b', 'c']), category_maxsize=2)
        assert vi_string.datatype_suggest == 'pd.string'

    def test_values_info_nanless_ints(self):
        # Teste, ob nanless_ints korrekt funktioniert
        vi_int = values_info(pd.Series([1, 2, 3]), nanless_ints=True)
        assert vi_int.datatype_suggest == 'np.int8'
        vi_int_with_nan = values_info(pd.Series([1, 2, np.nan, 3]), nanless_ints=True)
        assert vi_int_with_nan.datatype_suggest == 'pd.Int8'
        vi_float_with_nan = values_info(pd.Series([1.1, 2.2, np.nan, 3.3]), nanless_ints=True)
        assert vi_float_with_nan.datatype_suggest == 'np.float32'
        vi_float_with_nan_nanless = values_info(pd.Series([1.1, 2.2, 3.3]), nanless_ints=True)
        assert vi_float_with_nan_nanless.datatype_suggest == 'np.float32'
        vi_int_with_nan_nanless_int = values_info(pd.Series([1, 2, 3]), nanless_ints=True)
        assert vi_int_with_nan_nanless_int.datatype_suggest == 'np.int8'
        vi_int_with_nan_nanless_int_with_nan = values_info(pd.Series([1, 2, np.nan, 3]), nanless_ints=True)
        assert vi_int_with_nan_nanless_int_with_nan.datatype_suggest == 'pd.Int8'
        vi_float_with_nan_nanless_float = values_info(pd.Series([1.1, 2.2, 3.3]), nanless_ints=True)
        assert vi_float_with_nan_nanless_float.datatype_suggest == 'np.float32'
        vi_float_with_nan_nanless_float_with_nan = values_info(pd.Series([1.1, 2.2, np.nan, 3.3]), nanless_ints=True)
        assert vi_float_with_nan_nanless_float_with_nan.datatype_suggest == 'np.float32'
