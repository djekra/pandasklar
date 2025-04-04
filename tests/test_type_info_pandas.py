import pytest
import pandas as pd
import polars as pl
import numpy as np
from pandasklar.type_info_pandas import type_info_pandas


@pytest.mark.jetzt # pytest -m jetzt -x
class TestTypeInfo:

    def test_type_info_with_string(self):
        # Teste, ob type_info mit einem String korrekt funktioniert
        ti = type_info_pandas("Int32")
        assert ti.name == "pd.Int32"
        assert ti.name_short == "Int32"
        assert ti.framework == "pd"
        assert ti.nan_allowed == True
        assert ti.is_hashable == True
        assert ti.xmin == -2147483648
        assert ti.xmax == 2147483647

    def test_type_info_with_class(self):
        # Teste, ob type_info mit einer Klasse korrekt funktioniert
        ti = type_info_pandas(np.float64)
        assert ti.name == "np.float64"
        assert ti.name_short == "float64"
        assert ti.framework == "np"
        assert ti.nan_allowed == True
        assert ti.is_hashable == True
        assert ti.xmin == -1.7976931348623157e+308
        assert ti.xmax == 1.7976931348623157e+308

    def test_type_info_with_series(self):
        # Teste, ob type_info mit einer Series korrekt funktioniert
        s = pd.Series([1, 2, 3])
        ti = type_info_pandas(s)
        assert ti.name == "np.int64"
        assert ti.name_short == "int64"
        assert ti.framework == "np"
        assert ti.nan_allowed == False
        assert ti.is_hashable == True
        assert ti.instance1 == 1
        assert ti.instance2 == 3
        assert ti.xmin == -9223372036854775808
        assert ti.xmax == 9223372036854775807

    def test_type_info_different_datatypes(self):
        # Teste, ob type_info mit verschiedenen Datentypen korrekt funktioniert
        ti_int = type_info_pandas("int32")
        assert ti_int.name == "np.int32"
        ti_float = type_info_pandas("float64")
        assert ti_float.name == "np.float64"
        ti_string = type_info_pandas("string")
        assert ti_string.name == "pd.string"
        ti_bool = type_info_pandas(pd.Series([True, False]))
        assert ti_bool.name == "bool"
        ti_datetime = type_info_pandas(pd.Series(pd.to_datetime(['2023-01-01', '2023-01-02'])))
        assert ti_datetime.name == "datetime64[ns]"
        ti_category = type_info_pandas(pd.Series(['a', 'b', 'a']).astype('category'))
        assert ti_category.name == "pd.category"
        ti_object = type_info_pandas(pd.Series([1, 'a', True]))
        assert ti_object.name == "object"

    def test_type_info_hashable(self):
        # Teste, ob is_hashable korrekt gesetzt wird
        ti_int = type_info_pandas("int32")
        assert ti_int.is_hashable == True
        ti_list = type_info_pandas(pd.Series([[1, 2], [3, 4]]))
        assert ti_list.is_hashable == False
        ti_dict = type_info_pandas(pd.Series([{'a': 1}, {'b': 2}]))
        assert ti_dict.is_hashable == False
        ti_set = type_info_pandas(pd.Series([{1, 2}, {3, 4}]))
        assert ti_set.is_hashable == False
        ti_tuple = type_info_pandas(pd.Series([(1, 2), (3, 4)]))
        assert ti_tuple.is_hashable == True
        ti_frozenset = type_info_pandas(pd.Series([frozenset([1, 2]), frozenset([3, 4])]))
        assert ti_frozenset.is_hashable == True

    def test_type_info_nan_allowed(self):
        # Teste, ob nan_allowed korrekt gesetzt wird
        ti_int = type_info_pandas("int32")
        assert ti_int.nan_allowed == False
        ti_float = type_info_pandas("float64")
        assert ti_float.nan_allowed == True
        ti_string = type_info_pandas("string")
        assert ti_string.nan_allowed == True
        ti_int8 = type_info_pandas("Int8")
        assert ti_int8.nan_allowed == True
        ti_int16 = type_info_pandas("Int16")
        assert ti_int16.nan_allowed == True
        ti_int32 = type_info_pandas("Int32")
        assert ti_int32.nan_allowed == True
        ti_int64 = type_info_pandas("Int64")
        assert ti_int64.nan_allowed == True
        ti_uint8 = type_info_pandas("uint8")
        assert ti_uint8.nan_allowed == False
        ti_uint16 = type_info_pandas("uint16")
        assert ti_uint16.nan_allowed == False
        ti_uint32 = type_info_pandas("uint32")
        assert ti_uint32.nan_allowed == False
        ti_uint64 = type_info_pandas("uint64")
        assert ti_uint64.nan_allowed == False
        ti_float16 = type_info_pandas("float16")
        assert ti_float16.nan_allowed == True
        ti_float32 = type_info_pandas("float32")
        assert ti_float32.nan_allowed == True
        ti_float64 = type_info_pandas("float64")
        assert ti_float64.nan_allowed == True

    def test_type_info_valid_values(self):
        # Teste, ob xmin und xmax korrekt gesetzt werden
        ti_int8 = type_info_pandas("int8")
        assert ti_int8.xmin == -128
        assert ti_int8.xmax == 127
        ti_uint8 = type_info_pandas("uint8")
        assert ti_uint8.xmin == 0
        assert ti_uint8.xmax == 255
        ti_float32 = type_info_pandas("float32")
        assert ti_float32.xmin == -3.4028234663852886e+38
        assert ti_float32.xmax == 3.4028234663852886e+38
        ti_int16 = type_info_pandas("int16")
        assert ti_int16.xmin == -32768
        assert ti_int16.xmax == 32767
        ti_int32 = type_info_pandas("int32")
        assert ti_int32.xmin == -2147483648
        assert ti_int32.xmax == 2147483647
        ti_int64 = type_info_pandas("int64")
        assert ti_int64.xmin == -9223372036854775808
        assert ti_int64.xmax == 9223372036854775807
        ti_uint16 = type_info_pandas("uint16")
        assert ti_uint16.xmin == 0
        assert ti_uint16.xmax == 65535
        ti_uint32 = type_info_pandas("uint32")
        assert ti_uint32.xmin == 0
        assert ti_uint32.xmax == 4294967295
        ti_uint64 = type_info_pandas("uint64")
        assert ti_uint64.xmin == 0
        assert ti_uint64.xmax == 18446744073709551615
        ti_float16 = type_info_pandas("float16")
        assert ti_float16.xmin == -65504.0
        assert ti_float16.xmax == 65504.0

    def test_type_info_instance(self):
        # Teste, ob instance1 und instance2 korrekt gesetzt werden
        s = pd.Series([1, 2, 3])
        ti = type_info_pandas(s)
        assert ti.instance1 == 1
        assert ti.instance2 == 3
        s = pd.Series([1, 2, None, 3])
        ti = type_info_pandas(s)
        assert ti.instance1 == 1
        assert ti.instance2 == 3
        s = pd.Series([None, None, None])
        ti = type_info_pandas(s)
        assert ti.instance1 == None
        assert ti.instance2 == None
        s = pd.Series([1, 2, None, 3, None])
        ti = type_info_pandas(s)
        assert ti.instance1 == 1
        assert ti.instance2 == 3
        s = pd.Series([None, 1, 2, None, 3, None])
        ti = type_info_pandas(s)
        assert ti.instance1 == 1
        assert ti.instance2 == 3
        s = pd.Series([None, None, None], dtype='Int64')
        ti = type_info_pandas(s)
        assert ti.instance1 == None
        assert ti.instance2 == None
        s = pd.Series([None, 1, 2, None, 3, None], dtype='Int64')
        ti = type_info_pandas(s)
        assert ti.instance1 == 1
        assert ti.instance2 == 3
        s = pd.Series([None, None, None], dtype='float64')
        ti = type_info_pandas(s)
        assert ti.instance1 == None
        assert ti.instance2 == None
        s = pd.Series([None, 1.1, 2.2, None, 3.3, None], dtype='float64')
        ti = type_info_pandas(s)
        assert ti.instance1 == 1.1
        assert ti.instance2 == 3.3
        s = pd.Series([None, None, None], dtype='string')
        ti = type_info_pandas(s)
        assert ti.instance1 == None
        assert ti.instance2 == None
        s = pd.Series([None, 'a', 'b', None, 'c', None], dtype='string')
        ti = type_info_pandas(s)
        assert ti.instance1 == 'a'
        assert ti.instance2 == 'c'
        s = pd.Series([None, None, None], dtype='object')
        ti = type_info_pandas(s)
        assert ti.instance1 == None
        assert ti.instance2 == None
        s = pd.Series([None, [1, 2], [3, 4], None, [5, 6], None], dtype='object')
        ti = type_info_pandas(s)
        assert ti.instance1 == [1, 2]
        assert ti.instance2 == [5, 6]
        s = pd.Series([None, (1, 2), (3, 4), None, (5, 6), None], dtype='object')
        ti = type_info_pandas(s)
        assert ti.instance1 == (1, 2)
        assert ti.instance2 == (5, 6)
        s = pd.Series([None, {'a': 1}, {'b': 2}, None, {'c': 3}, None], dtype='object')
        ti = type_info_pandas(s)
        assert ti.instance1 == {'a': 1}
        assert ti.instance2 == {'c': 3}
        s = pd.Series([None, {1, 2}, {3, 4}, None, {5, 6}, None], dtype='object')
        ti = type_info_pandas(s)
        assert ti.instance1 == {1, 2}
        assert ti.instance2 == {5, 6}
        s = pd.Series([None, frozenset([1, 2]), frozenset([3, 4]), None, frozenset([5, 6]), None], dtype='object')
        ti = type_info_pandas(s)
        assert ti.instance1 == frozenset([1, 2])
        assert ti.instance2 == frozenset([5, 6])
        s = pd.Series([None, range(1, 3), range(3, 5), None, range(5, 7), None], dtype='object')
        ti = type_info_pandas(s)
        assert ti.instance1 == range(1, 3)
        assert ti.instance2 == range(5, 7)
        s = pd.Series([None, None, None, None, None, None], dtype='object')
        ti = type_info_pandas(s)
        assert ti.instance1 == None
        assert ti.instance2 == None
        s = pd.Series([None, None, None, None, None, None], dtype='Int64')
        ti = type_info_pandas(s)
        assert ti.instance1 == None
        assert ti.instance2 == None
        s = pd.Series([None, None, None, None, None, None], dtype='float64')
        ti = type_info_pandas(s)
        assert ti.instance1 == None
        assert ti.instance2 == None
        s = pd.Series([None, None, None, None, None, None], dtype='string')
        ti = type_info_pandas(s)
        assert ti.instance1 == None
        assert ti.instance2 == None

    def test_type_info_mixed_types(self):
        # Teste, ob type_info mit gemischten Typen korrekt funktioniert
        s = pd.Series([1, 'a', True])
        ti = type_info_pandas(s)
        assert ti.name_instance == 'mix'
        assert ti.is_hashable == False
        s = pd.Series([1, 'a', True, None])
        ti = type_info_pandas(s)
        assert ti.name_instance == 'mix'
        assert ti.is_hashable == False
        s = pd.Series([1, 'a', True, None, 1.1])
        ti = type_info_pandas(s)
        assert ti.name_instance == 'mix'
        assert ti.is_hashable == False

    def test_type_info_empty_series(self):
        # Teste, ob type_info mit einer leeren Series korrekt funktioniert
        s = pd.Series([], dtype=object)
        ti = type_info_pandas(s)
        assert ti.name == "object"
        assert ti.name_instance == ""

        s = pd.Series([], dtype='int64')
        ti = type_info_pandas(s)
        assert ti.name == "np.int64"
        assert ti.name_instance == ""

        s = pd.Series([], dtype='float64')
        ti = type_info_pandas(s)
        assert ti.name == "np.float64"
        assert ti.name_instance == ""

        s = pd.Series([], dtype='string')
        ti = type_info_pandas(s)
        assert ti.name == "pd.string"
        assert ti.name_instance == ""



