import pytest
import pandas as pd
import numpy as np
from pandasklar.analyse import analyse_datatypes
from pandasklar.content import people, random_numbers
from pandasklar.dataframe import dataframe


#@pytest.mark.jetzt
class TestAnalyseDatatypes:

    def test_analyse_datatypes_empty_dataframe(self):
        # Teste, ob analyse_datatypes mit einem leeren DataFrame korrekt funktioniert
        df = pd.DataFrame()
        result = analyse_datatypes(df)
        assert result.shape == (1, 10)

    def test_analyse_datatypes_different_datatypes(self):
        # Teste, ob analyse_datatypes mit verschiedenen Datentypen korrekt funktioniert
        df = people(size=10, seed=84)
        result = analyse_datatypes(df)
        result = result.set_index('col_name')  # Index auf col_name setzen
        assert result.shape[0] == len(df.columns) + 1
        assert result.shape[1] == 9
        assert result.loc['__index__', 'datatype_instance'] == 'int64'
        assert result.loc['__index__', 'datatype'] == 'np.int64'
        assert result.loc['__index__', 'datatype_short'] == 'int64'
        assert result.loc['__index__', 'is_numeric'] == True
        assert result.loc['__index__', 'is_string'] == False
        assert result.loc['__index__', 'is_datetime'] == False
        assert result.loc['__index__', 'is_hashable'] == True
        assert result.loc['__index__', 'nan_allowed'] == False
        assert result.loc['first_name', 'datatype_instance'] == 'str'
        assert result.loc['first_name', 'datatype'] == 'pd.string'
        assert result.loc['first_name', 'datatype_short'] == 'string'
        assert result.loc['first_name', 'is_numeric'] == False
        assert result.loc['first_name', 'is_string'] == True
        assert result.loc['first_name', 'is_datetime'] == False
        assert result.loc['first_name', 'is_hashable'] == True
        assert result.loc['first_name', 'nan_allowed'] == True
        assert result.loc['age', 'datatype_instance'] == 'int8'
        assert result.loc['age', 'datatype'] == 'pd.Int8'
        assert result.loc['age', 'datatype_short'] == 'Int8'
        assert result.loc['age', 'is_numeric'] == True
        assert result.loc['age', 'is_string'] == False
        assert result.loc['age', 'is_datetime'] == False
        assert result.loc['age', 'is_hashable'] == True
        assert result.loc['age', 'nan_allowed'] == True
        assert result.loc['age_class', 'datatype_instance'] == 'int8'
        assert result.loc['age_class', 'datatype'] == 'pd.Int8'
        assert result.loc['age_class', 'datatype_short'] == 'Int8'
        assert result.loc['age_class', 'is_numeric'] == True
        assert result.loc['age_class', 'is_string'] == False
        assert result.loc['age_class', 'is_datetime'] == False
        assert result.loc['age_class', 'is_hashable'] == True
        assert result.loc['age_class', 'nan_allowed'] == True
        assert result.loc['postal_code', 'datatype_instance'] == 'int32'
        assert result.loc['postal_code', 'datatype'] == 'pd.Int32'
        assert result.loc['postal_code', 'datatype_short'] == 'Int32'
        assert result.loc['postal_code', 'is_numeric'] == True
        assert result.loc['postal_code', 'is_string'] == False
        assert result.loc['postal_code', 'is_datetime'] == False
        assert result.loc['postal_code', 'is_hashable'] == True
        assert result.loc['postal_code', 'nan_allowed'] == True
        assert result.loc['birthplace', 'datatype_instance'] == 'str'
        assert result.loc['birthplace', 'datatype'] == 'pd.string'
        assert result.loc['birthplace', 'datatype_short'] == 'string'
        assert result.loc['birthplace', 'is_numeric'] == False
        assert result.loc['birthplace', 'is_string'] == True
        assert result.loc['birthplace', 'is_datetime'] == False
        assert result.loc['birthplace', 'is_hashable'] == True
        assert result.loc['birthplace', 'nan_allowed'] == True
        assert result.loc['secret', 'datatype_instance'] == 'str'
        assert result.loc['secret', 'datatype'] == 'pd.string'
        assert result.loc['secret', 'datatype_short'] == 'string'
        assert result.loc['secret', 'is_numeric'] == False
        assert result.loc['secret', 'is_string'] == True
        assert result.loc['secret', 'is_datetime'] == False
        assert result.loc['secret', 'is_hashable'] == True
        assert result.loc['secret', 'nan_allowed'] == True
        assert result.loc['features', 'datatype_instance'] == 'set'
        assert result.loc['features', 'datatype'] == 'object'
        assert result.loc['features', 'datatype_short'] == 'object'
        assert result.loc['features', 'is_numeric'] == False
        assert result.loc['features', 'is_string'] == False
        assert result.loc['features', 'is_datetime'] == False
        assert result.loc['features', 'is_hashable'] == False
        assert result.loc['features', 'nan_allowed'] == True
        assert result.loc['history', 'datatype_instance'] == 'list'
        assert result.loc['history', 'datatype'] == 'object'
        assert result.loc['history', 'datatype_short'] == 'object'
        assert result.loc['history', 'is_numeric'] == False
        assert result.loc['history', 'is_string'] == False
        assert result.loc['history', 'is_datetime'] == False
        assert result.loc['history', 'is_hashable'] == False
        assert result.loc['history', 'nan_allowed'] == True






