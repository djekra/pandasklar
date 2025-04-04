
import pytest
import pandas as pd
import numpy as np
from pandasklar.analyse import change_datatype
from pandasklar.pandas import copy_datatype
from pandasklar.compare import compare_dataframes
from bpyth import rtype
import pandasklar.content as pak
import random
from .some_utils import generate_random_data_tcdt # Import aus test_utils.py

#@pytest.mark.jetzt # pytest -m jetzt -x
class TestCopyDatatype:

    def notest_copy_datatype(self):
        # Teste, ob copy_datatype die Datentypen korrekt kopiert
        df = generate_random_data_tcdt()
        dfc= generate_random_data_tcdt()
        dfc = change_datatype(dfc)

        # Überprüfe die ursprünglichen Datentypen
        assert df['int_grob'].dtype == np.int32
        assert df['int_fein'].dtype == np.int32
        assert df['float_summe'].dtype == np.float64
        assert df['int_nan'].dtype == pd.Int64Dtype()
        assert df['float_nan'].dtype == np.float64
        assert df['City'].dtype == object


        result = copy_datatype(df, dfc)

        # Überprüfe, ob die Datentypen korrekt kopiert wurden
        assert result['int_grob'].dtype == pd.Int8Dtype()
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

        result2 = copy_datatype(result, df)

        assert result2['int_grob'].dtype == np.int32
        assert result2['int_fein'].dtype == np.int32
        assert result2['float_summe'].dtype == np.float64
        assert result2['int_nan'].dtype == pd.Int64Dtype()
        assert result2['float_nan'].dtype == np.float64
        assert result2['City'].dtype == object

        # Vergleiche die Inhalte mit compare_dataframes
        comparison_df_dfc = compare_dataframes(df, dfc, decimals=5)
        assert comparison_df_dfc.loc['(Total)', 'content'] == False

        comparison_df_dfc = compare_dataframes(df, dfc, decimals=3)
        assert comparison_df_dfc.loc['(Total)', 'content'] == True

        comparison_df_result = compare_dataframes(df, result, decimals=5)
        assert comparison_df_result.loc['(Total)', 'content'] == False

        comparison_df_result = compare_dataframes(df, result, decimals=2)
        assert comparison_df_result.loc['(Total)', 'content'] == True


        comparison_df_result2 = compare_dataframes(df, result2, decimals=5)
        assert comparison_df_result2.loc['(Total)', 'content'] == False

        comparison_df_result2 = compare_dataframes(df, result2, decimals=1)
        assert comparison_df_result2.loc['(Total)', 'content'] == True


        comparison_dfc_result = compare_dataframes(dfc, result, decimals=2)
        assert comparison_dfc_result.loc['(Total)', 'content'] == True

        comparison_dfc_result2 = compare_dataframes(dfc, result2, decimals=2)
        assert comparison_dfc_result2.loc['(Total)', 'content'] == True

        comparison_result_result2 = compare_dataframes(result, result2, decimals=2)
        assert comparison_result_result2.loc['(Total)', 'content'] == True


