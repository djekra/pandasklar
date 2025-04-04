import pytest
import pandas as pd
import numpy as np
from pandasklar.analyse import change_datatype
from pandasklar.compare import compare_dataframes
from pandasklar.content import people, random_numbers
from pandasklar.pandas import copy_datatype
from .some_utils import generate_random_data_tcdt

#@pytest.mark.jetzt # pytest -m jetzt -x
class TestCompareDataframes:

    def test_compare_dataframes_identical(self):
        # Teste, ob compare_dataframes mit identischen DataFrames korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84)
        result = compare_dataframes(df1, df2)
        assert result.loc['(Total)', 'name'] == True
        assert result.loc['(Total)', 'dtype'] == True
        assert result.loc['(Total)', 'nnan'] == True
        assert result.loc['(Total)', 'nan_pat'] == True
        assert result.loc['(Total)', 'content'] == True
        assert result.loc['(Total)', 'sort'] == True
        assert result.loc['(Total)', 'eq'] == True

    def test_compare_dataframes_empty(self):
        # Teste, ob compare_dataframes mit zwei leeren DataFrames korrekt funktioniert
        df1 = pd.DataFrame()
        df2 = pd.DataFrame()
        result = compare_dataframes(df1, df2)
        assert result.loc['(Total)', 'name'] == True
        assert result.loc['(Total)', 'dtype'] == True
        assert result.loc['(Total)', 'nnan'] == True
        assert result.loc['(Total)', 'nan_pat'] == True
        assert result.loc['(Total)', 'content'] == True
        assert result.loc['(Total)', 'sort'] == True
        assert result.loc['(Total)', 'eq'] == True


    def test_compare_dataframes_different_columns(self):
        # Teste, ob compare_dataframes mit unterschiedlichen Spaltennamen korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84).rename(columns={'first_name': 'name'})
        result = compare_dataframes(df1, df2)
        assert result.loc['first_name', 'name'] == 'left_only'
        assert result.loc['name', 'name'] == 'right_only'
        assert result.loc['age', 'eq'] == True


    def test_compare_dataframes_different_dtype(self):
        # Teste, ob compare_dataframes mit unterschiedlichen Datentypen korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84).astype({'age': 'float64'})
        result = compare_dataframes(df1, df2)
        assert result.loc['(Total)', 'name'] == True
        assert result.loc['(Total)', 'dtype'] == False
        assert result.loc['(Total)', 'nnan'] == True
        assert result.loc['(Total)', 'nan_pat'] == True
        assert result.loc['(Total)', 'content'] == True
        #assert result.loc['(Total)', 'sort'] == True
        assert result.loc['(Total)', 'eq'] == True

    def test_compare_dataframes_different_nnan(self):
        # Teste, ob compare_dataframes mit unterschiedlicher Anzahl von NaNs korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84)
        df2.loc[0, 'birthplace'] = None
        result = compare_dataframes(df1, df2)
        assert result.loc['(Total)', 'name'] == True
        assert result.loc['(Total)', 'dtype'] == True
        assert result.loc['(Total)', 'nnan'] == False
        assert result.loc['(Total)', 'nan_pat'] == False
        assert result.loc['(Total)', 'content'] == False
        assert result.loc['(Total)', 'sort'] == False
        assert result.loc['(Total)', 'eq'] == False

    def test_compare_dataframes_different_content(self):
        # Teste, ob compare_dataframes mit unterschiedlichem Inhalt korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84)
        df2.loc[0, 'first_name'] = 'Test'
        result = compare_dataframes(df1, df2)
        assert result.loc['(Total)', 'name'] == True
        assert result.loc['(Total)', 'dtype'] == True
        assert result.loc['(Total)', 'nnan'] == True
        assert result.loc['(Total)', 'nan_pat'] == True
        assert result.loc['(Total)', 'content'] == False
        assert result.loc['(Total)', 'sort'] == False
        assert result.loc['(Total)', 'eq'] == False

    def test_compare_dataframes_different_sort(self):
        # Teste, ob compare_dataframes mit unterschiedlicher Sortierung korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84).sort_values('first_name', ascending=False)
        result = compare_dataframes(df1, df2)
        assert result.loc['(Total)', 'name'] == True
        assert result.loc['(Total)', 'dtype'] == True
        assert result.loc['(Total)', 'nnan'] == True
        assert result.loc['(Total)', 'nan_pat'] == False
        assert result.loc['(Total)', 'content'] == True
        assert result.loc['(Total)', 'sort'] == False
        assert result.loc['(Total)', 'eq'] == True

    def test_compare_dataframes_different_index(self):
        # Teste, ob compare_dataframes mit unterschiedlichen Indexen korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84).set_index('first_name')
        result = compare_dataframes(df1, df2)
        assert result.loc['first_name', 'name'] == 'left_only'
        assert result.loc['age', 'content'] == True
        assert result.loc['age', 'eq'] == False


    def test_compare_dataframes_left_only(self):
        # Teste, ob compare_dataframes mit zusätzlicher Spalte links korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84).drop('first_name', axis=1)
        result = compare_dataframes(df1, df2)
        assert result.loc['first_name', 'name'] == 'left_only'
        assert result.loc['age', 'dtype'] == True
        assert result.loc['age', 'sort'] == True

    def test_compare_dataframes_right_only(self):
        # Teste, ob compare_dataframes mit zusätzlicher Spalte rechts korrekt funktioniert
        df1 = people(size=10, seed=84).drop('first_name', axis=1)
        df2 = people(size=10, seed=84)
        result = compare_dataframes(df1, df2)
        assert result.loc['first_name', 'name'] == 'right_only'
        assert result.loc['age', 'dtype'] == True
        assert result.loc['age', 'sort'] == True

    def test_compare_dataframes_with_none(self):
        # Teste, ob compare_dataframes mit None korrekt funktioniert
        df1 = people(size=10, seed=84)
        with pytest.raises(ValueError):
            compare_dataframes(df1, None)
        with pytest.raises(ValueError):
            compare_dataframes(None, df1)

    def test_compare_dataframes_with_wrong_type(self):
        # Teste, ob compare_dataframes mit falschem Typ eine Exception wirft
        df1 = people(size=10, seed=84)
        with pytest.raises(ValueError):
            compare_dataframes(df1, [1, 2, 3])
        with pytest.raises(ValueError):
            compare_dataframes([1, 2, 3], df1)

    def test_compare_dataframes_output_format(self):
        # Teste, ob compare_dataframes die verschiedenen Ausgabeformate korrekt liefert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84)
        result_df = compare_dataframes(df1, df2, format='df')
        assert isinstance(result_df, pd.DataFrame)
        result_series = compare_dataframes(df1, df2, format='s')
        assert isinstance(result_series, pd.Series)
        result_dict = compare_dataframes(df1, df2, format='d')
        assert isinstance(result_dict, dict)
        result_bool = compare_dataframes(df1, df2, format='b')
        assert isinstance(result_bool, bool) or isinstance(result_bool, np.bool_)


    def test_compare_dataframes_different_index_and_sort(self):
        # Teste, ob compare_dataframes mit unterschiedlichen Indexen und unterschiedlicher Sortierung korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84).set_index('first_name').sort_index(ascending=False)
        result = compare_dataframes(df1, df2)
        assert result.loc['first_name', 'name'] == 'left_only'
        assert result.loc['age', 'content'] == True
        assert result.loc['age', 'eq'] == False
        assert result.loc['age', 'sort'] == False

    def test_compare_dataframes_different_index_and_content(self):
        # Teste, ob compare_dataframes mit unterschiedlichen Indexen und unterschiedlichem Inhalt korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84).set_index('first_name')
        df2.loc['Anna', 'age'] = 99
        result = compare_dataframes(df1, df2)
        assert result.loc['first_name', 'name'] == 'left_only'
        assert result.loc['age', 'content'] == False
        assert result.loc['age', 'eq'] == False

    def test_compare_dataframes_different_index_and_content_and_sort(self):
        # Teste, ob compare_dataframes mit unterschiedlichen Indexen, unterschiedlichem Inhalt und unterschiedlicher Sortierung korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84).set_index('first_name').sort_index(ascending=False)
        df2.loc['Anna', 'age'] = 99
        result = compare_dataframes(df1, df2)
        assert result.loc['first_name', 'name'] == 'left_only'
        assert result.loc['age', 'content'] == False
        assert result.loc['age', 'eq'] == False
        assert result.loc['age', 'sort'] == False

    def test_compare_dataframes_different_index_and_nnan(self):
        # Teste, ob compare_dataframes mit unterschiedlichen Indexen und unterschiedlicher Anzahl von NaNs korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84).set_index('first_name')
        df2.loc['Anna', 'birthplace'] = None
        result = compare_dataframes(df1, df2)
        assert result.loc['first_name', 'name'] == 'left_only'
        assert result.loc['birthplace', 'nnan'] == False
        assert result.loc['birthplace', 'content'] == False
        assert result.loc['birthplace', 'eq'] == False

    def test_compare_dataframes_different_index_and_dtype(self):
        # Teste, ob compare_dataframes mit unterschiedlichen Indexen und unterschiedlichen Datentypen korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84).set_index('first_name').astype({'age': 'float64'})
        result = compare_dataframes(df1, df2)
        assert result.loc['first_name', 'name'] == 'left_only'
        assert result.loc['age', 'dtype'] == False
        assert result.loc['age', 'content'] == True
        assert result.loc['age', 'eq'] == False

    def test_compare_dataframes_different_index_and_duplicates(self):
        # Teste, ob compare_dataframes mit unterschiedlichen Indexen und Duplikaten korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84).set_index('first_name')
        df2 = pd.concat([df2, df2.iloc[[0]]])
        result = compare_dataframes(df1, df2)
        assert result.loc['first_name', 'name'] == 'left_only'
        assert result.loc['age', 'content'] == False
        assert result.loc['age', 'eq'] == False

    def test_compare_dataframes_different_index_and_duplicates_and_content(self):
        # Teste, ob compare_dataframes mit unterschiedlichen Indexen, Duplikaten und unterschiedlichem Inhalt korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84).set_index('first_name')
        df2 = pd.concat([df2, df2.iloc[[0]]])
        df2.loc['Anna', 'age'] = 99
        result = compare_dataframes(df1, df2)
        assert result.loc['first_name', 'name'] == 'left_only'
        assert result.loc['age', 'content'] == False
        assert result.loc['age', 'eq'] == False

    def test_compare_dataframes_different_index_and_duplicates_and_nnan(self):
        # Teste, ob compare_dataframes mit unterschiedlichen Indexen, Duplikaten und unterschiedlicher Anzahl von NaNs korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84).set_index('first_name')
        df2 = pd.concat([df2, df2.iloc[[0]]])
        df2.loc['Anna', 'birthplace'] = None
        result = compare_dataframes(df1, df2)
        assert result.loc['first_name', 'name'] == 'left_only'
        assert result.loc['birthplace', 'nnan'] == False
        assert result.loc['birthplace', 'content'] == False
        assert result.loc['birthplace', 'eq'] == False

    def test_compare_datatype_content_decimals(self):

        df = generate_random_data_tcdt()
        result = change_datatype(df)

        comparison = compare_dataframes(df, result, decimals=None)
        assert comparison.loc['float_summe', 'content'] == False
        assert comparison.loc['float_nan', 'content'] == True

        comparison = compare_dataframes(df, result, decimals=2)
        assert comparison.loc['float_summe', 'content'] == True
        assert comparison.loc['float_nan', 'content'] == True

        comparison = compare_dataframes(df, result, decimals=3)
        assert comparison.loc['float_summe', 'content'] == True
        assert comparison.loc['float_nan', 'content'] == True

        comparison = compare_dataframes(df, result, decimals=4)
        assert comparison.loc['float_summe', 'content'] == False

        comparison = compare_dataframes(df, result, decimals=5)
        assert comparison.loc['float_summe', 'content'] == False

    def test_compare_datatype_content_decimals_sort3(self):
        # Teste, ob compare_dataframes mit change_datatype und sort korrekt funktioniert
        df = generate_random_data_tcdt()
        result2 = change_datatype(df)

        comparison = compare_dataframes(df, result2, decimals=None)
        assert comparison.loc['int_grob', 'sort'] == True
        assert comparison.loc['float_summe', 'sort'] == True
        assert comparison.loc['float_nan', 'sort'] == True
        assert comparison.loc['(Total)', 'sort'] == True

        comparison = compare_dataframes(df, result2, decimals=2)
        assert comparison.loc['int_grob', 'sort'] == True
        assert comparison.loc['float_summe', 'sort'] == True
        assert comparison.loc['float_nan', 'sort'] == True
        assert comparison.loc['(Total)', 'sort'] == True

        comparison = compare_dataframes(df, result2, decimals=3)
        assert comparison.loc['int_grob', 'sort'] == True
        assert comparison.loc['float_summe', 'sort'] == True
        assert comparison.loc['float_nan', 'sort'] == True
        assert comparison.loc['(Total)', 'sort'] == True

        comparison = compare_dataframes(df, result2, decimals=4)
        assert comparison.loc['int_grob', 'sort'] == True
        assert comparison.loc['float_summe', 'sort'] == True
        assert comparison.loc['float_nan', 'sort'] == True
        assert comparison.loc['(Total)', 'sort'] == True

        comparison = compare_dataframes(df, result2, decimals=5)
        assert comparison.loc['int_grob', 'sort'] == True
        assert comparison.loc['float_summe', 'sort'] == True
        assert comparison.loc['float_nan', 'sort'] == True
        assert comparison.loc['(Total)', 'sort'] == True



