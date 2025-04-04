import pytest
import pandas as pd
from pandasklar.compare import get_different_rows
from pandasklar.content import people, random_numbers

#@pytest.mark.jetzt # pytest -m jetzt -x
class TestCompareGetDifferentRowsIdentical:

    def test_get_different_rows_identical(self):
        # Teste, ob get_different_rows mit identischen DataFrames und use_index=False korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84)
        result = get_different_rows(df1, df2, use_index=True)
        assert result.empty
        result = get_different_rows(df2, df1, use_index=True)
        assert result.empty
        result = get_different_rows(df1, df2, use_index=False)
        assert result.empty
        result = get_different_rows(df2, df1, use_index=False)
        assert result.empty

    def test_get_different_rows_identical_sort(self):
        # Teste, ob get_different_rows mit identischen DataFrames und use_index=False korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84)
        df2 = df2.sort_values('secret')
        result = get_different_rows(df1, df2, use_index=False)
        assert result.empty
        result = get_different_rows(df2, df1, use_index=False)
        assert result.empty
        result = get_different_rows(df1, df2, use_index=True)
        assert result.empty
        result = get_different_rows(df2, df1, use_index=True)
        assert result.empty

#######################################################################
# use_index=True
#######################################################################


class TestCompareGetDifferentRowsFalse:

    def test_get_different_rows_different_content_index(self):
        # Teste, ob get_different_rows mit unterschiedlichem Inhalt und use_index=True korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84)
        df2.loc[3, 'first_name'] = 'Test'
        result = get_different_rows(df1, df2, use_index=True)
        assert result.shape == (2, 7)
        assert result[result['_merge'] == 'left_only'].shape[0] == 1
        assert result[result['_merge'] == 'right_only'].shape[0] == 1
        result = get_different_rows(df2, df1, use_index=True)
        assert result.shape == (2, 7)
        assert result[result['_merge'] == 'left_only'].shape[0] == 1
        assert result[result['_merge'] == 'right_only'].shape[0] == 1

    def test_get_different_rows_columns_index(self):
        # Teste, ob get_different_rows mit unterschiedlichen Spalten und use_index=True korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84).drop('age', axis=1)
        result = get_different_rows(df1, df2, use_index=True)
        assert result.empty
        result = get_different_rows(df2, df1, use_index=True)
        assert result.empty

    def test_get_different_rows_columns_content_index(self):
        # Teste, ob get_different_rows mit unterschiedlichen Spalten und use_index=True korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84).drop('age', axis=1)
        df2.loc[3, 'first_name'] = 'Test'
        result = get_different_rows(df1, df2, use_index=True)
        assert result.shape == (2, 6)
        assert result[result['_merge'] == 'left_only'].shape[0] == 1
        assert result[result['_merge'] == 'right_only'].shape[0] == 1
        result = get_different_rows(df2, df1, use_index=True)
        assert result.shape == (2, 6)
        assert result[result['_merge'] == 'left_only'].shape[0] == 1
        assert result[result['_merge'] == 'right_only'].shape[0] == 1



    def test_get_different_rows_different_content_sort_index(self):
        # Teste, ob get_different_rows mit unterschiedlichem Inhalt und use_index=True korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84)
        df2.loc[3, 'first_name'] = 'Test'
        df2 = df2.sort_values('secret')
        result = get_different_rows(df1, df2, use_index=True)
        assert result.shape == (2, 7)
        assert result[result['_merge'] == 'left_only'].shape[0] == 1
        assert result[result['_merge'] == 'right_only'].shape[0] == 1
        result = get_different_rows(df2, df1, use_index=True)
        assert result.shape == (2, 7)
        assert result[result['_merge'] == 'left_only'].shape[0] == 1
        assert result[result['_merge'] == 'right_only'].shape[0] == 1

    def test_get_different_rows_columns_sort_index(self):
        # Teste, ob get_different_rows mit unterschiedlichen Spalten und use_index=True korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84).drop('age', axis=1)
        df2 = df2.sort_values('secret')
        result = get_different_rows(df1, df2, use_index=True)
        assert result.empty
        result = get_different_rows(df2, df1, use_index=True)
        assert result.empty

    def test_get_different_rows_columns_content_sort_index(self):
        # Teste, ob get_different_rows mit unterschiedlichen Spalten und use_index=True korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84).drop('age', axis=1)
        df2.loc[3, 'first_name'] = 'Test'
        df2 = df2.sort_values('secret')
        result = get_different_rows(df1, df2, use_index=True)
        assert result.shape == (2, 6)
        assert result[result['_merge'] == 'left_only'].shape[0] == 1
        assert result[result['_merge'] == 'right_only'].shape[0] == 1
        result = get_different_rows(df2, df1, use_index=True)
        assert result.shape == (2, 6)
        assert result[result['_merge'] == 'left_only'].shape[0] == 1
        assert result[result['_merge'] == 'right_only'].shape[0] == 1

    def test_get_different_rows_different_rows_index(self):
        # Teste, ob get_different_rows mit unterschiedlicher Anzahl von Zeilen und use_index=True korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = df1.copy().drop(3)
        result = get_different_rows(df1, df2, use_index=True)
        assert result.shape == (1, 7)
        assert result[result['_merge'] == 'left_only'].shape[0] == 1
        result = get_different_rows(df2, df1, use_index=True)
        assert result.shape == (1, 7)
        assert result[result['_merge'] == 'right_only'].shape[0] == 1

#######################################################################
# use_index=False
#######################################################################

class TestCompareGetDifferentRowsTrue:

    def test_get_different_rows_different_content(self):
        # Teste, ob get_different_rows mit unterschiedlichem Inhalt und use_index=False korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84)
        df2.loc[3, 'first_name'] = 'Test'
        result = get_different_rows(df1, df2, use_index=False)
        assert result.shape == (2, 7)
        assert result[result['_merge'] == 'left_only'].shape[0] == 1
        assert result[result['_merge'] == 'right_only'].shape[0] == 1
        result = get_different_rows(df2, df1, use_index=False)
        assert result.shape == (2, 7)
        assert result[result['_merge'] == 'left_only'].shape[0] == 1
        assert result[result['_merge'] == 'right_only'].shape[0] == 1

    def test_get_different_rows_columns_index(self):
        # Teste, ob get_different_rows mit unterschiedlichen Spalten und use_index=False korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84).drop('age', axis=1)
        result = get_different_rows(df1, df2, use_index=False)
        assert result.empty
        result = get_different_rows(df2, df1, use_index=False)
        assert result.empty

    def test_get_different_rows_columns_content(self):
        # Teste, ob get_different_rows mit unterschiedlichen Spalten und use_index=False korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84).drop('age', axis=1)
        df2.loc[3, 'first_name'] = 'Test'
        result = get_different_rows(df1, df2, use_index=False)
        assert result.shape == (2, 6)
        assert result[result['_merge'] == 'left_only'].shape[0] == 1
        assert result[result['_merge'] == 'right_only'].shape[0] == 1
        result = get_different_rows(df2, df1, use_index=False)
        assert result.shape == (2, 6)
        assert result[result['_merge'] == 'left_only'].shape[0] == 1
        assert result[result['_merge'] == 'right_only'].shape[0] == 1



    def test_get_different_rows_different_content_sort(self):
        # Teste, ob get_different_rows mit unterschiedlichem Inhalt und use_index=False korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84)
        df2.loc[3, 'first_name'] = 'Test'
        df2 = df2.sort_values('secret')
        result = get_different_rows(df1, df2, use_index=False)
        assert result.shape == (2, 7)
        assert result[result['_merge'] == 'left_only'].shape[0] == 1
        assert result[result['_merge'] == 'right_only'].shape[0] == 1
        result = get_different_rows(df2, df1, use_index=False)
        assert result.shape == (2, 7)
        assert result[result['_merge'] == 'left_only'].shape[0] == 1
        assert result[result['_merge'] == 'right_only'].shape[0] == 1

    def test_get_different_rows_columns_sort(self):
        # Teste, ob get_different_rows mit unterschiedlichen Spalten und use_index=False korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84).drop('age', axis=1)
        df2 = df2.sort_values('secret')
        result = get_different_rows(df1, df2, use_index=False)
        assert result.empty
        result = get_different_rows(df2, df1, use_index=False)
        assert result.empty

    def test_get_different_rows_columns_content_sort(self):
        # Teste, ob get_different_rows mit unterschiedlichen Spalten und use_index=False korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84).drop('age', axis=1)
        df2.loc[3, 'first_name'] = 'Test'
        df2 = df2.sort_values('secret')
        result = get_different_rows(df1, df2, use_index=False)
        assert result.shape == (2, 6)
        assert result[result['_merge'] == 'left_only'].shape[0] == 1
        assert result[result['_merge'] == 'right_only'].shape[0] == 1
        result = get_different_rows(df2, df1, use_index=False)
        assert result.shape == (2, 6)
        assert result[result['_merge'] == 'left_only'].shape[0] == 1
        assert result[result['_merge'] == 'right_only'].shape[0] == 1

    def test_get_different_rows_different_rows(self):
        # Teste, ob get_different_rows mit unterschiedlicher Anzahl von Zeilen und use_index=True korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = df1.copy().drop(3)
        result = get_different_rows(df1, df2, use_index=False)
        assert result.shape == (1, 7)
        assert result[result['_merge'] == 'left_only'].shape[0] == 1
        result = get_different_rows(df2, df1, use_index=False)
        assert result.shape == (1, 7)
        assert result[result['_merge'] == 'right_only'].shape[0] == 1


#######################################################################
# CompareAll
#######################################################################

class TestCompareGetDifferentRowsCompareAll:


    def test_get_different_rows_mix0(self):
        # Teste, ob get_different_rows mit unterschiedlicher Anzahl von Zeilen und use_index=True korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = df1.copy().drop(3)
        result1 = get_different_rows(df1, df2, use_index=False)
        assert result1.shape == (1, 7)
        assert result1[result1['_merge'] == 'left_only'].shape[0] == 1
        result2 = get_different_rows(df2, df1, use_index=False)
        assert result2.shape == (1, 7)
        assert result2[result2['_merge'] == 'right_only'].shape[0] == 1
        result3 = get_different_rows(df1, df2, use_index=True)
        assert result3.shape == (1, 7)
        assert result3[result3['_merge'] == 'left_only'].shape[0] == 1
        result4 = get_different_rows(df2, df1, use_index=True)
        assert result4.shape == (1, 7)
        assert result4[result4['_merge'] == 'right_only'].shape[0] == 1

        compare12 = get_different_rows(result1.drop('_merge', axis=1), result2.drop('_merge', axis=1), use_index=False)
        compare34 = get_different_rows(result3.drop('_merge', axis=1), result3.drop('_merge', axis=1), use_index=False)
        compare13 = get_different_rows(result1.drop('_merge', axis=1), result3.drop('_merge', axis=1), use_index=False)
        assert compare12.empty
        assert compare34.empty
        assert compare13.empty


    def test_get_different_rows_mix1(self):
        # Teste, ob get_different_rows mit unterschiedlicher Anzahl von Zeilen und use_index=True korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84)
        df2.loc[3, 'first_name'] = 'Test'
        result1 = get_different_rows(df1, df2, use_index=False)
        assert result1.shape == (2, 7)
        assert result1[result1['_merge'] == 'left_only'].shape[0] == 1
        result2 = get_different_rows(df2, df1, use_index=False)
        assert result2.shape == (2, 7)
        assert result2[result2['_merge'] == 'right_only'].shape[0] == 1
        result3 = get_different_rows(df1, df2, use_index=True)
        assert result3.shape == (2, 7)
        assert result3[result3['_merge'] == 'left_only'].shape[0] == 1
        result4 = get_different_rows(df2, df1, use_index=True)
        assert result4.shape == (2, 7)
        assert result4[result4['_merge'] == 'right_only'].shape[0] == 1

        compare12 = get_different_rows(result1.drop('_merge', axis=1), result2.drop('_merge', axis=1), use_index=False)
        compare34 = get_different_rows(result3.drop('_merge', axis=1), result3.drop('_merge', axis=1), use_index=False)
        compare13 = get_different_rows(result1.drop('_merge', axis=1), result3.drop('_merge', axis=1), use_index=False)
        assert compare12.empty
        assert compare34.empty
        assert compare13.empty


    def test_get_different_rows_mix2(self):
        # Teste, ob get_different_rows mit unterschiedlicher Anzahl von Zeilen und use_index=True korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84)
        df2.loc[3, 'first_name'] = 'Test'
        df2 = df2.sort_values('secret')
        result1 = get_different_rows(df1, df2, use_index=False)
        assert result1.shape == (2, 7)
        assert result1[result1['_merge'] == 'left_only'].shape[0] == 1
        result2 = get_different_rows(df2, df1, use_index=False)
        assert result2.shape == (2, 7)
        assert result2[result2['_merge'] == 'right_only'].shape[0] == 1
        result3 = get_different_rows(df1, df2, use_index=True)
        assert result3.shape == (2, 7)
        assert result3[result3['_merge'] == 'left_only'].shape[0] == 1
        result4 = get_different_rows(df2, df1, use_index=True)
        assert result4.shape == (2, 7)
        assert result4[result4['_merge'] == 'right_only'].shape[0] == 1

        compare12 = get_different_rows(result1.drop('_merge', axis=1), result2.drop('_merge', axis=1), use_index=False)
        compare34 = get_different_rows(result3.drop('_merge', axis=1), result3.drop('_merge', axis=1), use_index=False)
        compare13 = get_different_rows(result1.drop('_merge', axis=1), result3.drop('_merge', axis=1), use_index=False)
        assert compare12.empty
        assert compare34.empty
        assert compare13.empty
