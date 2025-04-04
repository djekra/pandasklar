import pytest
import pandas as pd
from pandasklar.compare import compare_series
from pandasklar.content import people, random_numbers



import pytest
import pandas as pd
from pandasklar.compare import compare_series
from pandasklar.content import people, random_numbers


#@pytest.mark.jetzt  # pytest -m jetzt -x
class TestCompareSeries:

    def test_compare_series_empty(self):
        # Teste, ob compare_series mit identischen Series korrekt funktioniert
        s1 = pd.Series()
        s2 = pd.Series()
        result = compare_series(s1, s2)
        assert result['name'] == True
        assert result['dtype'] == True
        assert result['len'] == True
        assert result['nnan'] == True
        assert result['nan_pat'] == True
        assert result['content'] == True
        assert result['sort'] == True
        assert result['eq'] == True

    def test_compare_series_identical(self):
        # Teste, ob compare_series mit identischen Series korrekt funktioniert
        s1 = pd.Series([1, 2, 3], name='test')
        s2 = pd.Series([1, 2, 3], name='test')
        result = compare_series(s1, s2)
        assert result['name'] == True
        assert result['dtype'] == True
        assert result['len'] == True
        assert result['nnan'] == True
        assert result['nan_pat'] == True
        assert result['content'] == True
        assert result['sort'] == True
        assert result['eq'] == True

    def test_compare_series_identical_nan(self):
        # Teste, ob compare_series mit identischen Series korrekt funktioniert
        s1 = pd.Series([1, 2, None, 3], name='test')
        s2 = pd.Series([1, 2, None, 3], name='test')
        result = compare_series(s1, s2)
        assert result['name'] == True
        assert result['dtype'] == True
        assert result['len'] == True
        assert result['nnan'] == True
        assert result['nan_pat'] == True
        assert result['content'] == True
        assert result['sort'] == True
        assert result['eq'] == True

    def test_compare_series_different_name(self):
        # Teste, ob compare_series mit unterschiedlichen Namen korrekt funktioniert
        s1 = pd.Series([1, 2, 3], name='test1')
        s2 = pd.Series([1, 2, 3], name='test2')
        result = compare_series(s1, s2)
        assert result['name'] == False
        assert result['dtype'] == True
        assert result['len'] == True
        assert result['nnan'] == True
        assert result['nan_pat'] == True
        assert result['content'] == True
        assert result['sort'] == True
        assert result['eq'] == True

    def test_compare_series_different_dtype(self):
        # Teste, ob compare_series mit unterschiedlichen Datentypen korrekt funktioniert
        s1 = pd.Series([1, 2, 3], name='test')
        s2 = pd.Series([1.0, 2.0, 3.0], name='test')
        result = compare_series(s1, s2)
        assert result['name'] == True
        assert result['dtype'] == False
        assert result['len'] == True
        assert result['nnan'] == True
        assert result['nan_pat'] == True
        assert result['content'] == True
        assert result['sort'] == True
        assert result['eq'] == True

        s1 = pd.Series([1, 2, 3], name='test', dtype='int64')
        s2 = pd.Series(['1', '2', '3'], name='test', dtype='string')
        result = compare_series(s1, s2)
        assert result['name'] == True
        assert result['dtype'] == False
        assert result['len'] == True
        assert result['nnan'] == True
        assert result['nan_pat'] == True
        assert result['content'] == True
        assert result['sort'] == True
        assert result['eq'] == False

    def test_compare_series_different_length(self):
        # Teste, ob compare_series mit unterschiedlicher Länge korrekt funktioniert
        s1 = pd.Series([1, 2, 3], name='test')
        s2 = pd.Series([1, 2, 3, 4], name='test')
        result = compare_series(s1, s2)
        assert result['name'] == True
        assert result['dtype'] == True
        assert result['len'] == False
        assert result['nnan'] == True
        assert result['nan_pat'] == False
        assert result['content'] == False
        assert result['sort'] == False
        assert result['eq'] == False

    def test_compare_series_different_nnan(self):
        # Teste, ob compare_series mit unterschiedlicher Anzahl von NaNs korrekt funktioniert
        s1 = pd.Series([1, 2, None], name='test')
        s2 = pd.Series([1, None, None], name='test')
        result = compare_series(s1, s2)
        assert result['name'] == True
        assert result['dtype'] == True
        assert result['len'] == True
        assert result['nnan'] == False
        assert result['nan_pat'] == False
        assert result['content'] == False
        assert result['sort'] == False
        assert result['eq'] == False

    def test_compare_series_different_content(self):
        # Teste, ob compare_series mit unterschiedlichem Inhalt korrekt funktioniert
        s1 = pd.Series([1, 2, 3], name='test')
        s2 = pd.Series([1, 2, 4], name='test')
        result = compare_series(s1, s2)
        assert result['name'] == True
        assert result['dtype'] == True
        assert result['len'] == True
        assert result['nnan'] == True
        assert result['nan_pat'] == True
        assert result['content'] == False
        assert result['sort'] == True
        assert result['eq'] == False

    def test_compare_series_different_sort(self):
        # Teste, ob compare_series mit unterschiedlicher Sortierung korrekt funktioniert
        s1 = pd.Series([1, 2, 3], name='test')
        s2 = pd.Series([3, 2, 1], name='test')
        result = compare_series(s1, s2)
        assert result['name'] == True
        assert result['dtype'] == True
        assert result['len'] == True
        assert result['nnan'] == True
        assert result['nan_pat'] == True
        assert result['content'] == True
        assert result['sort'] == False
        assert result['eq'] == False

    def test_compare_series_different_eq(self):
        # Teste, ob compare_series mit unterschiedlichen Index-Daten-Beziehungen korrekt funktioniert
        s1 = pd.Series([1, 2, 3], name='test')
        s2 = pd.Series([1, 2, 3], name='test', index=[3, 2, 1])
        result = compare_series(s1, s2)
        assert result['name'] == True
        assert result['dtype'] == True
        assert result['len'] == True
        assert result['nnan'] == True
        assert result['nan_pat'] == True
        assert result['content'] == True
        assert result['sort'] == True
        assert result['eq'] == False


    def test_compare_series_eq_true_sort_false(self):
        # Teste, ob compare_series korrekt funktioniert, wenn eq == True und sort == False
        s1 = pd.Series([1, 2, 3], index=[0, 1, 2], name='test')
        s2 = pd.Series([3, 1, 2], index=[2, 0, 1], name='test')
        result = compare_series(s1, s2)
        assert result['name'] == True
        assert result['dtype'] == True
        assert result['len'] == True
        assert result['nnan'] == True
        assert result['nan_pat'] == True
        assert result['content'] == True
        assert result['sort'] == False
        assert result['eq'] == True

    def test_compare_series_eq_true_sort_false_nan(self):
        # Teste, ob compare_series korrekt funktioniert, wenn eq == True und sort == False
        s1 = pd.Series([1, None, 3], index=[0, 1, 2], name='test')
        s2 = pd.Series([3, 1, None], index=[2, 0, 1], name='test')
        result = compare_series(s1, s2)
        assert result['name'] == True
        assert result['dtype'] == True
        assert result['len'] == True
        assert result['nnan'] == True
        assert result['nan_pat'] == False
        assert result['content'] == True
        assert result['sort'] == False
        assert result['eq'] == True


    def test_compare_series_with_none_left(self):
        # Teste, ob compare_series mit None als linke Series korrekt funktioniert
        s2 = pd.Series([1, 2, 3], name='test')
        result = compare_series(None, s2)
        assert result['name'] == 'right_only'
        assert result['dtype'] == None
        assert result['len'] == None
        assert result['nnan'] == None
        assert result['nan_pat'] == None
        assert result['content'] == False
        assert result['sort'] == None
        assert result['eq'] == False

    def test_compare_series_with_none_right(self):
        # Teste, ob compare_series mit None als rechte Series korrekt funktioniert
        s1 = pd.Series([1, 2, 3], name='test')
        result = compare_series(s1, None)
        assert result['name'] == 'left_only'
        assert result['dtype'] == None
        assert result['len'] == None
        assert result['nnan'] == None
        assert result['nan_pat'] == None
        assert result['content'] == False
        assert result['sort'] == None
        assert result['eq'] == False

    def test_compare_series_with_wrong_type(self):
        # Teste, ob compare_series mit falschem Typ eine Exception wirft
        s1 = pd.Series([1, 2, 3], name='test')
        with pytest.raises(ValueError):
            compare_series(s1, [1, 2, 3])
        with pytest.raises(ValueError):
            compare_series([1, 2, 3], s1)

    def test_compare_series_output_format(self):
        # Teste, ob compare_series die verschiedenen Ausgabeformate korrekt liefert
        s1 = pd.Series([1, 2, 3], name='test')
        s2 = pd.Series([1, 2, 3], name='test')
        result_dict = compare_series(s1, s2, format='dict')
        assert isinstance(result_dict, dict)
        result_series = compare_series(s1, s2, format='series')
        assert isinstance(result_series, pd.Series)
        result_df = compare_series(s1, s2, format='dataframe')
        assert isinstance(result_df, pd.DataFrame)

    def test_compare_series_different_sort_and_index_same_content(self):
        # Teste, ob compare_series mit unterschiedlicher Sortierung und unterschiedlichen Indexen, aber gleichem Inhalt korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84).sort_values('age', ascending=False).reset_index(drop=True)
        s1 = df1.features
        s2 = df2.features
        result = compare_series(s1, s2)
        assert result['name'] == True
        assert result['dtype'] == True
        assert result['len'] == True
        assert result['nnan'] == True
        assert result['nan_pat'] == True
        assert result['content'] == True
        assert result['sort'] == False
        assert result['eq'] == False

    def test_compare_series_different_index_same_content(self):
        # Teste, ob compare_series mit unterschiedlichen Indexen und gleichem Inhalt korrekt funktioniert
        s1 = pd.Series([1, 2, 3], index=[0, 1, 2], name='test')
        s2 = pd.Series([1, 2, 3], index=["a", "b", "c"], name='test')
        result = compare_series(s1, s2)
        assert result['name'] == True
        assert result['dtype'] == True
        assert result['len'] == True
        assert result['nnan'] == True
        assert result['nan_pat'] == True
        assert result['content'] == True
        assert result['sort'] == True
        assert result['eq'] == False

    def test_compare_series_different_index_same_content_age(self):
        # Teste, ob compare_series mit unterschiedlichen Indexen und gleichem Inhalt ohne NaNs korrekt funktioniert
        df1 = people(size=10, seed=84)
        df2 = people(size=10, seed=84).set_index('first_name')
        s1 = df1['age']
        s2 = df2['age']
        result = compare_series(s1, s2)
        assert result['name'] == True
        assert result['dtype'] == True
        assert result['len'] == True
        assert result['nnan'] == True
        assert result['nan_pat'] == True
        assert result['content'] == True
        assert result['sort'] == True
        assert result['eq'] == False

    def test_compare_series_different_index_with_nan(self):
        # Teste, ob compare_series mit unterschiedlichen Indexen und NaNs korrekt funktioniert
        s1 = pd.Series([1, 2, None, 3], index=[0, 1, 2, 3], name='test')
        s2 = pd.Series([1, 2, None, 3], index=["a", "b", "c", "d"], name='test')
        result = compare_series(s1, s2)
        assert result['name'] == True
        assert result['dtype'] == True
        assert result['len'] == True
        assert result['nnan'] == True
        assert result['nan_pat'] == True
        assert result['content'] == True
        assert result['sort'] == True
        assert result['eq'] == False

    def test_compare_series_different_index_with_nan2(self):
        # Teste, ob compare_series mit unterschiedlichen Indexen und NaNs korrekt funktioniert
        s1 = pd.Series([1, 2, None, 3], index=[0, 1, 2, 3], name='test')
        s2 = pd.Series([1, 2, 3, None], index=["a", "b", "c", "d"], name='test')
        result = compare_series(s1, s2)
        assert result['name'] == True
        assert result['dtype'] == True
        assert result['len'] == True
        assert result['nnan'] == True
        assert result['nan_pat'] == False
        assert result['content'] == True
        assert result['sort'] == True
        assert result['eq'] == False

    def test_compare_series_different_index_with_nan_and_different_content(self):
        # Teste, ob compare_series mit unterschiedlichen Indexen, NaNs und unterschiedlichem Inhalt korrekt funktioniert
        s1 = pd.Series([1, 2, None, 3], index=[0, 1, 2, 3], name='test')
        s2 = pd.Series([1, 4, None, 3], index=["a", "b", "c", "d"], name='test')
        result = compare_series(s1, s2)
        assert result['name'] == True
        assert result['dtype'] == True
        assert result['len'] == True
        assert result['nnan'] == True
        assert result['nan_pat'] == True
        assert result['content'] == False
        assert result['sort'] == False
        assert result['eq'] == False


    def test_compare_series_different_index_and_sort(self):
        # Teste, ob compare_series mit unterschiedlichen Indexen und unterschiedlicher Sortierung korrekt funktioniert
        s1 = pd.Series([1, 2, 3], index=[0, 1, 2], name='test')
        s2 = pd.Series([3, 2, 1], index=["a", "b", "c"], name='test')
        result = compare_series(s1, s2)
        assert result['name'] == True
        assert result['dtype'] == True
        assert result['len'] == True
        assert result['nnan'] == True
        assert result['nan_pat'] == True
        assert result['content'] == True
        assert result['sort'] == False
        assert result['eq'] == False

    def test_compare_series_different_index_and_content_and_sort(self):
        # Teste, ob compare_series mit unterschiedlichen Indexen, unterschiedlichem Inhalt und unterschiedlicher Sortierung korrekt funktioniert
        s1 = pd.Series([1, 2, 3], index=[0, 1, 2], name='test')
        s2 = pd.Series([4, 5, 6], index=["a", "b", "c"], name='test')
        result = compare_series(s1, s2)
        assert result['name'] == True
        assert result['dtype'] == True
        assert result['len'] == True
        assert result['nnan'] == True
        assert result['nan_pat'] == True
        assert result['content'] == False
        assert result['sort'] == True
        assert result['eq'] == False

    def test_compare_series_different_index_and_content_and_sort_with_nan(self):
        # Teste, ob compare_series mit unterschiedlichen Indexen, unterschiedlichem Inhalt, unterschiedlicher Sortierung und NaNs korrekt funktioniert
        s1 = pd.Series([1, 2, None], index=[0, 1, 2], name='test')
        s2 = pd.Series([4, None, 6], index=["a", "b", "c"], name='test')
        result = compare_series(s1, s2)
        assert result['name'] == True
        assert result['dtype'] == True
        assert result['len'] == True
        assert result['nnan'] == True
        assert result['nan_pat'] == False
        assert result['content'] == False
        assert result['sort'] == True
        assert result['eq'] == False

    def test_compare_series_different_index_same_content_with_duplicates(self):
        # Teste, ob compare_series mit unterschiedlichen Indexen, gleichem Inhalt und Duplikaten korrekt funktioniert
        s1 = pd.Series([1, 2, 2, 3], index=[0, 1, 2, 3], name='test')
        s2 = pd.Series([2, 3, 1, 2], index=["a", "b", "c", "d"], name='test')
        result = compare_series(s1, s2)
        assert result['name'] == True
        assert result['dtype'] == True
        assert result['len'] == True
        assert result['nnan'] == True
        assert result['nan_pat'] == True
        assert result['content'] == True
        assert result['sort'] == False
        assert result['eq'] == False

    def test_compare_series_different_index_and_content_with_duplicates(self):
        # Teste, ob compare_series mit unterschiedlichen Indexen, unterschiedlichem Inhalt und Duplikaten korrekt funktioniert
        s1 = pd.Series([1, 2, 2, 3], index=[0, 1, 2, 3], name='test')
        s2 = pd.Series([4, 5, 5, 6], index=["a", "b", "c", "d"], name='test')
        result = compare_series(s1, s2)
        assert result['name'] == True
        assert result['dtype'] == True
        assert result['len'] == True
        assert result['nnan'] == True
        assert result['nan_pat'] == True
        assert result['content'] == False
        assert result['sort'] == True
        assert result['eq'] == False

    def test_compare_series_different_index_and_content_with_duplicates_and_nan(self):
        # Teste, ob compare_series mit unterschiedlichen Indexen, unterschiedlichem Inhalt, Duplikaten und NaNs korrekt funktioniert
        s1 = pd.Series([1, 2, 2, None], index=[0, 1, 2, 3], name='test')
        s2 = pd.Series([4, 5, None, 5], index=["a", "b", "c", "d"], name='test')
        result = compare_series(s1, s2)
        assert result['name'] == True
        assert result['dtype'] == True
        assert result['len'] == True
        assert result['nnan'] == True
        assert result['nan_pat'] == False
        assert result['content'] == False
        assert result['sort'] == True
        assert result['eq'] == False

    def test_compare_series_decimals(self):
        # Teste, ob compare_series mit dem Parameter 'decimals' korrekt funktioniert
        s1 = pd.Series([1.000001, 2.000002, 3.000003], name='test')
        s2 = pd.Series([1.000002, 2.000003, 3.000004], name='test')

        # Ohne decimals sollten die Inhalte unterschiedlich sein
        result = compare_series(s1, s2)
        assert result['content'] == False
        assert result['nan_pat'] == True

        # Mit decimals sollten die Inhalte irgendwann als gleich gelten
        result = compare_series(s1, s2, decimals=8)
        assert result['content'] == False
        assert result['nan_pat'] == True

        result = compare_series(s1, s2, decimals=7)
        assert result['content'] == False
        assert result['nan_pat'] == True

        result = compare_series(s1, s2, decimals=6)
        assert result['content'] == False
        assert result['nan_pat'] == True

        result = compare_series(s1, s2, decimals=5)
        assert result['content'] == True
        assert result['nan_pat'] == True

        result = compare_series(s1, s2, decimals=4)
        assert result['content'] == True
        assert result['nan_pat'] == True

        result = compare_series(s1, s2, decimals=3)
        assert result['content'] == True
        assert result['nan_pat'] == True

        result = compare_series(s1, s2, decimals=2)
        assert result['content'] == True
        assert result['nan_pat'] == True

        result = compare_series(s1, s2, decimals=1)
        assert result['content'] == True
        assert result['nan_pat'] == True

        result = compare_series(s1, s2, decimals=0)
        assert result['content'] == True
        assert result['nan_pat'] == True


    def test_compare_series_decimals2(self):
        # Teste, ob compare_series mit dem Parameter 'decimals' korrekt funktioniert
        s1 = pd.Series([1.00005, 2.000005, 3.000004], name='test')
        s2 = pd.Series([1.000004, 2.000004, 3.000005], name='test')

        # Ohne decimals sollten die Inhalte unterschiedlich sein
        result = compare_series(s1, s2)
        assert result['content'] == False

        # Mit decimals sollten die Inhalte irgendwann als gleich gelten
        result = compare_series(s1, s2, decimals=8)
        assert result['content'] == False

        result = compare_series(s1, s2, decimals=7)
        assert result['content'] == False

        result = compare_series(s1, s2, decimals=6)
        assert result['content'] == False

        result = compare_series(s1, s2, decimals=5)
        assert result['content'] == False

        result = compare_series(s1, s2, decimals=4)
        assert result['content'] == False

        result = compare_series(s1, s2, decimals=3)
        assert result['content'] == True

        result = compare_series(s1, s2, decimals=2)
        assert result['content'] == True

        result = compare_series(s1, s2, decimals=1)
        assert result['content'] == True

        result = compare_series(s1, s2, decimals=0)
        assert result['content'] == True

