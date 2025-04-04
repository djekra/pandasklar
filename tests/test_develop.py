import pytest
import pandas as pd
import numpy as np
from pandasklar.develop import check_mask, search_str
from pandasklar.content import people, random_numbers

#@pytest.mark.jetzt # pytest -m jetzt -x
class TestCheckMask:

    def test_check_mask_correct_number(self):
        # Teste, ob check_mask korrekt funktioniert, wenn die Anzahl der Zeilen korrekt ist
        df = people(size=100, seed=84)
        mask = df['age'] > 30
        check_mask(df, mask, 30, 70)

    def test_check_mask_too_few_rows_stop(self):
        # Teste, ob check_mask einen Fehler auslöst, wenn zu wenige Zeilen ausgewählt werden und stop=True ist
        df = people(size=100, seed=84)
        mask = df['age'] > 100
        with pytest.raises(Exception):
            check_mask(df, mask, 10, 20)

    def test_check_mask_too_many_rows_stop(self):
        # Teste, ob check_mask einen Fehler auslöst, wenn zu viele Zeilen ausgewählt werden und stop=True ist
        df = people(size=100, seed=84)
        mask = df['age'] > 0
        with pytest.raises(Exception):
            check_mask(df, mask, 10, 20)

    def test_check_mask_too_few_rows_no_stop(self):
        # Teste, ob check_mask eine Fehlermeldung zurückgibt, wenn zu wenige Zeilen ausgewählt werden und stop=False ist
        df = people(size=100, seed=84)
        mask = df['age'] > 100
        result = check_mask(df, mask, 10, 20, stop=False)
        assert 'ERROR' in result

    def test_check_mask_too_many_rows_no_stop(self):
        # Teste, ob check_mask eine Fehlermeldung zurückgibt, wenn zu viele Zeilen ausgewählt werden und stop=False ist
        df = people(size=100, seed=84)
        mask = df['age'] > 0
        result = check_mask(df, mask, 10, 20, stop=False)
        assert 'ERROR' in result

    def test_check_mask_no_expectations(self):
        # Teste, ob check_mask korrekt funktioniert, wenn keine Erwartungen angegeben werden
        df = people(size=100, seed=84)
        mask = df['age'] > 25
        check_mask(df, mask)

    def test_check_mask_only_expectation_circa(self):
        # Teste, ob check_mask korrekt funktioniert, wenn nur expectation_circa angegeben wird
        df = people(size=100, seed=84)
        mask = df['age'] > 30
        check_mask(df, mask, 60)

    def test_check_mask_manually(self):
        # Teste, ob check_mask korrekt funktioniert, wenn manually=True ist
        df = people(size=100, seed=84)
        mask = df['age'] > 25
        check_mask(df, mask, manually=True)

    def test_check_mask_mask_is_not_series(self):
        # Teste, ob check_mask korrekt funktioniert, wenn mask keine Series ist
        df = people(size=100, seed=84)
        mask = np.array([True] * 50 + [False] * 50)
        check_mask(df, mask, 30, 70)

    def test_check_mask_dataframe_with_only_nan(self):
        # Teste, ob check_mask korrekt funktioniert, wenn ein DataFrame mit nur NaN-Werten übergeben wird
        df = pd.DataFrame({'col1': [np.nan, np.nan], 'col2': [np.nan, np.nan]})
        mask = pd.Series([True, False])
        check_mask(df, mask)

    def test_check_mask_empty_mask(self):
        # Teste, ob check_mask korrekt funktioniert, wenn eine leere Maske übergeben wird
        df = people(size=100, seed=84)
        mask = pd.Series([], dtype=bool)
        check_mask(df, mask, 0, 100)

    def test_check_mask_all_true(self):
        # Teste, ob check_mask korrekt funktioniert, wenn eine Maske mit nur True übergeben wird
        df = people(size=100, seed=84)
        mask = pd.Series([True] * 100)
        check_mask(df, mask, 100, 100)

    def test_check_mask_all_false(self):
        # Teste, ob check_mask korrekt funktioniert, wenn eine Maske mit nur False übergeben wird
        df = people(size=100, seed=84)
        mask = pd.Series([False] * 100)
        check_mask(df, mask, 0, 0)

#@pytest.mark.jetzt # pytest -m jetzt -x
class TestSearchStr:

    def test_search_str_existing_string(self):
        # Teste, ob search_str korrekt funktioniert, wenn der Suchstring vorhanden ist
        df = people(size=100, seed=84)
        result = search_str(df, 'Berlin')
        assert result.shape[0] > 0
        assert result['birthplace'].str.contains('Berlin').any()

    def test_search_str_non_existing_string(self):
        # Teste, ob search_str korrekt funktioniert, wenn der Suchstring nicht vorhanden ist
        df = people(size=100, seed=84)
        result = search_str(df, 'Atlantis')
        assert result.shape[0] == 0

    def test_search_str_list_of_strings(self):
        # Teste, ob search_str korrekt funktioniert, wenn eine Liste von Suchstrings übergeben wird
        df = people(size=100, seed=84)
        result = search_str(df, ['Berlin', 'Bremen'])
        assert result.shape[0] > 0
        assert result['birthplace'].str.contains('Berlin|Bremen').any()

    def test_search_str_exclude_columns(self):
        # Teste, ob search_str korrekt funktioniert, wenn Spalten ausgeschlossen werden
        df = people(size=100, seed=84)
        result = search_str(df, 'Berlin', without='birthplace')
        assert result.shape[0] == 0

    def test_search_str_empty_string(self):
        # Teste, ob search_str korrekt funktioniert, wenn ein leerer String übergeben wird
        df = people(size=100, seed=84)
        result = search_str(df, '')
        assert result.shape[0] == 0

    def test_search_str_empty_list(self):
        # Teste, ob search_str korrekt funktioniert, wenn eine leere Liste übergeben wird
        df = people(size=100, seed=84)
        result = search_str(df, [])
        assert result.shape[0] == 0

    def test_search_str_no_string(self):
        # Teste, ob search_str korrekt funktioniert, wenn kein String übergeben wird
        df = people(size=100, seed=84)
        with pytest.raises(TypeError):
            search_str(df, None)




















