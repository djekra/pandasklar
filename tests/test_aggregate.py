import pytest
import pandas as pd
from pandasklar.aggregate import agg_numbers

#@pytest.mark.jetzt
class TestAggNumbers:

    def test_agg_numbers_list(self):
        # Teste, ob agg_numbers mit einer Liste korrekt funktioniert
        data = [1, 2, 3, 4, 5]
        result = agg_numbers(data)
        assert result == 15

    def test_agg_numbers_nested_list(self):
        # Teste, ob agg_numbers mit einer verschachtelten Liste korrekt funktioniert
        data = [1, [2, 3], [4, [5]]]
        result = agg_numbers(data)
        assert result == 15

    def test_agg_numbers_series(self):
        # Teste, ob agg_numbers mit einer Series korrekt funktioniert
        data = pd.Series([1, 2, 3, 4, 5])
        result = agg_numbers(data)
        assert result == 15

    def test_agg_numbers_nested_series(self):
        # Teste, ob agg_numbers mit einer verschachtelten Series korrekt funktioniert
        data = pd.Series([1, [2, 3], [4, [5]]])
        result = agg_numbers(data)
        assert result == 15

    def test_agg_numbers_mixed(self):
        # Teste, ob agg_numbers mit gemischten Daten korrekt funktioniert
        data = [1, [2, 3], pd.Series([4, 5])]
        result = agg_numbers(data)
        assert result == 15

    def test_agg_numbers_empty(self):
        # Teste, ob agg_numbers mit leeren Daten korrekt funktioniert
        data = []
        result = agg_numbers(data)
        assert result == 0

    def test_agg_numbers_no_numbers(self):
        # Teste, ob agg_numbers mit Daten ohne Zahlen korrekt funktioniert
        data = ['a', 'b', 'c']
        result = agg_numbers(data)
        assert result == 0

    def test_agg_numbers_mixed_types(self):
        # Teste, ob agg_numbers mit gemischten Datentypen korrekt funktioniert
        data = [1, 'a', 2.5, [3, 'b', 4.5]]
        result = agg_numbers(data)
        assert result == 11.0