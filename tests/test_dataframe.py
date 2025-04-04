
# 0.5

import pytest
import pandas as pd
import polars as pl
import numpy as np
from pandasklar.dataframe import dataframe
from bpyth import rtype

#@pytest.mark.jetzt # pytest -m jetzt -x
class TestDataframe:


    ###################################################################################
    # Leere und skalare Eingabeobjekte
    ###################################################################################

    # Leere Eingabeobjekte
    def test_dataframe_without_content(self):
        # Teste, ob dataframe mit None korrekt funktioniert
        datas = [ None, {}, [], set(), list(), (), np.array([], dtype=object), pd.Series(), pd.Series(name='Q'), pl.Series(), pl.Series().rename('Q') ]
        for i, data in enumerate(datas):
            result = dataframe(data, framework='pandas')
            assert isinstance(result, pd.DataFrame)
            assert rtype(result) == ('DataFrame',)
            assert result.shape == (0,0)
            assert result.index.name is None
            assert result.columns.tolist() == []

            result = dataframe(data, framework='polars')
            assert isinstance(result, pl.DataFrame)
            assert rtype(result) == ('DataFrame',)
            assert result.shape == (0,0)



    # Skalare Eingabeobjekte
    def test_dataframe_with_skalar_content(self, capsys):
        # Teste, ob dataframe mit None korrekt funktioniert
        datas = [ 1, 4.2, 'x', 'hello', True, False ]
        for i, data in enumerate(datas):
            with capsys.disabled():
                #print(f"Teste mit data: {data}")
                result = dataframe(data, framework='pandas')
                assert isinstance(result, pd.DataFrame)
                assert rtype(result)[:2] == ('DataFrame','Series')
                assert result.shape == (1,1)
                assert result.index.name is None

                result = dataframe(data, framework='polars')
                assert isinstance(result, pl.DataFrame)
                assert rtype(result)[:2] == ('DataFrame','Series')
                assert result.shape == (1,1)


    ###################################################################################
    # 1-dimensionale Eingabeobjekte
    ###################################################################################

    # Eindimensionale Eingabeobjekte -> 1 Zeile
    def test_dataframe_1dim_row(self, capsys):
        # Teste, ob dataframe mit None korrekt funktioniert
        datas = [ [1, 2, 3],
                  (1, 2, 3),
                  {'A': 1, 'B': 2, 'C': 3},
                ]
        for i, data in enumerate(datas):
            with capsys.disabled():
                #print(f"Teste mit data: {data}")
                result = dataframe(data, framework='pandas')
                assert isinstance(result, pd.DataFrame)
                assert rtype(result) == ('DataFrame', 'Series', 'int')
                assert result.shape == (1,3)
                assert result.index.name is None
                assert result.iat[0, 0] == 1
                assert result.iat[0, 1] == 2
                assert result.iat[0, 2] == 3
                assert result.columns[0] == 'A'
                assert result.columns[1] == 'B'
                assert result.columns[2] == 'C'

                result = dataframe(data, framework='polars')
                assert isinstance(result, pl.DataFrame)
                assert rtype(result) == ('DataFrame', 'Series', 'int')
                assert result.shape == (1,3)
                assert result[0, 0] == 1
                assert result[0, 1] == 2
                assert result[0, 2] == 3
                assert result.columns[0] == 'A'
                assert result.columns[1] == 'B'
                assert result.columns[2] == 'C'


    # Eindimensionale Eingabeobjekte -> 1 Spalte
    def test_dataframe_1dim_col_unnamed(self, capsys):

        d0 = {'A': [1, 2, 3]}
        d1 = np.array([1, 2, 3])
        d2 = pd.Series([1, 2, 3])
        d3 = pl.Series([1, 2, 3])
        datas = [ d0, d1, d2, d3 ]

        for i, data in enumerate(datas):
            with capsys.disabled():
                #print(f"test_dataframe_1dim_col_unnamed pandas {i}")
                result = dataframe(data, framework='pandas')
                assert isinstance(result, pd.DataFrame)
                assert rtype(result) == ('DataFrame', 'Series', 'int')
                assert result.shape == (3,1)
                assert result.index.name is None
                assert result.iat[0, 0] == 1
                assert result.iat[1, 0] == 2
                assert result.iat[2, 0] == 3
                assert result.columns[0] == 'A'

                #print(f"test_dataframe_1dim_col_unnamed polars {i}")
                result = dataframe(data, framework='polars')
                assert isinstance(result, pl.DataFrame)
                assert rtype(result) == ('DataFrame', 'Series', 'int')
                assert result.shape == (3,1)
                assert result[0, 0] == 1
                assert result[1, 0] == 2
                assert result[2, 0] == 3
                assert result.columns[0] == 'A'


    # Eindimensionale Eingabeobjekte -> 1 Spalte
    def test_dataframe_1dim_col_named(self, capsys):

        d0 = pd.Series(['ale', 'bola', 'cul'])
        d0.name = '1'
        d1 = pd.Series(['ale', 'bola', 'cul'])
        d1.name = 1
        d2 = pl.Series(['ale', 'bola', 'cul']).rename('1')
        d3 = {'1': ['ale', 'bola', 'cul']}
        d4 = { 1: ['ale', 'bola', 'cul']}
        datas = [ d0, d1, d2, d3, d4 ]

        for i, data in enumerate(datas):
            with capsys.disabled():
                #print(f"test_dataframe_1dim_col_named pandas {i}")
                result = dataframe(data, framework='pandas')
                assert isinstance(result, pd.DataFrame)
                assert rtype(result) == ('DataFrame', 'Series', 'str')
                assert result.shape == (3,1)
                assert result.index.name is None
                assert result.iat[0, 0] == 'ale'
                assert result.iat[1, 0] == 'bola'
                assert result.iat[2, 0] == 'cul'
                assert result.columns[0] in [1, '1']

                #print(f"test_dataframe_1dim_col_named polars {i}")
                result = dataframe(data, framework='polars')
                assert isinstance(result, pl.DataFrame)
                assert rtype(result) == ('DataFrame', 'Series', 'str')
                assert result.shape == (3,1)
                assert result[0, 0] == 'ale'
                assert result[1, 0] == 'bola'
                assert result[2, 0] == 'cul'
                assert result.columns[0] == '1'






    ###################################################################################
    # 2-dimensionale Eingabeobjekte
    ###################################################################################

    def test_dataframe_2dim_named(self, capsys):

        d0 = {'AA':  [ 2,   4,   8,  ],
              'BB':  ['z', 'v', 'a', ] }
        d1 = {'AA': np.array([2, 4, 8]),
              'BB': np.array(['z', 'v', 'a'])}
        d2 = {'AA':  ( 2,   4,   8,  ),
              'BB':  ('z', 'v', 'a', ) }

        d3 = [ {'AA': 2, 'BB': 'z'},
               {'AA': 4, 'BB': 'v'},
               {'AA': 8, 'BB': 'a'} ]
        d4 = ( {'AA': 2, 'BB': 'z'},
               {'AA': 4, 'BB': 'v'},
               {'AA': 8, 'BB': 'a'} )

        s0 = pd.Series([2, 4, 8])
        s0.name = 'AA'
        s1 = pd.Series(['z', 'v', 'a'])
        s1.name = 'BB'

        s2 = pl.Series([2, 4, 8]).rename('AA')
        s3 = pl.Series(['z', 'v', 'a']).rename('BB')

        d5 = [s0, s1]
        d6 = (s0, s1)
        d7 = [s2, s3]
        d8 = (s2, s3)
        datas = [ d0, d1, d2, d3, d4, d5, d6, d7, d8 ]

        for i, data in enumerate(datas):
            with capsys.disabled():
                # print(f"test_dataframe_2dim_named pandas {i}")
                result = dataframe(data, framework='pandas')
                assert isinstance(result, pd.DataFrame)
                assert result.shape == (3,2)
                assert result.index.name is None
                assert result.iat[0, 1] == 'z'
                assert result.iat[2, 0] == 8
                assert result.columns[0] == 'AA'
                assert result.columns[1] == 'BB'

                # print(f"test_dataframe_2dim_named polars {i}")
                result = dataframe(data, framework='polars')
                assert isinstance(result, pl.DataFrame)
                assert result.shape == (3,2)
                assert result[0, 1] == 'z'
                assert result[2, 0] == 8
                assert result.columns[0] == 'AA'
                assert result.columns[1] == 'BB'

    def test_dataframe_2dim_named_2(self, capsys):

        d0 = { 0:  [ 2,   4,   8,  ],
               1:  ['z', 'v', 'a', ] }
        d1 = { 0: np.array([2, 4, 8]),
               1: np.array(['z', 'v', 'a'])}
        d2 = { 0:  ( 2,   4,   8,  ),
               1:  ('z', 'v', 'a', ) }

        d3 = [ {0: 2, 1: 'z'},
               {0: 4, 1: 'v'},
               {0: 8, 1: 'a'} ]
        d4 = ( {0: 2, 1: 'z'},
               {0: 4, 1: 'v'},
               {0: 8, 1: 'a'} )

        s0 = pd.Series([2, 4, 8])
        s0.name = 0
        s1 = pd.Series(['z', 'v', 'a'])
        s1.name = 1

        s2 = pl.Series([2, 4, 8]).rename('0')
        s3 = pl.Series(['z', 'v', 'a']).rename('1')

        d5 = [s0, s1]
        d6 = (s0, s1)
        d7 = [s2, s3]
        d8 = (s2, s3)
        datas = [ d0, d1, d2, d3, d4, d5, d6, d7, d8 ]

        for i, data in enumerate(datas):
            with capsys.disabled():
                #print(f"test_dataframe_2dim_named_2 pandas {i}")
                result = dataframe(data, framework='pandas')
                assert isinstance(result, pd.DataFrame)
                assert result.shape == (3,2)
                assert result.index.name is None
                assert result.iat[0, 1] == 'z'
                assert result.iat[2, 0] == 8
                assert result.columns[0] in [0, '0']
                assert result.columns[1] in [1, '1']

                #print(f"test_dataframe_2dim_named_2 polars {i}")
                result = dataframe(data, framework='polars')
                assert isinstance(result, pl.DataFrame)
                assert result.shape == (3,2)
                assert result[0, 1] == 'z'
                assert result[2, 0] == 8
                assert result.columns[0] == '0'
                assert result.columns[1] == '1'


    def test_dataframe_2dim_unnamed(self, capsys):

        d0 = [  [2,'z'], [4,'v'], [8,'a']  ]
        d1 = [  (2,'z'), (4,'v'), (8,'a')  ]

        d2 = (  [2,4,8],  ['z','v','a']  )
        d3 = (  (2,4,8),  ('z','v','a')  )

        d4 = [pd.Series([2, 4, 8]), pd.Series(['z', 'v', 'a'])]
        d5 = (pl.Series([2, 4, 8]), pl.Series(['z', 'v', 'a']))

        datas = [ d0, d1, d2, d3, d4, d5 ]
        for i, data in enumerate(datas):
            with capsys.disabled():
                #print(f"test_dataframe_2dim_unnamed pandas {i}")
                result = dataframe(data, framework='pandas')
                assert isinstance(result, pd.DataFrame)
                assert result.shape == (3,2)
                assert result.index.name is None
                assert result.iat[0, 1] == 'z'
                assert result.iat[2, 0] == 8
                assert result.columns[0] == 'A'
                assert result.columns[1] == 'B'

                #print(f"test_dataframe_2dim_unnamed polars {i}")
                result = dataframe(data, framework='polars')
                assert isinstance(result, pl.DataFrame)
                assert result.shape == (3,2)
                assert result[0, 1] == 'z'
                assert result[2, 0] == 8
                assert result.columns[0] == 'A'
                assert result.columns[1] == 'B'




    ###################################################################################
    # Mehrdimensionale Eingabeobjekte
    ###################################################################################

    def test_dataframe_with_dict_of_list_of_list_of_ints(self):
        # Teste, ob dataframe mit einem Dictionary mit einer Liste korrekt funktioniert
        data = {
            'a': [ [1,1], [2,2], [3,3] ],
            'b': [[1, 1], [2, 2], [3, 3]],
                }
        result = dataframe(data)
        assert rtype(data) == ('dict', 'list', 'list', 'int')
        assert rtype(result) == ('DataFrame', 'Series', 'list', 'int')
        assert result.shape == (3, 2)
        assert result.index.name is None
        assert result.iloc[0, 0] == [1,1]


    def test_dataframe_with_list_of_list_of_list_of_ints(self):
        # Teste, ob dataframe mit einer Liste von Listen von Listen von Integern korrekt funktioniert
        data = [
            [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
            [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
            [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
            [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
            [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
        ]
        result = dataframe(data)
        assert rtype(result) == ('DataFrame', 'Series', 'list', 'int')
        assert result.shape == (5, 3)
        assert result.index.name is None
        assert result.iloc[0, 0] == [0, 0, 0]
        assert result.iloc[4, 2] == [2, 2, 2]


    def test_dataframe_with_tuple_of_lists(self):
        # Teste, ob dataframe mit einem Tupel von Listen korrekt funktioniert
        Number = [1, 2, 3, 4, 6]
        L1 = ['a', 'v', 'vvvv', 'e', 'Q']
        L2 = [100, 55, 315, 68, 23]
        L3 = ['18%', '105%', '56%', '12%', '4%']
        inp = (Number, L1, L2, L3)
        result = dataframe(inp)
        assert rtype(result) == ('DataFrame', 'Series', 'int')
        assert result.shape == (5, 4)
        assert result.index.name is None
        assert result.iloc[0, 0] == 1
        assert result.iloc[4, 3] == '4%'


    def test_dataframe_with_dict_of_lists(self):
        # Teste, ob dataframe mit einem Dictionary von Listen korrekt funktioniert
        inp = {'AA': [1, 1, 1, 1],
               'BB': [2, 4, 8, 16],
               'CC': [3, 6, 9, 12],
               'DD': [4, 4, 4, 4],
               'EE': [5, 10, 15, 20],
               }
        result = dataframe(inp)
        assert rtype(result) == ('DataFrame', 'Series', 'int')
        assert result.shape == (4, 5)
        assert result.index.name is None
        assert result.iloc[0, 0] == 1
        assert result.iloc[3, 4] == 20


    def test_dataframe_with_dict_of_tuples(self):
        # Teste, ob dataframe mit einem Dictionary von Tupeln korrekt funktioniert
        data = {'AA': (1, 1, 1, 1),
                'BB': (2, 4, 8, 16),
                'CC': (3, 6, 9, 12),
                'DD': (4, 4, 4, 4),
                'EE': (5, 10, 15, 20),
                }
        result = dataframe(data)
        assert rtype(result) == ('DataFrame', 'Series', 'int')
        assert result.shape == (4, 5)
        assert result.index.name is None
        assert result.iloc[0, 0] == 1
        assert result.iloc[3, 4] == 20


    def test_dataframe_with_list_of_dicts(self):
        # Teste, ob dataframe mit einer Liste von Dictionaries korrekt funktioniert
        data = [{'AA': 1, 'BB': 2, 'CC': 3},
                {'AA': 1, 'BB': 4, 'CC': 9},
                {'AA': 1, 'BB': 16, 'CC': 81}]
        result = dataframe(data)
        assert rtype(result) == ('DataFrame', 'Series', 'int')
        assert result.shape == (3, 3)
        assert result.index.name is None
        assert result.iloc[0, 0] == 1
        assert result.iloc[2, 2] == 81


    def test_dataframe_with_dict_of_dicts(self, capsys):
        # Teste, ob dataframe mit einem Dictionary von Dictionaries korrekt funktioniert
        data = {
            0: {'AA': 5, 'BB': 0, 'CC': 3, 'DD': 3},
            1: {'AA': 10, 'BB': 0, 'CC': 3, 'DD': 5},
            2: {'AA': 15, 'BB': 0, 'CC': 7, 'DD': 6}
        }

        with capsys.disabled():
            #print( "test_dataframe_with_dict_of_dicts2 with Pandas")
            result_pandas = dataframe(data, framework='pandas')
            assert isinstance(result_pandas, pd.DataFrame)
            assert rtype(result_pandas) == ('DataFrame', 'Series', 'int')
            assert result_pandas.shape == (4, 3)
            assert result_pandas.index.name is None
            assert result_pandas.iloc[0, 0] == 5
            assert result_pandas.iloc[0, 2] == 15
            assert result_pandas.columns[0] == 0
            assert result_pandas.columns[1] == 1
            assert result_pandas.columns[2] == 2


            #print("test_dataframe_with_dict_of_dicts2 with Polars")
            result_polars = dataframe(data, framework='polars')
            assert isinstance(result_polars, pl.DataFrame)
            assert rtype(result_polars) == ('DataFrame', 'Series', 'int')
            assert result_polars.shape == (4, 3)
            assert result_pandas.iloc[0, 0] == 5
            assert result_pandas.iloc[0, 2] == 15
            assert result_pandas.columns[0] == 0
            assert result_pandas.columns[1] == 1
            assert result_pandas.columns[2] == 2


    def test_dataframe_with_dict_of_mixed_types(self):
        # Teste, ob dataframe mit einem Dictionary von gemischten Typen korrekt funktioniert
        data = {'AA': np.array([-77] * 4, dtype='int32'),
                'BB': pd.Categorical(["test", "train", "test", "train"]),
                'CC': pd.Series(1, index=list(range(4)), dtype='float32'),
                'DD': 'foo',  # test=False !!
                }
        result = dataframe(data)
        assert rtype(result) == ('DataFrame', 'Series', 'int')
        assert result.shape == (4, 4)
        assert result.index.name is None
        assert result.iloc[0, 0] == -77
        assert result.iloc[3, 3] == 'foo'


    def test_dataframe_with_list_of_series(self):
        # Teste, ob dataframe mit einer Liste von Series korrekt funktioniert
        s1 = pd.Series([1, 2, 3], name='A')
        s2 = pd.Series([4, 5, 6], name='B')
        data = [s1, s2]
        result = dataframe(data)
        assert rtype(result) == ('DataFrame', 'Series', 'int')
        assert result.shape == (3, 2)
        assert result.index.name is None



    def test_dataframe_with_tuple_of_series(self):
        # Teste, ob dataframe mit einem Tupel von Series korrekt funktioniert
        s1 = pd.Series([1, 2, 3], name='A')
        s2 = pd.Series([4, 5, 6], name='B')
        data = (s1, s2)
        result = dataframe(data)
        assert rtype(result) == ('DataFrame', 'Series', 'int')
        assert result.shape == (3, 2)
        assert result.index.name is None

    def test_dataframe_with_list_of_tuples_from_counter(self):
        # Teste, ob dataframe mit einer Liste von Tupeln, die von Counter.most_common() erzeugt wurde, korrekt funktioniert
        from collections import Counter
        import re
        text = "Dies ist ein Testtext, der nicht nur ein Wort mehrfach enthält. Testtext Testtext."
        text = re.sub(r'[^\w\s]', '', text)  # Satzzeichen entfernen
        word_counts = Counter(text.lower().split())
        data = word_counts.most_common()
        result = dataframe(data)
        assert rtype(result) == ('DataFrame', 'Series', 'str')
        assert result.shape == (10, 2)
        assert result.index.name is None
        assert result.iloc[0, 0] == 'testtext'
        assert result.iloc[0, 1] == 3


    def test_dataframe_with_duplicate_series_names(self, capsys):
        # Teste, ob dataframe mit drei Series, von denen zwei den gleichen Namen haben, korrekt funktioniert
        s1 = pd.Series([1, 2, 3], name='A')
        s2 = pd.Series([4, 5, 6], name='B')
        s3 = pd.Series([7, 8, 9], name='A')
        data = [s1, s2, s3]
        with capsys.disabled():
            #print(f"test_dataframe_with_duplicate_series_names pandas")
            result = dataframe(data, framework='pandas')
            assert isinstance(result, pd.DataFrame)
            assert result.shape == (3, 3)
            assert result.index.name is None
            assert result.iat[0, 0] == 1
            assert result.iat[1, 1] == 5
            assert result.iat[2, 2] == 9
            assert result.columns[0] == 'A'
            assert result.columns[1] == 'B'
            assert result.columns[2] == 'A_1'

            #print(f"test_dataframe_with_duplicate_series_names polars")
            result = dataframe(data, framework='polars')
            assert isinstance(result, pl.DataFrame)
            assert result.shape == (3, 3)
            assert result[0, 0] == 1
            assert result[1, 1] == 5
            assert result[2, 2] == 9
            assert result.columns[0] == 'A'
            assert result.columns[1] == 'B'
            assert result.columns[2] == 'A_1'





















