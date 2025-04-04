import collections.abc
import warnings

import numpy as np
import polars as pl

from .config import Config
from .pandas import first_valid_value, last_valid_value


#################################################################################
# ...............................................................................
# Class type_info_polars
# ...............................................................................
#################################################################################

class type_info_polars:
    """
    Provides information about polars types and standardises them.
    Is initialised with anything, e.g. with the name of a class, or with the class itself.
    Or, even better, with a series.
    Ex:   i = type_info_polars('Int32')
          i.info()            # returns all attributes, including for example:
          i.class_object      # the class object
          i.name              # the name of the Dtype
          i.name_instance     # type of the contents of the series
          i.instance1         # an example instance that is not NaN
    """

    def __init__(self, search):

        # Beispielinstanzen
        self.instance1 = None
        self.instance2 = None
        if isinstance(search, pl.Series):  # Es wurde eine Polars Series übergeben
            self.instance1 = first_valid_value(search)
            self.instance2 = last_valid_value(search)
            search = str(search.dtype)
        elif not isinstance(search, str):  # Es wurde eine Klasse übergeben
            search = str(search)

        # search ist jetzt str
        self.search = search
        self.name = None
        self.framework = 'pl'
        self.name_short = search
        self.name_long = None
        self.class_object = None
        self.is_hashable = None
        self.nan_allowed = True
        self.name_instance = ''
        self.xmin = None
        self.xmax = None

        # name_instance
        if self.instance1 is not None:
            if (str(type(self.instance1)) == str(type(self.instance2))):
                self.name_instance = type(self.instance1).__name__
            else:
                self.name_instance = 'mix'

        # is_hashable
        if self.name_instance == 'mix':
            self.is_hashable = False
        else:
            self.is_hashable = isinstance(self.instance1, collections.abc.Hashable) and isinstance(self.instance2,
                                                                                                   collections.abc.Hashable)

        # name_short
        if '.' in search:
            self.name_short = search.split('.')[-1]
        self.name_short = self.name_short.replace("<class '", "").replace("'>", "").replace("polars.", "").replace(
            "DataType", "")

        # name_long
        if self.name_short.startswith(('Utf8', 'String')):
            self.name_long = 'pl.Utf8'
            self.name_short = 'Utf8'  # Namen standardisieren, damit astype ihn versteht
        elif self.name_short.startswith(('Int', 'UInt', 'Float', 'Boolean')):
            self.name_long = 'pl.' + self.name_short
        elif self.name_short.startswith(('Date', 'Time', 'Datetime')):
            self.name_long = 'pl.' + self.name_short
        elif self.name_short.startswith(('List')):
            self.name_long = 'pl.List'
            self.is_hashable = False
        elif self.name_short.startswith(('Null')):
            self.name_long = 'pl.Null'
            self.is_hashable = True
        elif self.name_short.startswith(('Object')):
            self.name_long = 'pl.Object'
            self.is_hashable = False

        # name
        self.name = self.framework + '.' + self.name_short

        # class_object
        if self.name_long:
            self.class_object = eval(self.name_long)

        # xmin und xmax
        if self.name_short.startswith(('Int', 'UInt')):
            self.nan_allowed = False
            if self.name_short.endswith('8'):
                self.xmin = -128 if self.name_short.startswith('Int') else 0
                self.xmax = 127 if self.name_short.startswith('Int') else 255
            elif self.name_short.endswith('16'):
                self.xmin = -32768 if self.name_short.startswith('Int') else 0
                self.xmax = 32767 if self.name_short.startswith('Int') else 65535
            elif self.name_short.endswith('32'):
                self.xmin = -2147483648 if self.name_short.startswith('Int') else 0
                self.xmax = 2147483647 if self.name_short.startswith('Int') else 4294967295
            elif self.name_short.endswith('64'):
                self.xmin = -9223372036854775808 if self.name_short.startswith('Int') else 0
                self.xmax = 9223372036854775807 if self.name_short.startswith('Int') else 18446744073709551615
        elif self.name_short.startswith('Float'):
            self.nan_allowed = True
            if self.name_short.endswith('32'):
                self.xmin = -3.402823466e+38
                self.xmax = 3.402823466e+38
            elif self.name_short.endswith('64'):
                self.xmin = -1.797693134e+308
                self.xmax = 1.797693134e+308
        elif self.name_short.startswith('Boolean'):
            self.nan_allowed = False
            self.xmin = False
            self.xmax = True
        elif self.name_short.startswith('Utf8'):
            self.nan_allowed = True
        elif self.name_short.startswith('List'):
            self.nan_allowed = True
        elif self.name_short.startswith('Null'):
            self.nan_allowed = True
        elif self.name_short.startswith('Object'):
            self.nan_allowed = True
        elif self.name_short.startswith(('Date', 'Time', 'Datetime')):
            self.nan_allowed = True

    def info(self):
        """Returns all attributes"""
        result = dict(self.__dict__)  # Kopie ziehen
        del result['search']
        return result