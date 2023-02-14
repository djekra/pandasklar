# __init__.py

#print('pandasklar init')

import locale 
locale.setlocale(locale.LC_ALL, '') 


#Basics
from .config       import *
from .pandas       import *

# Themen
from .develop      import *
from .scale        import *
from .compare      import *
from .analyse      import *
from .content      import *
from .string       import *
from .rank         import *
from .aggregate    import *
from .subsets      import *
from .plot         import *

# Klassen
from .type_info    import type_info
from .values_info  import values_info


