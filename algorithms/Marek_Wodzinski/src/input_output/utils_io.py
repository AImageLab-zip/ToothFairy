### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Union
from enum import Enum

### External Imports ###


### Internal Imports ###

########################

class InputOutputBackend(Enum):
    PYTORCH = 1
    NUMPY = 2
    SITK = 3


class Representation(Enum):
    PYTORCH = 1
    NUMPY = 2