# __init__.py for the rocq package

# Import the main classes from api.py to make them available
# when the user does `import rocq`
from .api import Simulator, Circuit

# You can also define __all__ to specify what `from rocq import *` imports
__all__ = ['Simulator', 'Circuit']

# Optionally, you could also try to import the backend here and
# expose some of its elements, or perform version checks, etc.
# For now, just exposing the API classes is sufficient.

# print("rocq package initialized") # For debug
