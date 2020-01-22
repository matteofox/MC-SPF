try:
    from ._version import __version__
except(ImportError):
    pass

from . import utils
from . import routines

__all__ = ['utils','routines']
