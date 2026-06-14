from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("diff-biophys")
except PackageNotFoundError:
    # Package not installed (e.g. running directly from source tree)
    __version__ = "unknown"

from . import cryo_em
from .ensemble import Ensemble

try:
    from . import torch_interop
except ImportError:
    pass
