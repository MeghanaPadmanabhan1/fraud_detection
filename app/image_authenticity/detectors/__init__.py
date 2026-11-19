"""Importing this package registers every built-in detector.

Add a new technique by dropping a module here that defines a Detector subclass
decorated with @register, then importing it below.
"""
from . import copy_move  # noqa: F401
from . import ela  # noqa: F401
from . import jpeg_ghost  # noqa: F401
from . import noise  # noqa: F401
from . import spectral  # noqa: F401
from . import vlm  # noqa: F401
from . import watermark  # noqa: F401
