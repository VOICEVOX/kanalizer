from typing import TYPE_CHECKING
from ._rust import __version__, convert, INPUT_CHARS, OUTPUT_CHARS
from ._error import IncompleteConversionError

if TYPE_CHECKING:
    from ._rust import Strategy

__all__ = [
    "__version__",
    "INPUT_CHARS",
    "OUTPUT_CHARS",
    "Strategy",
    "convert",
    "IncompleteConversionError",
]
