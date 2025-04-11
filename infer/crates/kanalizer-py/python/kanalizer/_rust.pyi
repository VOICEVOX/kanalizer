from typing import Final, Literal, overload

__version__: Final[str]
"""バージョン。"""

INPUT_CHARS: Final[set[str]]
"""Kanalizerの入力に使える文字の一覧。"""
OUTPUT_CHARS: Final[set[str]]
"""Kanalizerから出力されうる文字の一覧。"""

Strategy = Literal["greedy", "top_k", "top_p"]
"""デコードのアルゴリズム。"""

@overload
def convert(
    word: str,
    /,
    *,
    max_length: int = 32,
    strict: bool = True,
    error_on_incomplete: bool = True,
    strategy: Literal["greedy"] = "greedy",
) -> str: ...
@overload
def convert(
    word: str,
    /,
    *,
    max_length: int = 32,
    strict: bool = True,
    error_on_incomplete: bool = True,
    strategy: Literal["top_k"],
    k: int = 10,
) -> str: ...
@overload
def convert(
    word: str,
    /,
    *,
    max_length: int = 32,
    strict: bool = True,
    error_on_incomplete: bool = True,
    strategy: Literal["top_p"],
    p: float = 0.9,
    t: float = 1.0,
) -> str: ...
def convert(
    word: str,
    /,
    *,
    max_length: int = 32,
    strategy: Strategy = "greedy",
    strict: bool = True,
    error_on_incomplete: bool = True,
    **kwargs,
) -> str:
    """
    推論を行う。

    Parameters
    ----------
    word : str
        英単語。
    max_length : int, default 32
        最大の出力長。
    strict : bool, default True
        入力の検証を行うかどうか。
        Falseの場合、無効な文字は無視されます。
    error_on_incomplete : bool, default True
        推論が終了しなかった場合にエラーを返すかどうか。
    strategy : Strategy, default "greedy"
        デコードのアルゴリズム。
    k : int, default 10
        strategy="top_k"のときのみ有効。Top-KアルゴリズムのK。
    p : float, default 0.9
        strategy="top_p"のときのみ有効。Top-PアルゴリズムのP。
    t : float, default 1.0
        strategy="top_p"のときのみ有効。Top-PアルゴリズムのTemperature。

    Raises
    ------
    ValueError
        - strictがTrue、かつ`word`が空文字列の場合。
        - strictがTrue、かつ`word`にKanalizerの入力に使えない文字が含まれている場合。
        - `max_length`が0以下の場合。
    IncompleteConversionError
        - `error_on_incomplete`がTrue、かつ推論が終了しなかった場合。
    """
    ...
