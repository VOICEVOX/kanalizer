from typing import Final, Literal, overload

KANAS: Final[list[str]]
"""c2kの入力に使える文字の一覧。"""
ASCII_ENTRIES: Final[list[str]]
"""c2kで出力される文字の一覧。"""

Strategy = Literal["greedy", "top_k", "top_p"]

class C2k:
    """英単語 -> カタカナの推論を行う。"""

    @overload
    def __init__(
        self,
        *,
        max_length: int = 32,
        strategy: Literal["greedy"] = "greedy",
    ) -> None: ...
    @overload
    def __init__(
        self,
        *,
        max_length: int = 32,
        strategy: Literal["top_k"],
        k: int = 10,
    ) -> None: ...
    @overload
    def __init__(
        self,
        *,
        max_length: int = 32,
        strategy: Literal["top_p"],
        p: float = 0.9,
        t: float = 1.0,
    ) -> None: ...
    def __init__(
        self, *, max_length: int = 32, strategy: Strategy = "greedy", **kwargs
    ) -> None:
        """
        新しいインスタンスを生成する。

        Parameters
        ----------
        max_length : int, default 32
            最大の出力長。
        strategy : str, default "greedy"
            デコードのアルゴリズム。
        k : int, default 10
            strategy="top_k"のときのみ有効。k。
        p : float, default 0.9
            strategy="top_p"のときのみ有効。p。
        t : float, default 1.0
            strategy="top_p"のときのみ有効。t。
        """

        ...

    def __call__(self, word: str) -> str:
        """
        推論を行う。

        Parameters
        ----------
        word : str
            英単語。

        Returns
        -------
        str
            カタカナ。
        """
        ...
