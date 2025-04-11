class IncompleteConversionError(Exception):
    """
    変換が終了しなかった場合に発生するエラー。

    Attributes
    ----------
    incomplete_output : str
        不完全な出力。
    """

    incomplete_output: str

    def __init__(self, message: str, incomplete_output: str):
        super().__init__(message)
        self.incomplete_output = incomplete_output
