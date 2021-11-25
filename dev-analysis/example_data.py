import pathlib

_DATA_DIR = pathlib.Path(__file__).parent.parent / "example_data"


def data_path(relpath):
    return _DATA_DIR / relpath
