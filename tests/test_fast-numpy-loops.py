
from fast-numpy-loops import initialize
from fast-numpy-loops import main


def test_main():
    pass


def test_initialize():
    assert initialize([b'a', b'bc', b'abc']) == b'abc'
