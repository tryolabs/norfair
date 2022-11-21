import numpy as np
import pytest

from norfair.drawing.color import hex_to_bgr


def test_hex_parsing():
    assert hex_to_bgr("#010203") == (3, 2, 1)
    assert hex_to_bgr("#123") == (51, 34, 17)  # (16*3+3, 16*2+2, 16*1+1)
    assert hex_to_bgr("#ffffff") == (255, 255, 255)
