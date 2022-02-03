from math_demo import hyperbolic
import numpy as np

N_POINTS = 10

def test_caternary_symmetric():
    x = np.asarray([2 ** i for i in range(N_POINTS)])
    assert np.all(hyperbolic.caternary(x) == hyperbolic.caternary(-x))
