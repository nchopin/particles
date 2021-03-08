import numba as nb
import numpy as np


@nb.njit
def _is_prime(m):
    if m <= 1:
        return False
    if m == 2:
        return True
    if m % 2 == 0:
        return False
    i = 3
    while i < m:
        if m % i == 0:
            return False
        i += 2

    return True


@nb.njit
def _get_first_n_primes(n):
    """
    Examples
    --------
    >>> list(_get_first_n_primes(10))
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

    >>> list(_get_first_n_primes(1))
    [2]
    """
    res = np.zeros((n,), dtype=np.int32)
    i = 0
    k = 2
    while i < n:
        if _is_prime(k):
            res[i] = k
            i += 1
        k += 1
    return res


_halton_types = ["void(float64[:], int32)"]


@nb.guvectorize(_halton_types, "(n),()")
def _halton_one(res, b):
    """
    This is a slight modification of Wikipedia's article algorithm
    https://en.wikipedia.org/wiki/Halton_sequence#Implementation

    Parameters
    ----------
    res : array_like
        result sequence modified inplace
    b: int
        base of the sequence
    """
    n, d = 0, 1
    N = res.shape[0]
    i = 0
    while i < N:
        while True:
            x = d - n
            if x == 1:
                n = 1
                d *= b
            else:
                y = d // b
                while x <= y:
                    y //= b
                n = (b + 1) * y - x
            break
        res[i] = n / d
        i = i + 1


def halton(N, dim):
    """ Halton sequence.

    Component i of the sequence consists of a Van der Corput sequence in base b_i,
    where b_i is the i-th prime number.

    Parameters
    ----------
    N : int
        length of sequence
    dim: int
        dimension

    Returns
    -------
    (N, dim) numpy array.

    Examples
    --------
    >>> res = halton(10000, 5)
    >>> res.mean(0)
    array([0.49983248, 0.49975861, 0.49987205, 0.49974173, 0.49955577])
    """

    primes = _get_first_n_primes(dim)
    res = np.empty((dim, N), dtype=np.float64)
    _halton_one(res, primes)
    return res.T
