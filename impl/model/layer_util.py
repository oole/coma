import numpy as np

def chebyshev(L, X, K):
    """
    Returns Chebyshev polynomials T_k of order up to K.

    :param L:
    :param X:
    :param K:
    """
    M,N = X.shape

    assert L.dtype == X.dtype

    Xt = np.empty((K, M, N), L.dtype)

    Xt[0, ...] = X

    if K > 1:
        Xt[1, ...] = L.dot(X)

    for k in range(2, K):
        Xt[k, ...] = 2 * L.dot(Xt[k-1, ...]) - Xt[k-2, ...]
    return Xt