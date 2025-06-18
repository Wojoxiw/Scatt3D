# Test the svd_compressed algorithm in dask.
#
# Daniel Sjöberg, 2024-12-23

from mpi4py import MPI
import numpy as np
import dask
import dask.array as da
from dask.base import wait
from dask.array.linalg import compression_matrix, tsqr
from dask.array.utils import svd_flip

# Corrected version of the dask implementation of svd_compressed
def svd_compressed(
    a,
    k,
    iterator="power",
    n_power_iter=0,
    n_oversamples=10,
    seed=None,
    compute=False,
    coerce_signs=True,
):
    """Randomly compressed rank-k thin Singular Value Decomposition.

    This computes the approximate singular value decomposition of a large
    array.  This algorithm is generally faster than the normal algorithm
    but does not provide exact results.  One can balance between
    performance and accuracy with input parameters (see below).

    Parameters
    ----------
    a: Array
        Input array
    k: int
        Rank of the desired thin SVD decomposition.
    iterator: {'power', 'QR'}, default='power'
        Define the technique used for iterations to cope with flat
        singular spectra or when the input matrix is very large.
    n_power_iter: int, default=0
        Number of power iterations, useful when the singular values
        decay slowly. Error decreases exponentially as `n_power_iter`
        increases. In practice, set `n_power_iter` <= 4.
    n_oversamples: int, default=10
        Number of oversamples used for generating the sampling matrix.
        This value increases the size of the subspace computed, which is more
        accurate at the cost of efficiency.  Results are rarely sensitive to this choice
        though and in practice a value of 10 is very commonly high enough.
    compute : bool
        Whether or not to compute data at each use.
        Recomputing the input while performing several passes reduces memory
        pressure, but means that we have to compute the input multiple times.
        This is a good choice if the data is larger than memory and cheap to
        recreate.
    coerce_signs : bool
        Whether or not to apply sign coercion to singular vectors in
        order to maintain deterministic results, by default True.


    Examples
    --------
    >>> u, s, v = svd_compressed(x, 20)  # doctest: +SKIP

    Returns
    -------
    u:  Array, unitary / orthogonal
    s:  Array, singular values in decreasing order (largest first)
    v:  Array, unitary / orthogonal

    References
    ----------
    N. Halko, P. G. Martinsson, and J. A. Tropp.
    Finding structure with randomness: Probabilistic algorithms for
    constructing approximate matrix decompositions.
    SIAM Rev., Survey and Review section, Vol. 53, num. 2,
    pp. 217-288, June 2011
    https://arxiv.org/abs/0909.4061
    """
    comp = compression_matrix(
        a,
        k,
        iterator=iterator,
        n_power_iter=n_power_iter,
        n_oversamples=n_oversamples,
        seed=seed,
        compute=compute,
    )
    if compute:
        comp = comp.persist()
        wait(comp)
    a_compressed = comp.dot(a)
    v, s, u = tsqr(a_compressed.T.conj(), compute_svd=True)
    u = comp.T.conj().dot(u.T.conj())
    v = v.T.conj()
    u = u[:, :k]
    s = s[:k]
    v = v[:k, :]
    if coerce_signs:
        u, v = svd_flip(u, v)
    return u, s, v

Nrows = 30
Ncols = 300

A = np.random.randn(Nrows, Ncols) + 1j*np.random.randn(Nrows, Ncols)
x = np.random.randn(Ncols) + 1j*np.random.randn(Ncols)
b = np.dot(A, x)
u, s, vh = np.linalg.svd(A)
rank = len(s)
#B = u[:,:rank] @ np.diag(s) @ vh[:rank,:]
B = np.dot(u[:,:rank], np.dot(np.diag(s), vh[:rank,:]))
print('Numpy version working: ', np.allclose(A, B))
print(u.shape, s.shape, vh.shape, x.shape, b.shape)
x_svd = np.dot(vh[:rank,:].T.conj(), 1/s*np.dot(u.T.conj(), b))
b_svd = np.dot(u, s*np.dot(vh[:rank,:], x_svd))
print('Numpy solution error: ', np.linalg.norm(x_svd - x))
print('Numpy rhs error: ', np.linalg.norm(b_svd - b))

A_da = da.from_array(A)
x_da = da.from_array(x)
b_da = da.from_array(b)
k = np.min([Nrows, Ncols])
u_da, s_da, vh_da = svd_compressed(A_da, k)
#u_da, s_da, vh_da = da.linalg.svd_compressed(A_da, k)
print(u_da.shape, s_da.shape, vh_da.shape)
B_da = da.dot(u_da[:,:rank], da.dot(da.diag(s_da), vh_da[:rank,:]))
print('Dask version working: ', np.allclose(A_da.compute(), B_da.compute()))
print('Norm: ', np.linalg.norm(A_da.compute() - B_da.compute()))
x_svd_da = da.dot(vh_da.T.conj(), 1/s_da*da.dot(u_da.T.conj(), b_da))
b_svd_da = da.dot(u_da, s_da*da.dot(vh_da, x_svd_da))
print('Dask solution error: ', np.linalg.norm(x_svd_da.compute() - x_da.compute()))
print('Dask rhs error: ', np.linalg.norm(b_svd_da.compute() - b_da.compute()))

