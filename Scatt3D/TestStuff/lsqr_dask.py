# encoding: utf-8
# Investigate solving with LSQR, particularly with respect to
# partitioning of the matrix.
#
# Daniel Sj�berg, 2025-01-07

#from dask.distributed import Client
#client = Client()

import dask.array as da
import numpy as np

m = 200   # Number of rows
n = 1000  # Number of columns

if True:
    # Compute numpy solution
    A_np = np.random.randn(m, n)
    x0_np = np.random.randn(n)
    b_np = np.matmul(A_np, x0_np)
    print(np.shape(A_np), np.shape(b_np))
    x_np, res_np, rank_np, s_np = np.linalg.lstsq(A_np, b_np)
    error_np = np.linalg.norm(x_np - x0_np)

    # Compute dask solution
    M = m ## must chunk them... to the number of processes?
    N = n/80
    
    A_da = da.from_array(A_np).rechunk()
    b_da = da.from_array(b_np).rechunk()
    print('chunks', A_da.chunksize, b_da.chunksize)
    x_da, res_da, rank_da, s_da = da.linalg.lstsq(A_da, b_da)
    x_da = x_da.compute()
    error_da = np.linalg.norm(x_da - x0_np)
    
    print(np.shape(x_np), np.shape(x_da))
    
    diff = np.linalg.norm(x_np - x_da)

    print(f'error_np = {error_np}\nerror_da = {error_da}\ndiff = {diff}')
else:
    A = da.random.random((m, n))
    x0 = da.random.random(n)
    b = da.dot(A, x0)
    x, res, rank, s = da.linalg.lstsq(A, b)
    error = da.linalg.norm(x - x0)
    print(f'error = {error.compute()}')
#client.close()
