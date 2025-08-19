# encoding: utf-8
## this file will have much of the postprocessing and associated functions

from mpi4py import MPI
import numpy as np
import dolfinx
import ufl
import basix
import functools
from timeit import default_timer as timer
import gmsh
import sys
from scipy.constants import c as c0, mu_0 as mu0, epsilon_0 as eps0, pi
from matplotlib import pyplot as plt
import h5py
import PyScalapack ## https://github.com/USTC-TNS/TNSP/tree/main/PyScalapack
import ctypes.util

def scalapackLeastSquares(prob, A_np, b_np, checkVsNp):
    # Setup
    m, n = A_np.shape ## rows, columns
    nrhs = 1 ## rhs columns
    nprow, npcol = 1, prob.MPInum ## MPI grid - number of process row*col must be <= MPInum. Using many columns since I expect few rows in A
    mb, nb = 1, 1 ## block size for the arrays. I guess 1 is fine?
    scalapack = PyScalapack(ctypes.util.find_library("scalapack"), ctypes.util.find_library("blas"))

    with (
            scalapack(b'C', nprow, npcol) as context, ## computational context
            scalapack(b'C', 1, 1) as context0, ## feeder context
    ):
        ## make the arrays in the feeder context
        A0 = context0.array(m, n, mb, nb, dtype=np.complex128) 
        b0 = context0.array(max(m, n), nrhs, mb, nb, dtype=np.complex128) ## must be larger to... contain space for computations?
        x0 = context0.array(n, nrhs, mb, nb, dtype=np.complex128)
        
        if context0: ## give numpy values to the feeder context's arrays
            A0.data[...] = A_np
            b0.data[:m] = b_np
        ## make the computational context's arrays, then redistribute the feeder's arrays over to them
        A = context.array(m, n, mb, nb, dtype=np.complex128)
        scalapack.pgemr2d["Z"]( # Z for complex double
            *(m, n), ## matrix dimensions
            *A0.scalapack_params(),
            *A.scalapack_params(),
            context.ictxt,
        )
        b = context.array(max(m, n), nrhs, mb, nb, dtype=np.complex128)
        scalapack.pgemr2d["Z"](
            *(max(m, n), nrhs), ## matrix dimensions
            *b0.scalapack_params(),
            *b.scalapack_params(),
            context.ictxt,
        )
        ## perform the computation
    
        # Workspace query
        work = (np.ctypeslib.as_ctypes_type(np.float64) * 2)() ## based on the example in observer.py in TNSP/tetragono/tetragono/sampling_lattice/observer.py
        lwork = scalapack.neg_one ## -1 to query for optimal size
        info = scalapack.ctypes.c_int() ## presumably this is changed to 0 on success... or changed away on failure. 
        
        scalapack.pzgels( # see documentation for pzgels: https://www.netlib.org/scalapack/explore-html/d2/dcf/pzgels_8f_aad4aa6a5bf9443ac8be9dc92f32d842d.html#aad4aa6a5bf9443ac8be9dc92f32d842d
            b'N', ## for solve A x = b, T for trans A^T x = b
            A.m, A.n, nrhs, # rows, columns, rhs columns
            *A.scalapack_params(), ## the array, row and column indices, and descriptor
            *b.scalapack_params(),
            work, lwork, info) ## pointer to the workspace data, size of workspace, output code  
          
        if context.rank.value == 0:
            print("Work queried as:", int(work[0]), ', info:' , info.value, ', lwork:', lwork.value)
        if info.value != 0:
            raise RuntimeError(f"Error in pzgels with info = {info.value}")
        lwork = int(work[0]) ## size of workspace
        
        work = np.zeros(lwork, dtype=np.complex128, order='F')
        # Solve
        scalapack.pzgels(
            b'N',
            A.m, A.n, nrhs,
            *A.scalapack_params(),
            *b.scalapack_params(),
            scalapack.numpy_ptr(work), lwork, info)
        if info.value != 0:
            raise RuntimeError(f"Error in pzgels with info = {info.value}")
        
        ## now redistribute the result into the feeder context
        scalapack.pgemr2d["Z"](
            *(n, nrhs),
            *b.scalapack_params(),
            *x0.scalapack_params(),
            context.ictxt,
        )
        
        if context0: ## print a check of the results
            print('numpy norm |Ax-b|:', np.linalg.norm(np.dot(A_np, x_nplstsq) - b_np))
            print('pzgels norm |Ax-b|:', np.linalg.norm(np.dot(A_np, x0.data) - b_np))
            
        if(checkVsNp):
            x_nplstsq = np.linalg.lstsq(A_np, b_np)[0]
            print('difference:', np.linalg.norm(x0.data-x_nplstsq))

def testLSTQ(problemName): ## Try least squares using scalapack
    pass

def testSVD(problemName): ## Takes data files saved from a problem after running makeOptVectors, does stuff on it
    ## load in all the data
    data = np.load(problemName+'output.npz')
    b = data['b']
    fvec = data['fvec']
    S_ref = data['S_ref']
    S_dut = data['S_dut']
    epsr_mat = data['epsr_mat']
    epsr_defect = data['epsr_defect']
    N_antennas = data['N_antennas']
    Nf = len(fvec)
    Np = S_ref.shape[-1]
    Nb = len(b)
    
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, problemName+'output-qs.xdmf', 'r') as f:
        mesh = f.read_mesh()
    Wspace = dolfinx.fem.functionspace(mesh, ('DG', 0))
    cells = dolfinx.fem.Function(Wspace)
    
    idx = mesh.topology.original_cell_index
    Nb = Nf*N_antennas*N_antennas
    
    with h5py.File(problemName+'output-qs.h5', 'r') as f:
        cell_volumes = np.array(f['Function']['real_f']['-3']).squeeze()
        cell_volumes[:] = cell_volumes[idx]
        epsr_array_ref = np.array(f['Function']['real_f']['-2']).squeeze() + 1j*np.array(f['Function']['imag_f']['-2']).squeeze()
        epsr_array_ref = epsr_array_ref[idx]
        epsr_array_dut = np.array(f['Function']['real_f']['-1']).squeeze() + 1j*np.array(f['Function']['imag_f']['-1']).squeeze()
        epsr_array_dut = epsr_array_dut[idx]
        N = len(cell_volumes)
        A = np.zeros((Nb, N), dtype=complex) ## the matrix of scaled E-field stuff
        for n in range(Nb):
            A[n,:] = np.array(f['Function']['real_f'][str(n)]).squeeze() + 1j*np.array(f['Function']['imag_f'][str(n)]).squeeze()
            A[n,:] = A[n,idx]
        
    if (True): ## a priori
        idx = np.nonzero(np.abs(epsr_array_ref) > 1)[0] ## indices of non-air
        x = np.zeros(np.shape(A)[1])
        print('A', np.shape(A))
        print('b', np.shape(b))
        print('in-object cells',np.size(idx))
        A = A[:, idx]
        A_inv = np.linalg.pinv(A, rcond=1e-2)
        x[idx] = np.dot(A_inv, b)
        #x[idx] = np.linalg.lstsq(A, b)
        
    #### alternatively, try elemental?
    #with dolfinx.io.XDMFFile(MPI.COMM_WORLD, problemName+'output-qs.h5', 'r') as f:
    
                
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, problemName+'testoutput.xdmf', 'w') as f:
        f.write_mesh(mesh)
        cells.x.array[:] = epsr_array_dut + 0j
        f.write_function(cells, 0)
        cells.x.array[:] = epsr_array_ref + 0j
        f.write_function(cells, -1)
        
        cells.x.array[:] = x + 0j
        f.write_function(cells, 1)            
    
    print('do SVD now')