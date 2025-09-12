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
from numba.core.types import none

def scalapackLeastSquares(MPInum, A_np, b_np, checkVsNp=False):
    # Setup
    m, n = A_np.shape ## rows, columns
    nrhs = 1 ## rhs columns
    nprow, npcol = 1, MPInum ## MPI grid - number of process row*col must be <= MPInum. Using many columns since I expect few rows in A
    mb, nb = 16, 16 ## block size for the arrays. Apparently 2^5-9 are standard. Larger seems to increase work size (and so mem cost?) - but 16x16 is faster than 1x1
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
            t1 = timer()
        ## make the computational context's arrays, then redistribute the feeder's arrays over to them
        A = context.array(m, n, mb, nb, dtype=np.complex128)
        scalapack.pgemr2d["Z"]( # Z for complex double
            *(m, n), ## matrix dimensions
            *A0.scalapack_params(),
            *A.scalapack_params(),
            context.ictxt,
        )
        b = context.array(max(m, n), nrhs, mb, nb, dtype=np.complex128)
        scalapack.pgemr2d["Z"]( ## read more https://info.gwdg.de/wiki/doku.php?id=wiki:hpc:scalapack
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
            scalatime = timer() - t1
            print('pzgels norm |Ax-b|:', np.linalg.norm(np.dot(A_np, x0.data) - b_np))
            
            if(checkVsNp):
                t1 = timer()
                x_nplstsq = np.linalg.lstsq(A_np, b_np)[0]
                nptime = timer() - t1
                print('numpy norm |Ax-b|:', np.linalg.norm(np.dot(A_np, x_nplstsq) - b_np))
                print(f'Time to pzgels solve: {scalatime:.2f}, time to numpy solve: {nptime:.2f}')
                print('Norm of difference between solutions:', np.linalg.norm(x0.data-x_nplstsq))

def testLSTSQ(problemName, MPInum): ## Try least squares using scalapack
    comm = MPI.COMM_WORLD
    ## load in all the data with the main process
    if(comm.rank == 0):
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
    
    ## mesh stuff on each process?
    with dolfinx.io.XDMFFile(comm, problemName+'output-qs.xdmf', 'r') as f:
        mesh = f.read_mesh()
    Wspace = dolfinx.fem.functionspace(mesh, ('DG', 0))
    cells = dolfinx.fem.Function(Wspace)
    idx = mesh.topology.original_cell_index
    dofs = Wspace.dofmap.index_map ## has info about dofs on each process
    
    ## load in the problem data
    if(comm.rank == 0):
        print('data loaded in')
        sys.stdout.flush()
        with h5py.File(problemName+'output-qs.h5', 'r') as f: ## this is serial, so only needs to occur on the main process
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
        print('all data loaded in')
        sys.stdout.flush()
        ## non a-priori
        #A_inv = np.linalg.pinv(A, rcond=1e-3)
        x = scalapackLeastSquares(MPInum, A, b, True) #np.dot(A_inv, b) 
        print('scalapack finished')
        sys.stdout.flush()
        if (True): ## a priori
            idx_ap = np.nonzero(np.abs(epsr_array_ref) > 1)[0] ## indices of non-air
            x_ap = np.zeros(np.shape(A)[1])
            print('A:', np.shape(A))
            print('b:', np.shape(b))
            print('in-object cells:', np.size(idx_ap))
            A = A[:, idx_ap]
            
            if(False): ## solve with numpy
                A_inv = np.linalg.pinv(A, rcond=1e-3)
                x_ap[idx_ap] = np.dot(A_inv, b)
                #x[idx_ap] = np.linalg.lstsq(A, b)
            else: ## solve with scalapack
                x_ap[idx_ap] = scalapackLeastSquares(MPInum, A, b, True)
    else: ## distribute the solution to all ranks (not sure if the None is needed)
        x = None
        x_ap = None
    x = comm.bcast(x, root=0)
    x_ap = comm.bcast(x_ap, root=0)
    
    ## write back the result      
    with dolfinx.io.XDMFFile(comm, problemName+'testoutput.xdmf', 'w') as f:
        f.write_mesh(mesh)
        cells.x.array[:] = epsr_array_dut + 0j
        f.write_function(cells, 0)
        cells.x.array[:] = epsr_array_ref + 0j
        f.write_function(cells, -1)
        
        ## non a-priori
        cells.x.array[:] = x[dofs.local_range[0]:dofs.local_range[1]] + 0j ## this should give the local values into each process (hopefully the ones that get written)
        f.write_function(cells, 1)  
        
        ## a-priori
        cells.x.array[:] = x_ap[dofs.local_range[0]:dofs.local_range[1]] + 0j
        f.write_function(cells, 2)            

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