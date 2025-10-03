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
import spgl1
import gc

def scalapackLeastSquares(comm, MPInum, A_np=None, b_np=None, checkVsNp=False):
    '''
    Uses pzgels to least-squares solve a problem
    
    :param MPInum: Number of MPI processes
    :param A_np: Numpy matrix A. Defaults to None for non-master processes
    :param b_np: numpy vector b
    :param checkVsNp: If True, print some comparisons
    '''
    A_np = np.array(A_np, order = 'F') ## put into fortran-major ordering
    b_np = np.array(b_np, order = 'F')
    if(comm.rank == 0): ## setup, starting from master rank
        m, n = A_np.shape ## rows, columns
        nrhs = 1 ## rhs columns
        nprow, npcol = 1, MPInum ## MPI grid - number of process row*col must be <= MPInum. Using many columns since I expect few rows in A
        mb, nb = 16, 16 ## block size for the arrays. Apparently 2^5-9 are standard. Larger seems to increase work size (and so mem cost?) - but 16x16 is faster than 1x1
    else:
        m, n, nrhs, nprow, npcol, mb, nb = 0, 0, 0, 0, 0, 0, 0
    m = comm.bcast(m, root=0)
    n = comm.bcast(n, root=0)
    nrhs = comm.bcast(nrhs, root=0)
    nprow = comm.bcast(nprow, root=0)
    npcol = comm.bcast(npcol, root=0)
    mb = comm.bcast(mb, root=0)
    nb = comm.bcast(nb, root=0)
    
    if(MPI.Get_processor_name() == 'eit000211'): ## for local runs, ctypes finds system libs but need spack's
        scalapack = PyScalapack('/mnt/d/spack/opt/spack/linux-x86_64_v4/netlib-scalapack-2.2.2-sdnhcgsm5j7ltwb4x75rp37l2hlbkdyy/lib/libscalapack.so', '/mnt/d/spack/opt/spack/linux-x86_64_v4/openblas-0.3.30-sftpa2a4yu5lt7aahe5nq2gp7zsnqryg/lib/libopenblas.so')
    else:
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
            print(f'numpy values set, starting timer: (rank {comm.rank})')
            sys.stdout.flush()
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
            print("Work queried as:", int(work[0]), f' for rank {context.rank.value}, info:' , info.value, ', lwork:', lwork.value, f'{context.rank.value=}', f'{comm.rank=}')
        if info.value != 0:
            raise RuntimeError(f"Error in pzgels with info = {info.value}")
        sys.stdout.flush()
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
                x_nplstsq = np.linalg.pinv(A_np, rcond = 1e-4) @ b_np #np.linalg.lstsq(A_np, b_np, rcond=1e-3)[0] ## should be the same thing?
                nptime = timer() - t1
                print('numpy norm |Ax-b|:', np.linalg.norm(np.dot(A_np, x_nplstsq) - b_np))
                print('numpy norm alt1 |Ax-b|:', np.linalg.norm(np.dot(A_np, np.linalg.pinv(A_np, rcond = 1e-4) @ b_np) - b_np))
                print('numpy norm alt2 |Ax-b|:', np.linalg.norm(np.dot(A_np, np.linalg.pinv(A_np, rcond = 1e-6) @ b_np) - b_np))
                print('numpy norm alt3 |Ax-b|:', np.linalg.norm(np.dot(A_np, np.linalg.pinv(A_np, rcond = 1e-8) @ b_np) - b_np))
                print('numpy norm alt4 |Ax-b|:', np.linalg.norm(np.dot(A_np, np.linalg.pinv(A_np, rcond = 1e-10) @ b_np) - b_np))
                print(f'Time to pzgels solve: {scalatime:.2f}, time to numpy solve: {nptime:.2f}')
                print('Norm of difference between solutions:', np.linalg.norm(x0.data-x_nplstsq))
            sys.stdout.flush()
            return x0.data[:, 0], x_nplstsq[:, 0]
        

def solveFromQs(problemName, MPInum): ## Try various solution methods... keeping everything on one process
    comm = MPI.COMM_WORLD
    commself = MPI.COMM_SELF
    if(comm.rank == 0):
    
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
        
        ## mesh stuff on just one process?
        with dolfinx.io.XDMFFile(commself, problemName+'output-qs.xdmf', 'r') as f:
            mesh = f.read_mesh()
            Wspace = dolfinx.fem.functionspace(mesh, ('DG', 0))
            cells = dolfinx.fem.Function(Wspace)
            
            idx = mesh.topology.original_cell_index ## map from indices in the original cells to the current mesh cells
            dofs = Wspace.dofmap.index_map ## has info about dofs on each process
            
        
        ## load in the problem data
        print('data loaded in')
        sys.stdout.flush()
        
        
        with h5py.File(problemName+'output-qs.h5', 'r') as f: ## this is serial, so only needs to occur on the main process
            dofs_map = np.array(f['Function']['real_f']['-4']).squeeze()[idx] ## f being the default name of the function as seen in paraview
            cell_volumes = np.array(f['Function']['real_f']['-3']).squeeze()[idx]
            epsr_array_ref = np.array(f['Function']['real_f']['-2']).squeeze()[idx] + 1j*np.array(f['Function']['imag_f']['-2']).squeeze()[idx]
            epsr_array_dut = np.array(f['Function']['real_f']['-1']).squeeze()[idx] + 1j*np.array(f['Function']['imag_f']['-1']).squeeze()[idx]
            N = len(cell_volumes)
            idx_non_pml = np.nonzero(np.abs(epsr_array_ref) > -1)[0] ## PML cells should have a value of -1
            N_non_pml = len(idx_non_pml)
            A = np.zeros((Nb, N_non_pml), dtype=complex) ## the matrix of scaled E-field stuff
            for n in range(Nb):
                Apart = np.array(f['Function']['real_f'][str(n)]).squeeze() + 1j*np.array(f['Function']['imag_f'][str(n)]).squeeze()
                A[n,:] = Apart[idx][idx_non_pml] ## idx to order as in the mesh, non_pml to remove the pml
                
        print('all data loaded in')
        sys.stdout.flush()
        idx_ap = np.nonzero(np.abs(epsr_array_ref[idx_non_pml]) > 1)[0] ## indices of non-air
        A_ap = A[:, idx_ap]
        print('shape of A:', np.shape(A))
        print('shape of b:', np.shape(b))
        print('in-object cells:', np.size(idx_ap))
        
        
        print('Solving with scalapack least squares and numpy svd...', end='')
        ## non a-priori
        #A_inv = np.linalg.pinv(A, rcond=1e-3)
        b_now = np.array(np.zeros((np.size(b), 1)), order = 'F') ## not sure how much of this is necessary, but reshape it to fit what scalapack expects
        b_now[:, 0] = b
        x, x_np = np.zeros(N, dtype=complex), np.zeros(N, dtype=complex)
        x[idx_non_pml], x_np[idx_non_pml] = scalapackLeastSquares(comm, MPInum, A, b_now, True) ## only the master process gets the result
    else: ## on other processes, call it with nothing
        scalapackLeastSquares(comm, MPInum)
        
    ## a priori lsq
    if(comm.rank == 0):
        x_ap = np.zeros(N, dtype=complex)
        x_np_ap = np.zeros(N, dtype=complex)
        x_ap[idx_ap], x_np_ap[idx_ap] = scalapackLeastSquares(comm, MPInum, A_ap, b_now, True) ## only the master process gets the result
        print(' done')
    else: ## on other processes, call it with nothing
        scalapackLeastSquares(comm, MPInum)
        
    
    if(comm.rank == 0):
        ## write back the result
        with dolfinx.io.XDMFFile(commself, problemName+'testoutput.xdmf', 'w') as f:
            f.write_mesh(mesh)
            cells.x.array[:] = epsr_array_ref + 0j
            f.write_function(cells, -1)
            cells.x.array[:] = epsr_array_dut + 0j
            f.write_function(cells, 0)
            
            ## non a-priori
            cells.x.array[:] = x + 0j
            f.write_function(cells, 1)  
            cells.x.array[:] = x_np + 0j
            f.write_function(cells, 2)  
            
            ## a-priori
            cells.x.array[:] = x_ap + 0j
            f.write_function(cells, 3)    
            cells.x.array[:] = x_np_ap + 0j
            f.write_function(cells, 4)
            
            
            print('Solving with spgl...', end='') ## this method is only implemented for real numbers, to make a large real matrix (hopefully this does not run me out of memory)
            sigma = 1e-6 ## guess for a good sigma
            tau = 1e0 ## guess for a good tau
            iter_lim = 3366
            
            A1, A2 = np.shape(A)[0], np.shape(A)[1]
            Ak = np.zeros((A1*2, A2*2)) ## real A
            bk = np.hstack((np.real(b),np.imag(b))) ## real b
            Ak[:A1,:A2] = np.real(A) ## A11
            Ak[:A1,A2:] = -1*np.imag(A) ## A12
            Ak[A1:,:A2] = 1*np.imag(A) ## A21
            Ak[A1:,A2:] = np.real(A) ## A22
            
            xsol, resid, grad, info = spgl1.spgl1(Ak, bk, iter_lim=iter_lim, verbosity=1)
            x_spgl_bp = np.zeros(N, dtype=complex)
            x_spgl_bp[idx_non_pml] = xsol[:A2] + 1j*xsol[A2:]
            
            xsol, resid, grad, info = spgl1.spgl1(Ak, bk, iter_lim=iter_lim, sigma=sigma, verbosity=1)
            x_spgl_bpdn = np.zeros(N, dtype=complex)
            x_spgl_bpdn[idx_non_pml] = xsol[:A2] + 1j*xsol[A2:]
            
            xsol, resid, grad, info = spgl1.spgl1(Ak, bk, iter_lim=iter_lim, tau=tau, verbosity=1)
            x_spgl_lasso = np.zeros(N, dtype=complex)
            x_spgl_lasso[idx_non_pml] = xsol[:A2] + 1j*xsol[A2:]
            
            del Ak ## maybe this will help with clearing memory
            gc.collect()
            
            ### a-priori
            A1, A2 = np.shape(A_ap)[0], np.shape(A_ap)[1]
            Ak_ap = np.zeros((A1*2, A2*2)) ## real A
            Ak_ap[:A1,:A2] = np.real(A_ap) ## A11
            Ak_ap[:A1,A2:] = -1*np.imag(A_ap) ## A12
            Ak_ap[A1:,:A2] = 1*np.imag(A_ap) ## A21
            Ak_ap[A1:,A2:] = np.real(A_ap) ## A22
            
            xsol, resid, grad, info = spgl1.spgl1(Ak_ap, bk, iter_lim=iter_lim, verbosity=1)
            x_spgl_bp_ap = np.zeros(N, dtype=complex)
            x_spgl_bp_ap[idx_ap] = xsol[:A2] + 1j*xsol[A2:]
            
            xsol, resid, grad, info = spgl1.spgl1(Ak_ap, bk, iter_lim=iter_lim, sigma=sigma, verbosity=1)
            x_spgl_bpdn_ap = np.zeros(N, dtype=complex)
            x_spgl_bpdn_ap[idx_ap] = xsol[:A2] + 1j*xsol[A2:]
            
            xsol, resid, grad, info = spgl1.spgl1(Ak_ap, bk, iter_lim=iter_lim, tau=tau, verbosity=1)
            x_spgl_lasso_ap = np.zeros(N, dtype=complex)
            x_spgl_lasso_ap[idx_ap] = xsol[:A2] + 1j*xsol[A2:]
            
            ## non a-priori
            cells.x.array[:] = x_spgl_bp + 0j
            f.write_function(cells, 5)
            cells.x.array[:] = x_spgl_bpdn + 0j
            f.write_function(cells, 6)
            cells.x.array[:] = x_spgl_lasso + 0j
            f.write_function(cells, 7)
            
            ## a-priori
            cells.x.array[:] = x_spgl_bp_ap + 0j
            f.write_function(cells, 8)
            cells.x.array[:] = x_spgl_bpdn_ap + 0j
            f.write_function(cells, 9)
            cells.x.array[:] = x_spgl_lasso_ap + 0j
            f.write_function(cells, 10)
            
            print(' done')
                    