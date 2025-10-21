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
import cvxpy as cp
import resource
import psutil, threading, os, time
from timeit import default_timer as timer
import scipy

def cvxpySolve(A, b, solveType, solver=cp.SCS, cell_volumes=None, tau=2e-1, sigma=1e-5, solver_settings={}, verbose=False): ## put this in a function to allow gc?
    N_x = np.shape(A)[1]
    x_cvxpy = cp.Variable(N_x, complex=True)
    if(solveType==0):
        objective_1norm = cp.Minimize(cp.norm(A @ x_cvxpy - b, p=1))
        problem_cvxpy = cp.Problem(objective_1norm)
    if(solveType==1):
        objective_2norm = cp.Minimize(cp.norm(A @ x_cvxpy - b, p=2))
        problem_cvxpy = cp.Problem(objective_2norm)
    if(solveType==2): ## bpdn
        if(cell_volumes is None): ## can normalize with cell volumes, or not
            objective_bpdn = cp.Minimize(cp.norm(x_cvxpy, p=1))
        else:
            objective_bpdn = cp.Minimize(cp.norm(x_cvxpy*cell_volumes, p=1))
        constraint_bpdn = [cp.norm(A @ x_cvxpy - b, p=2) <= sigma]
        problem_cvxpy = cp.Problem(objective_bpdn, constraint_bpdn)
    if(solveType==3): ## lasso
        objective_2norm = cp.Minimize(cp.norm(A @ x_cvxpy - b, p=2))
        if(cell_volumes is None): ## can normalize with cell volumes, or not
            constraint_lasso = [cp.norm(x_cvxpy, p=1) <= tau]
        else:
            constraint_lasso = [cp.norm(x_cvxpy*cell_volumes, p=1) <= tau]
        problem_cvxpy = cp.Problem(objective_2norm, constraint_lasso)
    problem_cvxpy.solve(verbose=verbose, solver=solver)
    print(f'cvxpy norm of residual: {cp.norm(A @ x_cvxpy - b, p=2).value} ({solveType=})')
    return x_cvxpy.value
    
def reconstructionError(epsr_rec, epsr_ref, epsr_dut, cell_volumes, printIt=True):
    '''
    Find the 'error' figure of merit in reconstructed delta eps_r
    :param epsr_rec: Reconstructed delta eps_r
    :param epsr_ref: Reference eps_r
    :param epsr_dut: Dut eps_r
    :param cell_volumes: Volume of each cell - results are normalized by this
    '''
    delta_epsr_actual = epsr_dut-epsr_ref
    #error = np.mean(np.abs(epsr_rec - delta_epsr_actual) * cell_volumes)
    error = np.mean(np.abs(epsr_rec/np.mean(epsr_rec) - delta_epsr_actual/np.mean(delta_epsr_actual)) * cell_volumes) ## try to account for the reconstruction being innacurate in scale, if still somewhat accurate in shape
    if(printIt):
        print(f'Reconstruction error: {error:.3e}')
    return error

def numpySVDfindOptimal(A, b, epsr_ref, epsr_dut, cell_volumes): ## optimize for rcond
    u, s, vt = np.linalg.svd(A.conjugate(), full_matrices=False) ## singular values in descending order
    def calcAccuracy(svd_t, verbose=False, returnSol=False): ### uses the computed svd decomp. and an SVD threshold to find a figure of merit (for optimizing that threshold)
        tol = np.asarray(10**-(svd_t)) ### use this to help the optimizer... otherwise, could maybe try simulated annealing
        cutoff = tol[..., np.newaxis] * np.amax(s, axis=-1, keepdims=True)
        large = s > cutoff
        sinv = np.divide(1, s, where=large)
        sinv[~large] = 0
        A_inv = np.matmul( np.transpose(vt), np.multiply( sinv[..., np.newaxis], np.transpose(u) ) )
        x_try = np.dot(A_inv, b) ## set the reconstructed indices
        err = reconstructionError(x_try, epsr_ref, epsr_dut, cell_volumes)
        if verbose: ## so I don't spam the console
            print(f'Error calculated as {err:.3e}, svd_t = 10**-{svd_t}')
        if(returnSol):
            return x_try
        else:
            return err
    optsol = scipy.optimize.minimize(calcAccuracy, 2, method='Nelder-Mead')## initially guess that we want to include only up to relative values of 1e-2
    optThresh = optsol.x
    print(f'Optimal threshold calculated as '+str(optThresh))
    x = calcAccuracy(optThresh, returnSol = True, verbose = True)
    return x

def testSolverSettings(A, b, epsr_ref, epsr_dut, cell_volumes): # Varies settings in the ksp solver/preconditioner, plots the time and iterations a computation takes. Uses the sphere-scattering test case
    '''
    Tests to try to find the best settings for a solver, using the reconstruction error as a goal.
    :param A: Matrix A
    :param b: Vector b
    :param epsr_ref: Reference epsr
    :param epsr_dut: DUT epsr
    :param cell_volumes: Cell volumes
    '''
    
    settings = [] ## list of solver settings
    
    ## MG tests
    testName = 'cvxpy_type0_testsolvers'
    for solver in cp.installed_solvers():
        settings.append( {'solver': solver, 'solveType': 0} )
    
                            
    num = len(settings)
    for i in range(num):
        print(f'Settings {i}:', settings[i])
    
    omegas = np.arange(num) ## Number of the setting being varied, if it is not a numerical quantity
    ts = np.zeros(num)
    errors = np.zeros(num)
    mems = np.zeros(num) ## to get the memories for each run, use psutil on each process with sampling every 0.5 seconds
    for i in range(num):
        print('\033[94m' + f'Run {i+1}/{num} with settings:' + '\033[0m', settings[i])
        
        process_mem = 0
        process_done = False
        proc = psutil.Process(os.getpid())
        def getMem():
            nonlocal process_mem
            while not process_done:
                process_mem = max(proc.memory_info().rss/1024**2, process_mem) ## get max mem
                time.sleep(0.4362)
        
        try:
            t = threading.Thread(target=getMem)
            t.start()
            starTime = timer()
            x_rec = cvxpySolve(A, b, cell_volumes=cell_volumes, **settings[i])
            ts[i] = timer() - starTime
            errors[i] = reconstructionError(x_rec, epsr_ref, epsr_dut, cell_volumes)
            mems[i] = process_mem
        except Exception as error: ## if the solver isn't defined or something, try skipping it
            print('\033[31m' + 'Warning: solver failed' + '\033[0m', error)
            ts[i] = np.nan
            errors[i] = np.nan
            mems[i] = np.nan
        process_done = True
                
    fig, ax1 = plt.subplots()
    fig.subplots_adjust(right=0.45)
    fig.set_size_inches(29.5, 14.5)
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines.right.set_position(("axes", 1.2))
    ax3.set_yscale('log')
    ax1.grid()
    
    l1, = ax1.plot(omegas, mems, label = 'Memory Costs [GB]', linewidth = 2, color='tab:red')
    l2, = ax2.plot(omegas, ts, label = 'Time [s]', linewidth = 2, color='tab:blue')
    l3, = ax3.plot(omegas, errors, label = 'errors', linewidth = 2, color = 'orange')
    
    plt.title(f'Solver Time by Setting ({testName})')
    ax1.set_xlabel(r'Setting (composite try #)')
    ax1.set_ylabel('#')
    ax2.set_ylabel('Time [s]')
    ax3.set_ylabel('log10(Errors)')
    
    ax1.yaxis.label.set_color(l1.get_color())
    ax2.yaxis.label.set_color(l2.get_color())
    ax3.yaxis.label.set_color(l3.get_color())
    tkw = dict(size=4, width=1.5)
    ax1.tick_params(axis='y', colors=l1.get_color(), **tkw)
    ax2.tick_params(axis='y', colors=l2.get_color(), **tkw)
    ax3.tick_params(axis='y', colors=l3.get_color(), **tkw)
    ax1.tick_params(axis='x', **tkw)
    ax1.legend(handles=[l1, l2, l3])
    
    fig.tight_layout()
    fig.tight_layout() ## need both of these for some reason
    #plt.savefig(prob.dataFolder+prob.name+testName+'_post-processing_solversettingsplot.png')
    
    nprint = min(num, 10)
    print(f'Top {nprint} Options #s:') ## lowest errors
    idxsort = np.argsort(errors)
    for k in range(nprint):
        print(f'#{idxsort[k]+1}: t={ts[idxsort[k]]:.3e}, error={errors[idxsort[k]]}, mem={mems[idxsort[k]]:.3f}GiB --- ')
        print(settings[idxsort[k]])
        print()
    
    plt.show()

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
                try:
                    t1 = timer()
                    x_nplstsq = np.linalg.pinv(A_np, rcond = 3e-4) @ b_np #np.linalg.lstsq(A_np, b_np, rcond=1e-3)[0] ## should be the same thing?
                    nptime = timer() - t1
                    print('numpy norm |Ax-b|:', np.linalg.norm(np.dot(A_np, x_nplstsq) - b_np))
                    #print('numpy norm alt1 |Ax-b|:', np.linalg.norm(np.dot(A_np, np.linalg.pinv(A_np, rcond = 1e-4) @ b_np) - b_np))
                    #print('numpy norm alt2 |Ax-b|:', np.linalg.norm(np.dot(A_np, np.linalg.pinv(A_np, rcond = 1e-6) @ b_np) - b_np))
                    #print('numpy norm alt3 |Ax-b|:', np.linalg.norm(np.dot(A_np, np.linalg.pinv(A_np, rcond = 1e-8) @ b_np) - b_np))
                    #print('numpy norm alt4 |Ax-b|:', np.linalg.norm(np.dot(A_np, np.linalg.pinv(A_np, rcond = 1e-10) @ b_np) - b_np))
                    print(f'Time to pzgels solve: {scalatime:.2f}, time to numpy solve: {nptime:.2f}')
                    print('Norm of difference between solutions:', np.linalg.norm(x0.data-x_nplstsq))
                except Exception as error:
                    print(f'Numpy solution error, skipping: {error}')
                    x_nplstsq = np.zeros(np.shape(x0.data))
                    
            sys.stdout.flush()
            return x0.data[:, 0]
    
def solveFromQs(problemName, MPInum): ## Try various solution methods... keeping everything on one process
    gc.collect()
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
            epsr_ref = np.array(f['Function']['real_f']['-2']).squeeze()[idx] + 1j*np.array(f['Function']['imag_f']['-2']).squeeze()[idx]
            epsr_dut = np.array(f['Function']['real_f']['-1']).squeeze()[idx] + 1j*np.array(f['Function']['imag_f']['-1']).squeeze()[idx]
            N = len(cell_volumes)
            idx_non_pml = np.nonzero(np.real(dofs_map) > -1)[0] ## PML cells should have a value of -1
            N_non_pml = len(idx_non_pml)
            A = np.zeros((Nb, N_non_pml), dtype=complex) ## the matrix of scaled E-field stuff
            for n in range(Nb):
                Apart = np.array(f['Function']['real_f'][str(n)]).squeeze() + 1j*np.array(f['Function']['imag_f'][str(n)]).squeeze()
                A[n,:] = Apart[idx][idx_non_pml] ## idx to order as in the mesh, non_pml to remove the pml
            del Apart ## maybe this will help with clearing memory
        gc.collect()
                
        print('all data loaded in')
        sys.stdout.flush()
        idx_ap = np.nonzero(np.abs(epsr_ref) > 1)[0] ## indices of non-air - possibly change this to work on delta epsr, for interpolating between meshes
        A_ap = A[:, np.nonzero(np.abs(epsr_ref[idx_non_pml]) > 1)[0]] ## using indices of non-air, but when already filtered for non-pml indices
        print('shape of A:', np.shape(A), f'{N} cells, {N_non_pml} non-pml cells')
        print('shape of b:', np.shape(b))
        print('in-object cells:', np.size(idx_ap))
        
        ## prepare the file
        f = dolfinx.io.XDMFFile(comm=commself, filename=problemName+'post-process.xdmf', file_mode='w')  
        f.write_mesh(mesh)
        cells.x.array[:] = epsr_ref + 0j
        f.write_function(cells, -1)
        cells.x.array[:] = epsr_dut + 0j
        f.write_function(cells, 0)
        f.close()
        
        x_temp = np.zeros(N, dtype=complex) ## to hold reconstructed epsr values
        
        #=======================================================================
        # ## test SVD thresholding, AP and non-AP
        # ##
        # f = dolfinx.io.XDMFFile(comm=commself, filename=problemName+'post-process.xdmf', file_mode='a') 
        # x_temp[idx_ap] = numpySVDfindOptimal(A_ap, b, epsr_ref[idx_ap], epsr_dut[idx_ap], cell_volumes[idx_ap])
        # cells.x.array[:] = x_temp + 0j
        # f.write_function(cells, -5) 
        # x_temp[idx_non_pml] = numpySVDfindOptimal(A, b, epsr_ref[idx_non_pml], epsr_dut[idx_non_pml], cell_volumes[idx_non_pml])
        # cells.x.array[:] = x_temp + 0j
        # f.write_function(cells, -6)
        # f.close()
        # ##
        # ##
        #=======================================================================
        
        ## test other solver settings
        ##
        testSolverSettings(A_ap, b, epsr_ref[idx_ap], epsr_dut[idx_ap], cell_volumes[idx_ap])
        exit()
        ##
        ##
        
        print('Solving with scalapack least squares and numpy svd...')
        ## non a-priori
        #A_inv = np.linalg.pinv(A, rcond=1e-3)
        b_now = np.array(np.zeros((np.size(b), 1)), order = 'F') ## not sure how much of this is necessary, but reshape it to fit what scalapack expects
        b_now[:, 0] = b
        x_temp = np.zeros(N, dtype=complex)
        x_temp[idx_non_pml] = scalapackLeastSquares(comm, MPInum, A, b_now, True) ## only the master process gets the result
    else: ## on other processes, call it with nothing
        scalapackLeastSquares(comm, MPInum)
        
    ## a priori lsq
    if(comm.rank == 0):
        x_ap = np.zeros(N, dtype=complex)
        x_ap[idx_ap] = scalapackLeastSquares(comm, MPInum, A_ap, b_now, True) ## only the master process gets the result
        
        print('done scalapack + numpy solution')
    else: ## on other processes, call it with nothing
        scalapackLeastSquares(comm, MPInum)
        
    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2 ## should give max. RSS for the process in GB - possibly this is slightly less than the memory required
    mems = comm.gather(mem_usage, root=0)
    if( comm.rank == 0 ):
        totalMem = sum(mems) ## keep the total usage. Only the master rank should be used, so this should be fine
        print(f'Current max. memory usage: {totalMem:.2e} GB, {mem_usage:.2e} for the master process')
    
    if(comm.rank == 0):
        ## write back the result
        f = dolfinx.io.XDMFFile(comm=commself, filename=problemName+'post-process.xdmf', file_mode='a')
        ## non a-priori
        cells.x.array[:] = x_ap + 0j
        f.write_function(cells, 1)  
        ## a-priori
        cells.x.array[:] = x_temp + 0j
        f.write_function(cells, 2)    
        
        f.close() ## in case one of the solution methods ends in an error, close and reopen after each method
        
        print('Scalapack done, computing numpy solutions...')
        rcond = 10**-1.8 ## based on some quick tests, an optimum is somewhere between 10**-1.2 and 10**-2.5
        f = dolfinx.io.XDMFFile(comm=commself, filename=problemName+'post-process.xdmf', file_mode='a')
        
        x_temp[idx_ap] = np.linalg.pinv(A_ap, rcond = rcond) @ b
        cells.x.array[:] = x_temp + 0j
        f.write_function(cells, 3)
        x_temp[idx_non_pml] = np.linalg.pinv(A, rcond = rcond) @ b
        cells.x.array[:] = x_temp + 0j
        f.write_function(cells, 4)  
        
        f.close()
        
        print('Solving with spgl...') ## this method is only implemented for real numbers, to make a large real matrix (hopefully this does not run me out of memory)
        sigma = 1e-5 ## guess for a good sigma
        tau = 2e-1 ## guess for a good tau
        iter_lim = 9366
          
        f = dolfinx.io.XDMFFile(comm=commself, filename=problemName+'post-process.xdmf', file_mode='a') ## 'a' is append mode? to add more functions, hopefully
        ## a-priori
        A1, A2 = np.shape(A_ap)[0], np.shape(A_ap)[1]
        Ak_ap = np.zeros((A1*2, A2*2)) ## real A
        Ak_ap[:A1,:A2] = np.real(A_ap) ## A11
        Ak_ap[:A1,A2:] = -1*np.imag(A_ap) ## A12
        Ak_ap[A1:,:A2] = 1*np.imag(A_ap) ## A21
        Ak_ap[A1:,A2:] = np.real(A_ap) ## A22
        bk = np.hstack((np.real(b),np.imag(b))) ## real b
         
        xsol, resid, grad, info = spgl1.spg_bp(Ak_ap, bk, iter_lim=iter_lim, verbosity=1)
        x_temp = np.zeros(N, dtype=complex)
        x_temp[idx_ap] = xsol[:A2] + 1j*xsol[A2:]
        cells.x.array[:] = x_temp + 0j
        f.write_function(cells, 13)
          
        xsol, resid, grad, info = spgl1.spg_bpdn(Ak_ap, bk, iter_lim=iter_lim, sigma=sigma, verbosity=1)
        x_temp[idx_ap] = xsol[:A2] + 1j*xsol[A2:]
        cells.x.array[:] = x_temp + 0j
        f.write_function(cells, 14)
          
        xsol, resid, grad, info = spgl1.spg_lasso(Ak_ap, bk, iter_lim=iter_lim, tau=tau, verbosity=1)
        x_temp[idx_ap] = xsol[:A2] + 1j*xsol[A2:]
        cells.x.array[:] = x_temp + 0j
        f.write_function(cells, 15)
         
        f.close()
         
        iter_lim = 2366
        f = dolfinx.io.XDMFFile(comm=commself, filename=problemName+'post-process.xdmf', file_mode='a') ## 'a' is append mode? to add more functions, hopefully
        ## non a-priori
        A1, A2 = np.shape(A)[0], np.shape(A)[1]
        Ak = np.zeros((A1*2, A2*2)) ## real A
        Ak[:A1,:A2] = np.real(A) ## A11
        Ak[:A1,A2:] = -1*np.imag(A) ## A12
        Ak[A1:,:A2] = 1*np.imag(A) ## A21
        Ak[A1:,A2:] = np.real(A) ## A22
         
        xsol, resid, grad, info = spgl1.spg_bp(Ak, bk, iter_lim=iter_lim, verbosity=1)
        x_temp = np.zeros(N, dtype=complex)
        x_temp[idx_non_pml] = xsol[:A2] + 1j*xsol[A2:]
        cells.x.array[:] = x_temp + 0j
        f.write_function(cells, 16)
         
        xsol, resid, grad, info = spgl1.spg_bpdn(Ak, bk, iter_lim=iter_lim, sigma=sigma, verbosity=1)
        x_temp[idx_non_pml] = xsol[:A2] + 1j*xsol[A2:]
        cells.x.array[:] = x_temp + 0j
        f.write_function(cells, 17)
          
        xsol, resid, grad, info = spgl1.spg_lasso(Ak, bk, iter_lim=iter_lim, tau=tau, verbosity=1)
        x_temp[idx_non_pml] = xsol[:A2] + 1j*xsol[A2:]
        cells.x.array[:] = x_temp + 0j
        f.write_function(cells, 18)
        f.close()
          
        del Ak ## maybe this will help with clearing memory
        gc.collect()
         
        print('done spgl solution')
        
    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2 ## should give max. RSS for the process in GB - possibly this is slightly less than the memory required
    mems = comm.gather(mem_usage, root=0)
    if( comm.rank == 0 ):
        totalMem = sum(mems) ## keep the total usage. Only the master rank should be used, so this should be fine
        print(f'Current max. memory usage: {totalMem:.2e} GB, {mem_usage:.2e} for the master process')
        
        print()
        print()
        print('solving with cvxpy...')
        t_cvx = timer()

        f = dolfinx.io.XDMFFile(comm=commself, filename=problemName+'post-process.xdmf', file_mode='a') ## 'a' is append mode? to add more functions, hopefully
        ## a-priori
        
        x_temp = np.zeros(N, dtype=complex)
        x_temp[idx_ap] = cvxpySolve(A_ap, b, 0, cell_volumes=cell_volumes[idx_ap])
        cells.x.array[:] = x_temp + 0j
        f.write_function(cells, 5)
        
        x_temp[idx_ap] = cvxpySolve(A_ap, b, 1, cell_volumes=cell_volumes[idx_ap])
        cells.x.array[:] = x_temp + 0j
        f.write_function(cells, 6)

        x_temp[idx_ap] = cvxpySolve(A_ap, b, 2, cell_volumes=cell_volumes[idx_ap])
        cells.x.array[:] = x_temp + 0j
        f.write_function(cells, 7)

        x_temp[idx_ap] = cvxpySolve(A_ap, b, 3, cell_volumes=cell_volumes[idx_ap])
        cells.x.array[:] = x_temp + 0j
        f.write_function(cells, 8)
        f.close()
        
        f = dolfinx.io.XDMFFile(comm=commself, filename=problemName+'post-process.xdmf', file_mode='a') ## 'a' is append mode? to add more functions, hopefully
        ## then non a-priori:
        
        x_temp[idx_non_pml] = cvxpySolve(A, b, 0, cell_volumes=cell_volumes[idx_non_pml])
        cells.x.array[:] = x_temp + 0j
        f.write_function(cells, 9)
        
        x_temp[idx_non_pml] = cvxpySolve(A, b, 1, cell_volumes=cell_volumes[idx_non_pml])
        cells.x.array[:] = x_temp + 0j
        f.write_function(cells, 10)

        x_temp[idx_non_pml] = cvxpySolve(A, b, 2, cell_volumes=cell_volumes[idx_non_pml])
        cells.x.array[:] = x_temp + 0j
        f.write_function(cells, 11)

        x_temp[idx_non_pml] = cvxpySolve(A, b, 3, cell_volumes=cell_volumes[idx_non_pml])
        cells.x.array[:] = x_temp + 0j
        f.write_function(cells, 12)
        f.close()
        
        print(f'done cvxpy solution, in {timer()-t_cvx:.2f} s')
        
    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2 ## should give max. RSS for the process in GB - possibly this is slightly less than the memory required
    mems = comm.gather(mem_usage, root=0)
    if( comm.rank == 0 ):
        totalMem = sum(mems) ## keep the total usage. Only the master rank should be used, so this should be fine
        print(f'Current max. memory usage: {totalMem:.2e} GB, {mem_usage:.2e} for the master process')