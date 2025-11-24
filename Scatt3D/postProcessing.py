# encoding: utf-8
## this file will have much of the postprocessing and associated functions - should only require the datafiles saved from scatteringProblem

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


if not hasattr(np.lib, "isreal"): ## spgl1 calls np.lib.isreal, which apparently no longer exists - mamba installation doesnt seem to have the newer spgl1
    np.lib.isreal = np.isreal


def cvxpySolve(A, b, problemType, solver='CLARABEL', cell_volumes=None, sigma=1e-4, tau=5e4, mu=1e0, verbose=False, solve_settings={}): ## put this in a function to allow gc?
    if(solver=='CLARABEL'): ## some settings for this that maybe make a marginal difference in solve time
        solve_settings = {'max_step_fraction': 0.95, 'tol_ktratio': 1e-5, 'tol_gap_abs': 1e-6, **solve_settings}
    N_x = np.shape(A)[1]
    x_cvxpy = cp.Variable(N_x, complex=True)
    if(problemType==0):
        objective_1norm = cp.Minimize(cp.norm(A @ x_cvxpy - b, p=1))
        problem_cvxpy = cp.Problem(objective_1norm)
    elif(problemType==1):
        objective_2norm = cp.Minimize(cp.norm(A @ x_cvxpy - b, p=2))
        problem_cvxpy = cp.Problem(objective_2norm)
    elif(problemType==2): ## bpdn
        if(cell_volumes is None): ## can normalize with cell volumes, or not
            objective_bpdn = cp.Minimize(cp.norm(x_cvxpy, p=1))
        else:
            objective_bpdn = cp.Minimize(cp.norm(cp.multiply(x_cvxpy, cell_volumes), p=1))
        constraint_bpdn = [cp.norm(A @ x_cvxpy - b, p=2) <= sigma]
        problem_cvxpy = cp.Problem(objective_bpdn, constraint_bpdn)
    elif(problemType==3): ## lasso
        objective_2norm = cp.Minimize(cp.norm(A @ x_cvxpy - b, p=2))
        if(cell_volumes is None): ## can normalize with cell volumes, or not
            constraint_lasso = [cp.norm(x_cvxpy, p=1) <= tau]
        else:
            constraint_lasso = [cp.norm(cp.multiply(x_cvxpy, cell_volumes), p=1) <= tau]
        problem_cvxpy = cp.Problem(objective_2norm, constraint_lasso)
    elif(problemType==4): ## tikhonov
        objective_tik = cp.Minimize(cp.norm(A @ x_cvxpy - b, p=2) + mu*cp.norm(x_cvxpy, p=2))
        problem_cvxpy = cp.Problem(objective_tik)
        
    try:
        problem_cvxpy.solve(verbose=verbose, solver=solver, **solve_settings)
        if(verbose):
            print(f'cvxpy norm of residual: {cp.norm(A @ x_cvxpy - b, p=2).value} ({problemType=})')
        return x_cvxpy.value
    except Exception as error: ## if the solver 'makes insufficient progress' or something, just skip
        print('\033[31m' + 'Warning: solver failed' + '\033[0m', error)
        return np.zeros(N_x)
    
def reconstructionError(delta_epsr_rec, epsr_ref, epsr_dut, cell_volumes, indices='defect', printIt=False):
    '''
    Find the 'error' figure of merit in reconstructed delta eps_r
    :param delta_epsr_rec: The reconstructed delta eps_r
    :param epsr_ref: Reference eps_r
    :param epsr_dut: Dut eps_r
    :param cell_volumes: Volume of each cell - results are normalized by this
    :param indices: The indices to reconstruct. If '', all. If 'defect', just the defect cells. If 'ap', just the object. If 'non-pml', the non-pml indices. Not needed if the epsrs are already chosen... and most of these options are not implemented yet
    :param printIt: if True, print the error
    '''
    delta_epsr_actual = epsr_dut-epsr_ref
    if(indices=='defect'):
        idx = np.nonzero(np.abs(delta_epsr_actual) != 0)[0] ## should just be the defect cells
        idxNonDef = np.nonzero(np.abs(delta_epsr_actual) == 0)[0]
        noiseVol = np.sum(cell_volumes[idxNonDef])
        noiseError = np.mean(np.abs(delta_epsr_rec[idxNonDef]/np.mean(np.abs(delta_epsr_rec[idxNonDef]) + 1e-9) - delta_epsr_actual[idxNonDef]/np.mean(np.abs(delta_epsr_actual[idxNonDef]) + 1e-9)) * cell_volumes[idxNonDef])/noiseVol ## add a 'noise' term
    else:
        idx = np.nonzero(epsr_ref != -999)[0] ## should be all indices... not sure how else to write this
    delta_epsr_actual = delta_epsr_actual[idx]
    delta_epsr_rec = delta_epsr_rec[idx]
    epsr_ref = epsr_ref[idx]
    epsr_dut = epsr_dut[idx]
    cell_volumes = cell_volumes[idx]
    
    #error = np.mean(np.abs(delta_epsr_rec - delta_epsr_actual) * cell_volumes)/np.sum(cell_volumes)
    #error = np.mean(np.abs(delta_epsr_rec/np.mean(delta_epsr_rec + 1e-9) - delta_epsr_actual/np.mean(delta_epsr_actual + 1e-9)) * cell_volumes)/np.sum(cell_volumes) ## try to account for the reconstruction being innacurate in scale, if still somewhat accurate in shape... otherwise a near-zero reconstruction looks good
    error = np.mean(np.abs(delta_epsr_rec/np.mean(np.abs(delta_epsr_rec) + 1e-16) - delta_epsr_actual/np.mean(np.abs(delta_epsr_actual) + 1e-16)) * cell_volumes)/np.sum(cell_volumes)
    #error = np.sum(np.abs(np.real(delta_epsr_rec - delta_epsr_actual)) * cell_volumes/np.sum(cell_volumes))
    #error = np.sum(np.abs(np.real(delta_epsr_rec - delta_epsr_actual))**2 * cell_volumes/np.sum(cell_volumes))**(1/2)
    if(indices=='defect'):
        error = error + noiseError/5
        
    zeroError = np.mean(np.abs(delta_epsr_actual/np.mean(np.abs(delta_epsr_actual) + 1e-16)) * cell_volumes)/np.sum(cell_volumes)
    
    error = np.sum(np.abs(delta_epsr_rec - delta_epsr_actual)*cell_volumes)
    zeroError = np.sum(np.abs(delta_epsr_actual)*cell_volumes)
    
    
    error = error/zeroError ## normalize so a guess of delta epsr = 0 gives an error of 1
    
    if(printIt):
        print(f'Reconstruction error: {error:.3e}')
    if(np.allclose(delta_epsr_rec, 0)):
        error = 1
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
        x_try = np.dot(A_inv, b)[0, :] ## set the reconstructed indices
        err = reconstructionError(x_try, epsr_ref, epsr_dut, cell_volumes)
        if verbose: ## so I don't spam the console
            print(f'Error calculated as {err:.3e}, svd_t = 10**-{svd_t}')
        if(returnSol):
            return x_try
        else:
            return np.log10(err)
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
    
    #===========================================================================
    # ## cvxpy solver tests ## seems I should use CLARABEL - they all give similar results, but CLARABEL is faster. perhaps GLPK for problem 0, which it actually works on
    # testName = 'cvxpy_type1_testsolvers'
    # for solver in cp.installed_solvers():
    #     settings.append( {'solver': solver, 'problemType': 1} )
    #===========================================================================
        
    #===========================================================================
    # ## CLARABEL settings tests
    # testName = 'cvxpy_CLARABEL_settingstest' ## testing this, I see almost no difference between any runs. Presumably I could make the tolerances larger and accept a worse result... just picked the marginally fastest settings
    # for tolgab in [1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]:
    #     for tolkt in [1e-5, 1e-4, 1e-6, 1e-7]:
    #         for stepFrac in [.97, .98, .99, .995]:
    #             for directKKT in [{}, {'direct_kkt_solver': True}]:
    #                 solvsetts = {'tol_gap_abs': tolgab, 'tol_ktratio': tolkt, 'max_step_fraction': stepFrac, **directKKT}
    #                 settings.append( {'solver': 'CLARABEL', 'problemType': 1, 'solve_settings': solvsetts} )
    #===========================================================================
    
    #===========================================================================
    # ## cvxpy sigma test
    # testName = 'cvxpy_sigmatest'
    # for sigma in [-1, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 1e-5, 1e-6, 1e-7, 1e-8]:
    #     settings.append( {'problemType': 2, 'sigma': sigma} )
    #===========================================================================
        
    #===========================================================================
    # ## cvxpy tau test
    # testName = 'cvxpy_tautest'
    # for tau in [1e4, 1e3, 1e2, 1e1, 1, 5, 50, 5e-1, 500, 5000, 1e5]:
    #     settings.append( {'problemType': 3, 'tau': tau} )
    #===========================================================================
        
    #===========================================================================
    # ## cvxpy mu test
    # testName = 'cvxpy_mutest' ## every value of mu above 1e-4 just gave a zero result, and below 1e-4 were worse than a zero
    # for mu in [1e4, 1e3, 1e2, 1e1, 1, 5, 50, 5e-1, 500, 5000, 1e5, 1e-1, 1e-2, 1e-4, 1e-6]:
    #     settings.append( {'problemType': 4, 'mu': mu} )
    #===========================================================================
    
    #===========================================================================
    # ## spgl settings tests ## it seems iterations after the first few hundred don't have a huge effect, at least for the a-priori case. Using the complex values also seems irrelevant, and the best results have large taus
    # testName = 'spgl_settingstest'
    # for tausigma in [{}, {'tau': 1e-2}, {'tau': 1}, {'tau': 1e2}, {'tau': 1e4}, {'tau': 1e6}, {'sigma': 1e-2}, {'sigma': 1e-4}, {'sigma': 1e-6}, {'sigma': 1e-8}]:
    #     for iters in [500, 1000, 9000]:
    #         for prevs in [1, 3, 10]:
    #             for complexNs in [True, False]:
    #                 settings.append( {'iter_lim': iters, 'n_prev_vals': prevs, 'iscomplex': complexNs, **tausigma} )
    #===========================================================================
        
    #===========================================================================
    # ## spgl sigma test ## best results seems to come with many iterations, sigma between 1e-2 and 1e-5, 4 or 10 nprevs, and any x0
    # testName = 'spgl_sigmatest'
    # for sigma in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 1e-5, 1e-6, 1e-7, 1e-8]:
    #     for iters in [250, 500, 1000, 2500, 9000]:
    #         for prevs in [1, 4, 10]:
    #             for x0 in [np.hstack((epsr_ref, epsr_ref)), np.hstack((np.zeros(np.shape(epsr_ref)), np.zeros(np.shape(epsr_ref)))), np.hstack((np.ones(np.shape(epsr_ref)), np.ones(np.shape(epsr_ref))))]:
    #                 settings.append( {'sigma': sigma, 'iter_lim': iters, 'n_prev_vals': prevs, 'iscomplex': True, 'x0': x0} )
    #===========================================================================
                    
    #===========================================================================
    # ## spgl tau test ## tau between 1e4 and 1e5, start with x0 of ones so it has to go down, small iter lim and n_prev_vals ?
    # testName = 'spgl_tautest'
    # for tau in [1e4, 1e3, 1e2, 1e1, 1, 5, 50, 5e-1, 500, 5000, 1e5]:
    #     for iters in [250, 500, 1000, 2500, 9000]:
    #         for prevs in [1, 3, 7, 10]:
    #             for x0 in [np.hstack((epsr_ref, epsr_ref)), np.hstack((np.zeros(np.shape(epsr_ref)), np.zeros(np.shape(epsr_ref)))), np.hstack((np.ones(np.shape(epsr_ref)), np.ones(np.shape(epsr_ref))))]:
    #                 settings.append( {'tau': tau, 'iter_lim': iters, 'n_prev_vals': prevs, 'iscomplex': True, 'x0': x0} )
    #===========================================================================
           
    num = len(settings)
    for i in range(num):
        print(f'Settings {i}:', settings[i])
        
    omegas = np.arange(num) ## Number of the setting being varied
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
                process_mem = max(proc.memory_info().rss/1024**3, process_mem) ## get max mem
                time.sleep(0.4362)
        
        try:
            t = threading.Thread(target=getMem)
            t.start()
            starTime = timer()
            
            #xsol, resid, grad, info = spgl1.spgl1(A, b, verbosity=1, **settings[i]) ## for spgl
            #size = np.size(epsr_ref)
            #x_rec = xsol[:size] + 1j*xsol[size:]
            
            x_rec = cvxpySolve(A, b, cell_volumes=cell_volumes, verbose=True, **settings[i]) ## for cvxpy
            
            ts[i] = timer() - starTime
            errors[i] = reconstructionError(x_rec, epsr_ref, epsr_dut, cell_volumes)
            mems[i] = process_mem
            print(f'Run completed with memory: {mems[i]:3f} GB, error: {errors[i]:2e} in: {ts[i]:2f} s')
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
    plt.savefig('data3D/'+testName+'_post-processing_solversettingsplot.png')
    
    nprint = min(num, 10)
    print(f'Top {nprint} Options #s:') ## lowest errors
    idxsort = np.argsort(errors)
    for k in range(nprint):
        print(f'#{idxsort[k]+1}: t={ts[idxsort[k]]:.3e}, error={errors[idxsort[k]]:3e}, mem={mems[idxsort[k]]:.2f}GB --- ')
        print(settings[idxsort[k]])
        print()
    
    #plt.show()

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
        
def addAmplitudePhaseNoise(Ss, amp, phase, random=True): ## add relative amplitude and/or absolute phase noise to the scattering parameters, to see how it affects the reconstruction. If not random, just offset all parameters
    amp = 1+amp ## so it's relative amplitude
    if(random): ## normal distribution
        Ss = Ss*np.exp(1j*np.random.normal(0, phase, np.shape(Ss)))*np.random.normal(0, amp, np.shape(Ss))
    else:
        Ss = Ss*np.exp(1j*phase)*amp
    return Ss

def solveFromQs(problemName, solutionName='', antennasToUse=[], frequenciesToUse=[], onlyNAntennas=0, onlyAPriori=True, returnResults=[]):
    '''
    Try various solution methods... keeping everything on one process
    :param problemName: The filename, used to find and save files
    :param solutionName: Name to be appended to the solution files - default is nothing
    :param antennasToUse: Use only data from these antennas - list of their indices. If empty (default), use all
    :param frequenciesToUse: Use only data from these frequencies - list of their indices. If empty (default), use all
    :param onlyNAntennas: Use indices such that it is like we only had N antennas to measure with. If 0, uses all data
    :param onlyAPriori: only perform the a-priori reconstruction, using just the object's cells. This is to keep the matrix so small it can be computed in memory
    :param returnResults: List of timesteps to compute + return the error from - if empty, this is ignored
    '''
    gc.collect()
    comm = MPI.COMM_WORLD
    commself = MPI.COMM_SELF
    if(comm.rank == 0):
        print(f'Postprocessing of {problemName}, {solutionName} starting:')
    
        ## load in all the data
        data = np.load(problemName+'output.npz')
        b = data['b']
        fvec = data['fvec']
        S_ref = data['S_ref']
        S_dut = data['S_dut']
        #epsr_mat = data['epsr_mat']
        #epsr_defect = data['epsr_defect']
        N_antennas = data['N_antennas']
        antenna_radius = data['antenna_radius'] ## radius at which the antennas are placed
        Nf = len(fvec)
        Np = S_ref.shape[-1]
        
        #=======================================================================
        # ### PLOT S11 STUFF
        # plt.plot(fvec/1e9, 20*np.log10(np.abs(S_ref.flatten())), label='FEM sim') ## try plotting the Ss
        # #plt.plot(np.abs(S_dut.flatten()))
        # fekof = 'TestStuff/FEKO patch S11.dat'
        # fekoData = np.transpose(np.loadtxt(fekof, skiprows = 2))
        # plt.plot(fekoData[0]/1e9, 20*np.log10(np.abs(fekoData[1]+1j*fekoData[2])), label='FEKO')
        # plt.plot()
        # plt.grid()
        # plt.ylabel(r'S$_{11}$ [dB]')
        # plt.xlabel(r'Frequency [GHz]')
        # plt.title(r'Simulated vs FEKO S$_{11}$ Mag.')
        # plt.legend()
        # plt.show()
        # ## then plot the phase of S11, also
        # plt.plot(fvec/1e9, np.angle(S_ref.flatten()), label='FEM sim')
        # plt.plot(fekoData[0]/1e9, np.angle(fekoData[1]+1j*fekoData[2]), label='FEKO')
        # plt.plot(fvec/1e9, np.angle(S_ref.flatten()) + (np.angle(fekoData[1]+1j*fekoData[2])[0]-np.angle(S_ref.flatten())[0]) , label='FEM sim (matched)')
        # plt.grid()
        # plt.ylabel(r'Phase of S$_{11}$ [radians]')
        # plt.xlabel(r'Frequency [GHz]')
        # plt.title(r'Simulated vs FEKO S$_{11}$ Phase')
        # plt.legend()
        # plt.show()
        #=======================================================================
        
        Nb = len(b) ## number of rows, or 'data points' to be used
        
        ## mesh stuff on just one process?
        with dolfinx.io.XDMFFile(commself, problemName+'output-qs.xdmf', 'r') as f:
            mesh = f.read_mesh()
            Wspace = dolfinx.fem.functionspace(mesh, ('DG', 0))
            cells = dolfinx.fem.Function(Wspace)
            
            idxOrig = mesh.topology.original_cell_index ## map from indices in the original cells to the current mesh cells
            dofs = Wspace.dofmap.index_map ## has info about dofs on each process
            
        
        ## load in the problem data
        print('data loaded in')
        sys.stdout.flush()
        
        with h5py.File(problemName+'output-qs.h5', 'r') as f: ## this is serial, so only needs to occur on the main process
            dofs_map = np.array(f['Function']['real_f']['-4']).squeeze()[idxOrig] ## f being the default name of the function as seen in paraview
            cell_volumes = np.array(f['Function']['real_f']['-3']).squeeze()[idxOrig]
            epsr_ref = np.array(f['Function']['real_f']['-2']).squeeze()[idxOrig] + 1j*np.array(f['Function']['imag_f']['-2']).squeeze()[idxOrig]
            epsr_dut = np.array(f['Function']['real_f']['-1']).squeeze()[idxOrig] + 1j*np.array(f['Function']['imag_f']['-1']).squeeze()[idxOrig]
            N = len(cell_volumes)
            
            #idx_non_pml = np.nonzero(np.real(dofs_map) > -1)[0] ## PML cells should have a value of -1 - can use everything else as the indices for non a-priori reconstructions
            
            midpoints = dolfinx.mesh.compute_midpoints(mesh, mesh.topology.dim, np.arange(N, dtype=np.int32)) ## midpoints of every cell
            dist = np.linalg.norm(midpoints, axis=1)
            idx_non_pml = np.nonzero(dist < antenna_radius*0.7)[0] ## alternative reconstruction cells - those within a sphere of the centre
            
            ## choose which row/non-cell indices to use for the reconstruction - i.e. which frequencies/antennas
            idxNC = []
            
            for nf in range(Nf): ## frequency
                if(not (nf in frequenciesToUse or len(frequenciesToUse)==0)):
                    continue ## skip to next index
                for m in range(N_antennas): ## transmitting index
                    for n in range(N_antennas): ## receiving index
                        if( not ( (m in antennasToUse and n in antennasToUse) or len(antennasToUse)==0 ) ):
                            continue ## skip to next index
                        i = nf*N_antennas*N_antennas + m*N_antennas + n
                        
                        if(onlyNAntennas > 0): ## use only transmission to the N antennas that are most spread out (as if we only had N antennas to measure with) This includes the antenna itself - if N is 1, then there is only reflection
                            dist = int(N_antennas/onlyNAntennas)+1 ## index-distance between used antennas
                            if( np.abs(n-m)%dist != 0 ):
                                continue ## skip to next index
                                
                        if(True): ## check for a low reflection coefficient - if it is too high, this frequency is likely not good
                            refl = np.abs(S_ref[nf, m, m]) ## should be close enough between ref and dut cases
                            #print(refl) 
                            
                        idxNC.append(i) ## if the checks are passed, use this index
                        
            b = b[idxNC]
            
            N_non_pml = len(idx_non_pml)
            A = np.zeros((len(idxNC), N_non_pml), dtype=complex) ## the matrix of scaled E-field stuff
            indexCount = 0
            for nf in range(Nf):
                for m in range(N_antennas): ## transmitting index
                    for n in range(N_antennas): ## receiving index
                        i = nf*N_antennas*N_antennas + m*N_antennas + n
                        if(i in idxNC):
                            Apart = np.array(f['Function']['real_f'][str(i)]).squeeze() + 1j*np.array(f['Function']['imag_f'][str(i)]).squeeze()
                            A[indexCount,:] = Apart[idxOrig][idx_non_pml] ## idxOrig to order as in the mesh, non_pml to remove the pml
                            indexCount +=1
            del Apart ## maybe this will help with clearing memory
        gc.collect()
        
        print('all data loaded in')
        sys.stdout.flush()
        
        #idx_ap = np.nonzero(np.abs(epsr_ref) > 1)[0] ## indices of non-air - possibly change this to work on delta epsr, for interpolating between meshes
        idx_ap = np.nonzero(np.real(dofs_map) > 1)[0] ## basing it on the dofs map should be better, considering the possibility of a different DUT mesh
        A_ap = A[:, np.nonzero(np.real(dofs_map[idx_non_pml]) > 1)[0]] ## using indices of non-air, but when already filtered for non-pml indices
        print('shape of A:', np.shape(A), f'{N} cells, {N_non_pml} non-pml cells')
        print('shape of b:', np.shape(b))
        print('in-object cells:', np.size(idx_ap))
        sys.stdout.flush()
        
        ## prepare the solution/output file
        solutionFile = problemName+'post-process'+solutionName+'.xdmf'
        f = dolfinx.io.XDMFFile(comm=commself, filename=solutionFile, file_mode='w')  
        f.write_mesh(mesh)
        cells.x.array[:] = epsr_ref + 0j
        f.write_function(cells, -2)
        cells.x.array[:] = epsr_dut + 0j
        f.write_function(cells, -1)
        cells.x.array[:] = epsr_dut-epsr_ref + 0j
        f.write_function(cells, 0)
        f.close()
        
        x_temp = np.zeros(N, dtype=complex) ## to hold reconstructed epsr values
        
        ## optimize for various solver settings here
        
        #=======================================================================
        # ## optimize spgl variables
        # sigma = 1e-4 ## guess for a good sigma
        # tau = 6e4 ## guess for a good tau
        # iter_lim = 5366
        # spgl_settings = {'iter_lim': iter_lim, 'n_prev_vals': 10, 'iscomplex': True, 'verbosity': 0}
        # 
        # f = dolfinx.io.XDMFFile(comm=commself, filename=solutionFile, file_mode='a') ## 'a' is append mode? to add more functions, hopefully
        # ## a-priori
        # A1, A2 = np.shape(A_ap)[0], np.shape(A_ap)[1]
        # Ak_ap = np.zeros((A1*2, A2*2)) ## real A
        # Ak_ap[:A1,:A2] = np.real(A_ap) ## A11
        # Ak_ap[:A1,A2:] = -1*np.imag(A_ap) ## A12
        # Ak_ap[A1:,:A2] = 1*np.imag(A_ap) ## A21
        # Ak_ap[A1:,A2:] = np.real(A_ap) ## A22
        # bk = np.hstack((np.real(b),np.imag(b))) ## real b
        # 
        # def calcAccuracy(sigma, epsr_ref, epsr_dut, cell_volumes): ### find a figure of merit (for optimizing a value)
        #     xsol, resid, grad, info = spgl1.spgl1(Ak_ap, bk, sigma=sigma, **spgl_settings)
        #     x_try = xsol[:A2] + 1j*xsol[A2:]
        #     err = reconstructionError(x_try, epsr_ref, epsr_dut, cell_volumes)
        #     print(f'Error calculated as {err:.3e}, {sigma=}')
        #     return np.log10(err)
        # optGoal = functools.partial(calcAccuracy, epsr_ref=epsr_ref[idx_ap], epsr_dut=epsr_dut[idx_ap], cell_volumes=cell_volumes[idx_ap])
        # optsol = scipy.optimize.minimize(optGoal, sigma)
        # optSigma = optsol.x
        # print(f'SPGL optimal {optSigma=}')
        # xsol, resid, grad, info = spgl1.spgl1(Ak_ap, bk, sigma=optSigma, **spgl_settings)
        # x_temp[idx_ap] = xsol[:A2] + 1j*xsol[A2:]
        # cells.x.array[:] = x_temp + 0j
        # f.write_function(cells, -24)
        # print(f'Timestep -24 reconstruction error: {reconstructionError(x_temp[idx_ap], epsr_ref[idx_ap], epsr_dut[idx_ap], cell_volumes[idx_ap]):.3e}')
        # 
        # def calcAccuracy(tau, epsr_ref, epsr_dut, cell_volumes): ### find a figure of merit (for optimizing a value)
        #     xsol, resid, grad, info = spgl1.spgl1(Ak_ap, bk, tau=tau, **spgl_settings)
        #     x_try = xsol[:A2] + 1j*xsol[A2:]
        #     err = reconstructionError(x_try, epsr_ref, epsr_dut, cell_volumes)
        #     print(f'Error calculated as {err:.3e}, {tau=}')
        #     return np.log10(err)
        # optGoal = functools.partial(calcAccuracy, epsr_ref=epsr_ref[idx_ap], epsr_dut=epsr_dut[idx_ap], cell_volumes=cell_volumes[idx_ap])
        # optsol = scipy.optimize.minimize(optGoal, tau)
        # optTau = optsol.x
        # print(f'SPGL optimal {optTau=}')
        # xsol, resid, grad, info = spgl1.spgl1(Ak_ap, bk, tau=optTau, **spgl_settings)
        # x_temp[idx_ap] = xsol[:A2] + 1j*xsol[A2:]
        # cells.x.array[:] = x_temp + 0j
        # f.write_function(cells, -25)
        # print(f'Timestep -25 reconstruction error: {reconstructionError(x_temp[idx_ap], epsr_ref[idx_ap], epsr_dut[idx_ap], cell_volumes[idx_ap]):.3e}')
        # f.close()
        #=======================================================================
        
        #=======================================================================
        # ## optimize cvxpy variables
        # sigma = 5e-2 ## guess for a good sigma
        # tau = 6e4 ## guess for a good tau
        # mu = 1e2 ## guess for a good mu
        # 
        # f = dolfinx.io.XDMFFile(comm=commself, filename=solutionFile, file_mode='a') ## 'a' is append mode? to add more functions, hopefully
        # 
        # def calcAccuracy(sigma, epsr_ref, epsr_dut, cell_volumes): ### find a figure of merit (for optimizing a value)
        #     x_try = cvxpySolve(A_ap, b, 3, sigma=sigma, cell_volumes=cell_volumes, verbose=False)
        #     err = reconstructionError(x_try, epsr_ref, epsr_dut, cell_volumes)
        #     print(f'Error calculated as {err:.3e}, {sigma=}')
        #     return np.log10(err)
        # optGoal = functools.partial(calcAccuracy, epsr_ref=epsr_ref[idx_ap], epsr_dut=epsr_dut[idx_ap], cell_volumes=cell_volumes[idx_ap])
        # optsol = scipy.optimize.minimize(optGoal, sigma)
        # optSigma = optsol.x
        # print(f'SPGL optimal {optSigma=}')
        # x_temp[idx_ap] = cvxpySolve(A_ap, b, 2, sigma=optSigma, cell_volumes=cell_volumes)
        # cells.x.array[:] = x_temp + 0j
        # f.write_function(cells, -7)
        # print(f'Timestep -7 reconstruction error: {reconstructionError(x_temp[idx_ap], epsr_ref[idx_ap], epsr_dut[idx_ap], cell_volumes[idx_ap]):.3e}')
        # 
        # def calcAccuracy(tau, epsr_ref, epsr_dut, cell_volumes): ### find a figure of merit (for optimizing a value)
        #     x_try = cvxpySolve(A_ap, b, 3, tau=tau, cell_volumes=cell_volumes, verbose=False)
        #     err = reconstructionError(x_try, epsr_ref, epsr_dut, cell_volumes)
        #     print(f'Error calculated as {err:.3e}, {tau=}')
        #     return np.log10(err)
        # optGoal = functools.partial(calcAccuracy, epsr_ref=epsr_ref[idx_ap], epsr_dut=epsr_dut[idx_ap], cell_volumes=cell_volumes[idx_ap])
        # optsol = scipy.optimize.minimize(optGoal, tau)
        # optTau = optsol.x
        # print(f'SPGL optimal {optTau=}')
        # x_temp[idx_ap] = cvxpySolve(A_ap, b, 3, tau=optTau, cell_volumes=cell_volumes)
        # cells.x.array[:] = x_temp + 0j
        # f.write_function(cells, -8)
        # print(f'Timestep -8 reconstruction error: {reconstructionError(x_temp[idx_ap], epsr_ref[idx_ap], epsr_dut[idx_ap], cell_volumes[idx_ap]):.3e}')
        # 
        # def calcAccuracy(mu, epsr_ref, epsr_dut, cell_volumes): ### find a figure of merit (for optimizing a value)
        #     x_try = cvxpySolve(A_ap, b, 3, mu=mu, cell_volumes=cell_volumes, verbose=False)
        #     err = reconstructionError(x_try, epsr_ref, epsr_dut, cell_volumes)
        #     print(f'Error calculated as {err:.3e}, {mu=}')
        #     return np.log10(err)
        # optGoal = functools.partial(calcAccuracy, epsr_ref=epsr_ref[idx_ap], epsr_dut=epsr_dut[idx_ap], cell_volumes=cell_volumes[idx_ap])
        # optsol = scipy.optimize.minimize(optGoal, mu)
        # optMu = optsol.x
        # print(f'SPGL optimal {optMu=}')
        # x_temp[idx_ap] = cvxpySolve(A_ap, b, 4, mu=optMu, cell_volumes=cell_volumes)
        # cells.x.array[:] = x_temp + 0j
        # f.write_function(cells, -9)
        # print(f'Timestep -9 reconstruction error: {reconstructionError(x_temp[idx_ap], epsr_ref[idx_ap], epsr_dut[idx_ap], cell_volumes[idx_ap]):.3e}')
        # 
        # f.close()
        #=======================================================================
        
        
        #=======================================================================
        # ## optimize SVD threshold, AP and non-AP
        # ##
        # f = dolfinx.io.XDMFFile(comm=commself, filename=solutionFile, file_mode='a')
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
        
        mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2 ## should give max. RSS for the process in GB - possibly this is slightly less than the memory required
        mems = comm.gather(mem_usage, root=0)
        if( comm.rank == 0 ):
            totalMem = sum(mems) ## keep the total usage. Only the master rank should be used, so this should be fine
            print(f'Current max. memory usage: {totalMem:.2e} GB, {mem_usage:.2e} for the master process')
        sys.stdout.flush()
            
        #return ## exit
            
        ## test other solver settings
        ##
        
        #=======================================================================
        # ## a-priori
        # A1, A2 = np.shape(A_ap)[0], np.shape(A_ap)[1]
        # Ak_ap = np.zeros((A1*2, A2*2)) ## real A
        # Ak_ap[:A1,:A2] = np.real(A_ap) ## A11
        # Ak_ap[:A1,A2:] = -1*np.imag(A_ap) ## A12
        # Ak_ap[A1:,:A2] = 1*np.imag(A_ap) ## A21
        # Ak_ap[A1:,A2:] = np.real(A_ap) ## A22
        # bk = np.hstack((np.real(b),np.imag(b))) ## real b
        # testSolverSettings(Ak_ap, bk, epsr_ref[idx_ap], epsr_dut[idx_ap], cell_volumes[idx_ap])
        #=======================================================================
        
        #=======================================================================
        # ## non a-priori
        # A1, A2 = np.shape(A)[0], np.shape(A)[1]
        # Ak = np.zeros((A1*2, A2*2)) ## real A
        # Ak[:A1,:A2] = np.real(A) ## A11
        # Ak[:A1,A2:] = -1*np.imag(A) ## A12
        # Ak[A1:,:A2] = 1*np.imag(A) ## A21
        # Ak[A1:,A2:] = np.real(A) ## A22
        # bk = np.hstack((np.real(b),np.imag(b))) ## real b
        # testSolverSettings(Ak, bk, epsr_ref[idx_non_pml], epsr_dut[idx_non_pml], cell_volumes[idx_non_pml])
        #=======================================================================
        
        #testSolverSettings(A_ap, b, epsr_ref[idx_ap], epsr_dut[idx_ap], cell_volumes[idx_ap])
        #return
        ##
        ##
        
    #===========================================================================
    #     print('Solving with scalapack least squares...') ## always has huge |x|, basically just noise
    #     ## non a-priori
    #     #A_inv = np.linalg.pinv(A, rcond=1e-3)
    #     b_now = np.array(np.zeros((np.size(b), 1)), order = 'F') ## not sure how much of this is necessary, but reshape it to fit what scalapack expects
    #     b_now[:, 0] = b
    #     x_temp = np.zeros(N, dtype=complex)
    #     if(not onlyAPriori):
    #         x_temp[idx_non_pml] = scalapackLeastSquares(comm, MPInum, A, b_now, True) ## only the master process gets the result
    # else: ## on other processes, call it with nothing
    #     if(not onlyAPriori):
    #         scalapackLeastSquares(comm, MPInum)
    #     
    # ## a priori lsq
    # if(comm.rank == 0):
    #     x_ap = np.zeros(N, dtype=complex)
    #     x_ap[idx_ap] = scalapackLeastSquares(comm, MPInum, A_ap, b_now, True) ## only the master process gets the result
    #     
    #     print('done scalapack solution')
    # else: ## on other processes, call it with nothing
    #     scalapackLeastSquares(comm, MPInum)
    #     
    # mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2 ## should give max. RSS for the process in GB - possibly this is slightly less than the memory required
    # mems = comm.gather(mem_usage, root=0)
    # if( comm.rank == 0 ):
    #     totalMem = sum(mems) ## keep the total usage. Only the master rank should be used, so this should be fine
    #     print(f'Current max. memory usage: {totalMem:.2e} GB, {mem_usage:.2e} for the master process')
    # 
    # if(comm.rank == 0):
    #     ## write back the result
    #     f = dolfinx.io.XDMFFile(comm=commself, filename=solutionFile, file_mode='a')
    #     ## a-priori
    #     cells.x.array[:] = x_ap + 0j
    #     f.write_function(cells, 1)
    #     print(f'Timestep 1 reconstruction error: {reconstructionError(x_ap[idx_ap], epsr_ref[idx_ap], epsr_dut[idx_ap], cell_volumes[idx_ap]):.3e}')
    #     ## non a-priori
    #     if(not onlyAPriori):
    #         cells.x.array[:] = x_temp + 0j
    #         f.write_function(cells, 2)    
    #         print(f'Timestep 2 reconstruction error: {reconstructionError(x_temp[idx_non_pml], epsr_ref[idx_non_pml], epsr_dut[idx_non_pml], cell_volumes[idx_non_pml]):.3e}')
    #     f.close() ## in case one of the solution methods ends in an error, close and reopen after each method
    #===========================================================================
        print()
        print('Computing numpy solutions...') ## can either optimization for rcond, or just pick one
        sys.stdout.flush()
        rcond = 10**-1.65 ## based on some quick tests, an optimum is somewhere between 10**-1.2 and 10**-2.5
        f = dolfinx.io.XDMFFile(comm=commself, filename=solutionFile, file_mode='a')
        x_temp = np.zeros(N, dtype=complex)
        
        x_temp[idx_ap] = numpySVDfindOptimal(A_ap, b, epsr_ref[idx_ap], epsr_dut[idx_ap], cell_volumes[idx_ap])#np.linalg.pinv(A_ap, rcond = rcond) @ b
        cells.x.array[:] = x_temp + 0j
        f.write_function(cells, 3)
        print(f'Timestep 3 reconstruction error: {reconstructionError(x_temp[idx_ap], epsr_ref[idx_ap], epsr_dut[idx_ap], cell_volumes[idx_ap]):.3e}')
        if(not onlyAPriori):
            x_temp[idx_non_pml] = numpySVDfindOptimal(A, b, epsr_ref[idx_non_pml], epsr_dut[idx_non_pml], cell_volumes[idx_non_pml])#np.linalg.pinv(A, rcond = rcond) @ b
            cells.x.array[:] = x_temp + 0j
            f.write_function(cells, 4)  
            print(f'Timestep 4 reconstruction error: {reconstructionError(x_temp[idx_non_pml], epsr_ref[idx_non_pml], epsr_dut[idx_non_pml], cell_volumes[idx_non_pml]):.3e}')
        
        f.close()
        print()
        print('Solving with spgl...') ## this method is only implemented for real numbers, to make a large real matrix (hopefully this does not run me out of memory)
        sys.stdout.flush()
        sigma = 1e-2 ## guess for a good sigma
        iter_lim = 5633
        spgl_settings = {'iter_lim': iter_lim, 'n_prev_vals': 10, 'iscomplex': True, 'verbosity': 0}
          
        f = dolfinx.io.XDMFFile(comm=commself, filename=solutionFile, file_mode='a') ## 'a' is append mode? to add more functions, hopefully
        ## a-priori
        A1, A2 = np.shape(A_ap)[0], np.shape(A_ap)[1]
        Ak_ap = np.zeros((A1*2, A2*2)) ## real A
        Ak_ap[:A1,:A2] = np.real(A_ap) ## A11
        Ak_ap[:A1,A2:] = -1*np.imag(A_ap) ## A12
        Ak_ap[A1:,:A2] = 1*np.imag(A_ap) ## A21
        Ak_ap[A1:,A2:] = np.real(A_ap) ## A22
        bk = np.hstack((np.real(b),np.imag(b))) ## real b
        x_temp = np.zeros(N, dtype=complex)
        errs = [] ## in case I want to return errors
        
        if(not returnResults or 23 in returnResults): ## if it is empty, or requested
            xsol, resid, grad, info = spgl1.spgl1(Ak_ap, bk, **spgl_settings)
            x_temp[idx_ap] = xsol[:A2] + 1j*xsol[A2:]
            cells.x.array[:] = x_temp + 0j
            f.write_function(cells, 23)
            err = reconstructionError(x_temp[idx_ap], epsr_ref[idx_ap], epsr_dut[idx_ap], cell_volumes[idx_ap])
            errs.append(err)
            print(f'Timestep 23 reconstruction error: {err:.3e}')
        
        if(not returnResults or 24 in returnResults): ## if it is empty, or needed
            xsol, resid, grad, info = spgl1.spgl1(Ak_ap, bk, sigma=sigma, **spgl_settings)
            x_temp[idx_ap] = xsol[:A2] + 1j*xsol[A2:]
            cells.x.array[:] = x_temp + 0j
            f.write_function(cells, 24)
            err = reconstructionError(x_temp[idx_ap], epsr_ref[idx_ap], epsr_dut[idx_ap], cell_volumes[idx_ap])
            errs.append(err)
            print(f'Timestep 24 reconstruction error: {err:.3e}')
        
        tau = 6e4 ## guess for a good tau
        iter_lim = 633
        if(not returnResults or 25 in returnResults): ## if it is empty, or needed
            xsol, resid, grad, info = spgl1.spgl1(Ak_ap, bk, tau=tau, **spgl_settings)
            x_temp[idx_ap] = xsol[:A2] + 1j*xsol[A2:]
            cells.x.array[:] = x_temp + 0j
            f.write_function(cells, 25)
            err = reconstructionError(x_temp[idx_ap], epsr_ref[idx_ap], epsr_dut[idx_ap], cell_volumes[idx_ap])
            errs.append(err)
            print(f'Timestep 25 reconstruction error: {err:.3e}')
         
        f.close()
        if(not onlyAPriori):
            iter_lim = 1366
            spgl_settings = {'iter_lim': iter_lim, 'n_prev_vals': 10, 'iscomplex': True, 'verbosity': 0}
            f = dolfinx.io.XDMFFile(comm=commself, filename=solutionFile, file_mode='a') ## 'a' is append mode? to add more functions, hopefully
            ## non a-priori
            A1, A2 = np.shape(A)[0], np.shape(A)[1]
            Ak = np.zeros((A1*2, A2*2)) ## real A
            Ak[:A1,:A2] = np.real(A) ## A11
            Ak[:A1,A2:] = -1*np.imag(A) ## A12
            Ak[A1:,:A2] = 1*np.imag(A) ## A21
            Ak[A1:,A2:] = np.real(A) ## A22
            x_temp = np.zeros(N, dtype=complex)
             
            if(not returnResults or 26 in returnResults): ## if it is empty, or needed
                xsol, resid, grad, info = spgl1.spgl1(Ak, bk, **spgl_settings)
                x_temp[idx_non_pml] = xsol[:A2] + 1j*xsol[A2:]
                cells.x.array[:] = x_temp + 0j
                f.write_function(cells, 26)
                err = reconstructionError(x_temp[idx_non_pml], epsr_ref[idx_non_pml], epsr_dut[idx_non_pml], cell_volumes[idx_non_pml])
                errs.append(err)
                print(f'Timestep 26 reconstruction error: {err:.3e}')
            
            if(not returnResults or 27 in returnResults): ## if it is empty, or needed
                xsol, resid, grad, info = spgl1.spgl1(Ak, bk, sigma=sigma, **spgl_settings)
                x_temp[idx_non_pml] = xsol[:A2] + 1j*xsol[A2:]
                cells.x.array[:] = x_temp + 0j
                f.write_function(cells, 27)
                err = reconstructionError(x_temp[idx_non_pml], epsr_ref[idx_non_pml], epsr_dut[idx_non_pml], cell_volumes[idx_non_pml])
                errs.append(err)
                print(f'Timestep 27 reconstruction error: {err:.3e}')
            
            if(not returnResults or 28 in returnResults): ## if it is empty, or needed
                xsol, resid, grad, info = spgl1.spgl1(Ak, bk, tau=tau, **spgl_settings)
                x_temp[idx_non_pml] = xsol[:A2] + 1j*xsol[A2:]
                cells.x.array[:] = x_temp + 0j
                f.write_function(cells, 28)
                err = reconstructionError(x_temp[idx_non_pml], epsr_ref[idx_non_pml], epsr_dut[idx_non_pml], cell_volumes[idx_non_pml])
                errs.append(err)
                print(f'Timestep 28 reconstruction error: {err:.3e}')
            f.close()
          
            del Ak ## maybe this will help with clearing memory
            gc.collect()
         
        print('done spgl solution')
        print()
        sys.stdout.flush()
        
    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2 ## should give max. RSS for the process in GB - possibly this is slightly less than the memory required
    mems = comm.gather(mem_usage, root=0)
    if( comm.rank == 0 ):
        totalMem = sum(mems) ## keep the total usage. Only the master rank should be used, so this should be fine
        print(f'Current max. memory usage: {totalMem:.2e} GB, {mem_usage:.2e} for the master process')
        
        if(False):
            print('solving with cvxpy...')
            sys.stdout.flush()
            t_cvx = timer()
    
            f = dolfinx.io.XDMFFile(comm=commself, filename=solutionFile, file_mode='a') ## 'a' is append mode? to add more functions, hopefully
            ## a-priori
            
            if(not returnResults or 5 in returnResults): ## if it is empty, or needed
                x_temp = np.zeros(N, dtype=complex)
                x_temp[idx_ap] = cvxpySolve(A_ap, b, 0, cell_volumes=cell_volumes[idx_ap], solver = 'GLPK') ## GLPK where it can be used - seems faster and possibly better than CLARABEL. GLPK_MI seems faster, but possibly worse
                cells.x.array[:] = x_temp + 0j
                f.write_function(cells, 5)
                err = reconstructionError(x_temp[idx_ap], epsr_ref[idx_ap], epsr_dut[idx_ap], cell_volumes[idx_ap])
                errs.append(err)
                print(f'Timestep 5 reconstruction error: {err:.3e}')
            
            if(not returnResults or 6 in returnResults): ## if it is empty, or needed
                x_temp[idx_ap] = cvxpySolve(A_ap, b, 1, cell_volumes=cell_volumes[idx_ap])
                cells.x.array[:] = x_temp + 0j
                f.write_function(cells, 6)
                err = reconstructionError(x_temp[idx_ap], epsr_ref[idx_ap], epsr_dut[idx_ap], cell_volumes[idx_ap])
                errs.append(err)
                print(f'Timestep 6 reconstruction error: {err:.3e}')
            
            if(not returnResults or 7 in returnResults): ## if it is empty, or needed
                x_temp[idx_ap] = cvxpySolve(A_ap, b, 2, cell_volumes=cell_volumes[idx_ap])
                cells.x.array[:] = x_temp + 0j
                f.write_function(cells, 7)
                err = reconstructionError(x_temp[idx_ap], epsr_ref[idx_ap], epsr_dut[idx_ap], cell_volumes[idx_ap])
                errs.append(err)
                print(f'Timestep 7 reconstruction error: {err:.3e}')
            
            if(not returnResults or 8 in returnResults): ## if it is empty, or needed
                x_temp[idx_ap] = cvxpySolve(A_ap, b, 3, cell_volumes=cell_volumes[idx_ap])
                cells.x.array[:] = x_temp + 0j
                f.write_function(cells, 8)
                err = reconstructionError(x_temp[idx_ap], epsr_ref[idx_ap], epsr_dut[idx_ap], cell_volumes[idx_ap])
                errs.append(err)
                print(f'Timestep 8 reconstruction error: {err:.3e}')
            
            if(not returnResults or 9 in returnResults): ## if it is empty, or needed
                x_temp[idx_ap] = cvxpySolve(A_ap, b, 4, cell_volumes=cell_volumes[idx_ap])
                cells.x.array[:] = x_temp + 0j
                f.write_function(cells, 9)
                err = reconstructionError(x_temp[idx_ap], epsr_ref[idx_ap], epsr_dut[idx_ap], cell_volumes[idx_ap])
                errs.append(err)
                print(f'Timestep 9 reconstruction error: {err:.3e}')
            
            f.close()
            
            if(not onlyAPriori): ## then non a-priori:
                f = dolfinx.io.XDMFFile(comm=commself, filename=solutionFile, file_mode='a') ## 'a' is append mode? to add more functions, hopefully
                
                if(not returnResults or 10 in returnResults): ## if it is empty, or needed
                    x_temp[idx_non_pml] = cvxpySolve(A, b, 0, cell_volumes=cell_volumes[idx_non_pml], solver = 'GLPK')
                    cells.x.array[:] = x_temp + 0j
                    f.write_function(cells, 10)
                    err = reconstructionError(x_temp[idx_non_pml], epsr_ref[idx_non_pml], epsr_dut[idx_non_pml], cell_volumes[idx_non_pml])
                    errs.append(err)
                    print(f'Timestep 10 reconstruction error: {err:.3e}')
                
                if(not returnResults or 11 in returnResults): ## if it is empty, or needed
                    x_temp[idx_non_pml] = cvxpySolve(A, b, 1, cell_volumes=cell_volumes[idx_non_pml])
                    cells.x.array[:] = x_temp + 0j
                    f.write_function(cells, 11)
                    err = reconstructionError(x_temp[idx_non_pml], epsr_ref[idx_non_pml], epsr_dut[idx_non_pml], cell_volumes[idx_non_pml])
                    errs.append(err)
                    print(f'Timestep 11 reconstruction error: {err:.3e}')
                
                if(not returnResults or 12 in returnResults): ## if it is empty, or needed
                    x_temp[idx_non_pml] = cvxpySolve(A, b, 2, cell_volumes=cell_volumes[idx_non_pml])
                    cells.x.array[:] = x_temp + 0j
                    f.write_function(cells, 12)
                    err = reconstructionError(x_temp[idx_non_pml], epsr_ref[idx_non_pml], epsr_dut[idx_non_pml], cell_volumes[idx_non_pml])
                    errs.append(err)
                    print(f'Timestep 12 reconstruction error: {reconstructionError(x_temp[idx_non_pml], epsr_ref[idx_non_pml], epsr_dut[idx_non_pml], cell_volumes[idx_non_pml]):.3e}')
                
                if(not returnResults or 13 in returnResults): ## if it is empty, or needed
                    x_temp[idx_non_pml] = cvxpySolve(A, b, 3, cell_volumes=cell_volumes[idx_non_pml])
                    cells.x.array[:] = x_temp + 0j
                    f.write_function(cells, 13)
                    err = reconstructionError(x_temp[idx_non_pml], epsr_ref[idx_non_pml], epsr_dut[idx_non_pml], cell_volumes[idx_non_pml])
                    errs.append(err)
                    print(f'Timestep 13 reconstruction error: {reconstructionError(x_temp[idx_non_pml], epsr_ref[idx_non_pml], epsr_dut[idx_non_pml], cell_volumes[idx_non_pml]):.3e}')
                
                if(not returnResults or 14 in returnResults): ## if it is empty, or needed
                    x_temp[idx_non_pml] = cvxpySolve(A, b, 4, cell_volumes=cell_volumes[idx_non_pml])
                    cells.x.array[:] = x_temp + 0j
                    f.write_function(cells, 14)
                    err = reconstructionError(x_temp[idx_non_pml], epsr_ref[idx_non_pml], epsr_dut[idx_non_pml], cell_volumes[idx_non_pml])
                    errs.append(err)
                    print(f'Timestep 14 reconstruction error: {reconstructionError(x_temp[idx_non_pml], epsr_ref[idx_non_pml], epsr_dut[idx_non_pml], cell_volumes[idx_non_pml]):.3e}')
                
                f.close()
            
            print(f'done cvxpy solution, in {timer()-t_cvx:.2f} s')
            print()
        else:
            print('skipping cvxpy...')
        sys.stdout.flush()
        
    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2 ## should give max. RSS for the process in GB - possibly this is slightly less than the memory required
    mems = comm.gather(mem_usage, root=0)
    if( comm.rank == 0 ):
        totalMem = sum(mems) ## keep the total usage. Only the master rank should be used, so this should be fine
        print(f'Current max. memory usage: {totalMem:.2e} GB, {mem_usage:.2e} for the master process')
        return errs
    return None