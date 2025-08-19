## Simple test that should require all packages used in Scatt3D to be installed - if this runs, so should the main script.
# Adapted from an example in the dolfinx tutorial: https://github.com/jorgensd/dolfinx-tutorial/blob/main/chapter1/complex_mode.py
import os
import numpy as np
import dolfinx, ufl, basix
import dolfinx.fem.petsc
from mpi4py import MPI
import gmsh
from matplotlib import pyplot as plt
import functools
import psutil
import sys, petsc4py
from petsc4py import PETSc
import scipy
from memory_profiler import memory_usage
from timeit import default_timer as timer
import time
import sys
import meshMaker
import scatteringProblem
import memTimeEstimation




#===============================================================================
# ### scalapack4py trying
# from scalapack4py import ScaLAPACK4py
# import ctypes.util
# from ctypes import CDLL, RTLD_GLOBAL, POINTER, c_int, c_double
# 
# 
# comm = MPI.COMM_WORLD
# MPInum = comm.size
# scalapath = ctypes.util.find_library("scalapack") ## auto-find the path
# sl = ScaLAPACK4py(CDLL(scalapath, mode=RTLD_GLOBAL))
# 
# m, n, nrhs = 5, 5, 1 ## rows, columns, rhs columns
# dtype=np.complex128 ### using complex doubles
# 
# A_np = np.array(np.random.randn(m, n) + 1j*np.random.randn(m, n), dtype=dtype, order = 'F') if comm.rank==0 else None
# A_np = np.array(np.identity(n, dtype=dtype), dtype=dtype, order = 'F') if comm.rank==0 else None
# x_np = np.array(np.random.randn(n) + 1j*np.random.randn(n), dtype=dtype, order = 'F') if comm.rank==0 else None
# x_np = np.ones(n, dtype=dtype, order = 'F') if comm.rank==0 else None
# b_np = np.zeros((max(m, n), nrhs), dtype=dtype, order = 'F') if comm.rank==0 else None
# if(comm.rank == 0):
#     b_np[:, 0] = np.dot(A_np, x_np)
#     x_nplstsq = np.linalg.lstsq(A_np, b_np)[0]
# 
# print('Comm', comm.rank, 'a:', A_np)
# print('Comm', comm.rank, 'b:', b_np)
# 
# MP, NP = 1, MPInum ## process grid rows, columns (product should be num processes)
# ctx = sl.make_blacs_context(sl.get_default_system_context(), MP, NP)
# 
# ## make the local input versions
# descr_A = sl.make_blacs_desc(ctx, m, n)
# descr_b = sl.make_blacs_desc(ctx, max(m, n), nrhs) ## max(m, n) since it maybe needs extra space to use for calculations...?
# 
# A_scala = np.zeros((descr_A.locrow, descr_A.loccol), dtype=dtype, order='F')
# sl.scatter_numpy(A_np, POINTER(c_int)(descr_A), A_scala.ctypes.data_as(POINTER(c_double)), A_scala.dtype)
# b_scala = np.zeros((descr_b.locrow, descr_b.loccol), dtype=dtype, order='F')
# sl.scatter_numpy(b_np, POINTER(c_int)(descr_b), b_scala.ctypes.data_as(POINTER(c_double)), b_scala.dtype)
# 
# work = np.zeros(1, dtype=dtype, order='F')
# lwork = -1
# info = -1 
# ## workspace query first
# sl.pzgels('N', m, n, nrhs,
#           A_scala, 1, 1, descr_A,
#           b_scala, 1, 1, descr_b,
#           work, lwork, info) 
# print('Work query result:', work, lwork, info) ## info doesn't seem to be written
# ## actual work
# lwork = int(work[0])
# work = np.zeros((lwork), dtype=dtype, order='F')
# sl.pzgels('N', m, n, nrhs,
#           A_scala, 1, 1, descr_A,
#           b_scala, 1, 1, descr_b,
#           work, lwork, info)
# # transform local results into global
# 
# A_gather = sl.gather_numpy(POINTER(c_int)(descr_A), A_scala.ctypes.data_as(POINTER(c_double)), (m, n))
# b_gather = sl.gather_numpy(POINTER(c_int)(descr_b), b_scala.ctypes.data_as(POINTER(c_double)), (nrhs, n))
# 
# print(f'A: {A_gather}')
# print(f'b: {b_gather}')
# 
# print('numpy norm |Ax-b|:', np.linalg.norm(np.dot(A_np, x_nplstsq) - b_np)) if comm.rank==0 else None
# print('pzgels norm |Ax-b|:', np.linalg.norm(np.dot(A_np, b_gather) - b_np)) if comm.rank==0 else None
# 
# 
# exit()
#===============================================================================

###
###
from mpi4py import MPI
import numpy as np
import ctypes.util
from ctypes import c_double, c_int
  
#===============================================================================
# print(ctypes.util.find_library("scalapack"))
# print(ctypes.util.find_library("blacs"))
# print(ctypes.util.find_library("blas"))
#===============================================================================
comm = MPI.COMM_WORLD
MPInum = comm.size
  
import PyScalapack ## https://github.com/USTC-TNS/TNSP/tree/main/PyScalapack
  
# Setup
m, n, nrhs = 55, 480, 1 ## rows, columns, rhs columns
nprow, npcol = 1, MPInum ## MPI grid - number of process row*col must be <= MPInum
  
scalapack = PyScalapack(ctypes.util.find_library("scalapack"), ctypes.util.find_library("blas"))


m, n, nrhs = 128, 128, 1
mb, nb = 1, 1 ## block size for the arrays. I guess 1 is fine?
with (
        scalapack(b'C', nprow, npcol) as context, ## computational context
        scalapack(b'C', 1, 1) as context0, ## feeder context
):
    ## make the arrays in the feeder context
    A0 = context0.array(m, n, mb, nb, dtype=np.complex128) 
    b0 = context0.array(max(m, n), nrhs, mb, nb, dtype=np.complex128) ## must be larger to... contain space for computations?
    x0 = context0.array(n, nrhs, mb, nb, dtype=np.complex128)
    
    
    if context0: ## give numpy values to the feeder context's arrays
        ## create actual values in numpy - also to test results
        A_np = np.array(np.random.randn(*A0.data.shape) + 1j*np.random.randn(*A0.data.shape), order = 'F')
        x_np = np.array(np.random.randn(*x0.data.shape) + 1j*np.random.randn(*x0.data.shape), order = 'F')
        b_np = np.dot(A_np, x_np)
        x_nplstsq = np.linalg.lstsq(A_np, b_np)[0]
    
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

print(f"{MPI.COMM_WORLD.rank=} {MPI.COMM_WORLD.size=}, {MPI.COMM_SELF.rank=} {MPI.COMM_SELF.size=}, {MPI.Get_processor_name()=}")
sys.stdout.flush()
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
u_r = dolfinx.fem.Function(V, dtype=np.float64) 
u_r.interpolate(lambda x: x[0])
u_c = dolfinx.fem.Function(V, dtype=np.complex128)
u_c.interpolate(lambda x:0.5*x[0]**2 + 1j*x[1]**2)
#print(u_r.x.array.dtype)
#print(u_c.x.array.dtype)
 
from petsc4py import PETSc
from dolfinx.fem.petsc import assemble_vector
#print(PETSc.ScalarType)
assert np.dtype(PETSc.ScalarType).kind == 'c' ## require complex mode
 
 
import ufl
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = dolfinx.fem.Constant(mesh, PETSc.ScalarType(-1 - 2j))
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx
 
 
L2 = f * ufl.conj(v) * ufl.dx
#print(L)
#print(L2)
 
J = u_c**2 * ufl.dx
F = ufl.derivative(J, u_c, ufl.conj(v))
residual = assemble_vector(dolfinx.fem.form(F))
#print(residual.array)
 
 
mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim-1, boundary_facets)
bc = dolfinx.fem.dirichletbc(u_c, boundary_dofs)
petsc_options = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options=petsc_options)
if(comm.rank == 0):
    print('solving problem')
sys.stdout.flush()
uh = problem.solve()
if(comm.rank == 0):
    print('problem solved')
sys.stdout.flush()
 
x = ufl.SpatialCoordinate(mesh)
u_ex = 0.5 * x[0]**2 + 1j*x[1]**2
L2_error = dolfinx.fem.form(ufl.dot(uh-u_ex, uh-u_ex) * ufl.dx(metadata={"quadrature_degree": 5}))
local_error = dolfinx.fem.assemble_scalar(L2_error)
global_error = np.sqrt(mesh.comm.allreduce(local_error, op=MPI.SUM))
max_error = mesh.comm.allreduce(np.max(np.abs(u_c.x.array-uh.x.array)))
if(comm.rank == 0):
    print('global, max erors (max seems to be around e-16, glob. e-3):', np.abs(global_error), max_error)
     
    print('done')
