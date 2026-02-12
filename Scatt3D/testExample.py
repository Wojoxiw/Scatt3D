## Simple test that should require all packages used in Scatt3D to be installed - if this runs, so should the main script.
# Adapted from an example in the dolfinx tutorial: https://github.com/jorgensd/dolfinx-tutorial/blob/main/chapter1/complex_mode.py
import os
os.environ["OMP_NUM_THREADS"] = "1" # perhaps needed for MPI speedup if using many processes locally? These do not seem to matter on the cluster
os.environ['MKL_NUM_THREADS'] = '1' # maybe also relevent
os.environ['NUMEXPR_NUM_THREADS'] = '1' # maybe also relevent
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
import ctypes.util
import PyScalapack ## https://github.com/USTC-TNS/TNSP/tree/main/PyScalapack
import resource

import meshMaker
import scatteringProblem
import memTimeEstimation
import postProcessing


#===============================================================================
# pc = PETSc.PC().create() ## to test if hpddm is installed
# pc.setType("hpddm")
# print("HPDDM loaded OK")
#===============================================================================

#===============================================================================
# print(ctypes.util.find_library("scalapack"))
# print(ctypes.util.find_library("blacs"))
# print(ctypes.util.find_library("blas"))
#===============================================================================
comm = MPI.COMM_WORLD
MPInum = comm.size
print(f"{MPI.COMM_WORLD.rank=} {MPI.COMM_WORLD.size=}, {MPI.COMM_SELF.rank=} {MPI.COMM_SELF.size=}, {MPI.Get_processor_name()=}")
sys.stdout.flush()
  
#===============================================================================
# # scalapacktest
# m, n, nrhs = 200, 20000, 1 ## rows, columns, rhs columns
# nprow, npcol = 1, MPInum ## MPI grid - number of process row*col must be <= MPInum
# 
# if(comm.rank == 0):
#     A_np = np.array(np.random.randn(m,n) + 1j*np.random.randn(m,n), order = 'F')
#     x_np = np.array(np.random.randn(n, 1) + 1j*np.random.randn(n, 1), order = 'F')
#     b_np = np.dot(A_np, x_np)
#     
#     b_np = np.array(b_np, order = 'F')
#     
#     postProcessing.scalapackLeastSquares(comm, MPInum, A_np, b_np, checkVsNp=True)
# else:
#     postProcessing.scalapackLeastSquares(comm, MPInum)
# # scalapacktest
#===============================================================================


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
problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options=petsc_options, petsc_options_prefix='testProblem')
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
    
    print(f'Memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2} GB')
