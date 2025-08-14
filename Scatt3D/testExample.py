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

import ctypes.util
from ctypes import c_double, c_int

#print(ctypes.util.find_library("scalapack"))
#print(ctypes.util.find_library("blacs"))
#print(ctypes.util.find_library("blas"))
MPInum = MPI.COMM_WORLD.size

import PyScalapack ## https://github.com/USTC-TNS/TNSP/tree/main/PyScalapack

# Setup
m, n, nrhs = 100, 1000, 1 ## rows, columns, rhs columns
nprow, npcol = 1, MPInum ## MPI grid - number of process row*col must be <= MPInum

scalapack = PyScalapack(ctypes.util.find_library("scalapack"), ctypes.util.find_library("blas"), ctypes.util.find_library("scalapack"))
with scalapack(b'C', nprow, npcol) as context: ## b'C' for column-major, b'R' for row-major
    A = context.array(m, n, mb=128, nb=128, dtype=np.complex128)
    x = context.array(n, nrhs, mb=128, nb=128, dtype=np.complex128)
    b = context.array(max(m, n), nrhs, mb=128, nb=1, dtype=np.complex128) ## must be larger to... contain space for computations?

    # Fill A/b: each rank can write to A.data locally
    Anp = np.random.randn(*A.data.shape) + 1j*np.random.randn(*A.data.shape)
    xnp = np.random.randn(*x.data.shape) + 1j*np.random.randn(*x.data.shape)
    bnp = np.dot(Anp, xnp)
    
    xnplstsq = np.linalg.lstsq(Anp, bnp)[0]
    
    A.data[:] = Anp
    b.data[:m] = bnp
    
    print(np.linalg.norm(np.dot(Anp, xnplstsq) - bnp), 'numpy norm')

    if context.rank.value == 0:
        print(f"Matrix dimension is ({A.m}, {A.n})")
    print(f"Matrix local dimension at process " +  #
          f"({context.myrow.value}, {context.mycol.value})" +  #
          f" is ({A.local_m}, {A.local_n})")
    if context.rank.value == 0:
        print(f"Vector dimension is ({b.m}, {b.n})")
    print(f"Vector local dimension at process " +  #
          f"({context.myrow.value}, {context.mycol.value})" +  #
          f" is ({b.local_m}, {b.local_n})")
    
    # Workspace query
    work_query = np.empty(1, dtype=np.float64)
    lwork = c_int(-1) ## -1 to query for optimal size
    info = c_int(0)
    scalapack.pzgels( # see documentation for pzgels: https://www.netlib.org/scalapack/explore-html/d2/dcf/pzgels_8f_aad4aa6a5bf9443ac8be9dc92f32d842d.html#aad4aa6a5bf9443ac8be9dc92f32d842d
        b'N', ## for solve A x = b, T for trans A^T x = b
        A.m, A.n, nrhs, # rows, columns, rhs columns
        *A.scalapack_params(),
        *b.scalapack_params(),
        work_query.ctypes.data_as(ctypes.POINTER(c_double)), lwork, info) ## workspace??, size of work?, output code??  
    
    if context.rank.value == 0:
        print("work query info =", info, work_query, lwork)
    
    lwork = int(work_query[0]) ## size of workspace?
    info = c_int(0)
    work = np.empty(lwork, dtype=np.float64)
    # Solve
    scalapack.pzgels(
        b'N',
        A.m, A.n, nrhs,
        *A.scalapack_params(),
        *b.scalapack_params(),
        work.ctypes.data_as(ctypes.POINTER(c_double)), lwork, info) 
    
    if context.rank.value == 0:
        print("pdgels returned info =", info)
        x = b.data[:n, :].copy()  # minimum 2-norm solution
        print(np.shape(x[:, 0]))
        print(np.shape(Anp), np.shape(bnp[:m,0]))
        lsts = np.linalg.lstsq(Anp, bnp[:m, 0])[0]
        #print(x[:, 0], lsts)
        print('error:', np.linalg.norm(x[:, 0]-lsts))




print('PyScalapack test done')
exit()

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

print('solving problem')
sys.stdout.flush()
uh = problem.solve()
print('problem solved')
sys.stdout.flush()

x = ufl.SpatialCoordinate(mesh)
u_ex = 0.5 * x[0]**2 + 1j*x[1]**2
L2_error = dolfinx.fem.form(ufl.dot(uh-u_ex, uh-u_ex) * ufl.dx(metadata={"quadrature_degree": 5}))
local_error = dolfinx.fem.assemble_scalar(L2_error)
global_error = np.sqrt(mesh.comm.allreduce(local_error, op=MPI.SUM))
max_error = mesh.comm.allreduce(np.max(np.abs(u_c.x.array-uh.x.array)))
print('global, max erors (max seems to be around e-16, glob. e-3):', global_error, max_error)

print('done')

