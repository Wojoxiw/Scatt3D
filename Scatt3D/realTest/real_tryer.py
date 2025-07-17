# encoding: utf-8
### try solvers that require real numbers only

import os
import numpy as np
import dolfinx, ufl, basix
import dolfinx.fem.petsc

from mpi4py import MPI
import gmsh
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import functools
from scipy.constants import c as c0, mu_0 as mu0, epsilon_0 as eps0, pi

import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import scipy

import psutil
from memory_profiler import memory_usage
from timeit import default_timer as timer
import time
import sys
import meshMaker
import scatteringProblem
import memTimeEstimation
import postProcessing


def doIt():
    """
    Convert complex linear system A*x = b to real-augmented system
    
    [A][x] = [b] ==> |A_r  -A_i|  |x_r|  = |b_r|
                     |-A_i -A_r| |x_i|   = |-b_i|
    """
    
    ## load in the problem
    coords = np.load("realTest/coords.npz")['coords']
    Acload = np.load("realTest/A.npz")
    Ar = PETSc.Mat().createAIJ(size=tuple(Acload['shape']), csr=(Acload['indptr'], Acload['indices'], np.real(Acload['data'])))
    Ai = PETSc.Mat().createAIJ(size=tuple(Acload['shape']), csr=(Acload['indptr'], Acload['indices'], np.imag(Acload['data'])))
    A = PETSc.Mat().createNest([[Ar, -1*Ai], [-1*Ai, -1*Ar]])
    A.assemble()
    
    bc = np.load("realTest/b.npz")['data']
    bri = np.concatenate((np.real(bc), -1*np.imag(bc)))
    b = PETSc.Vec().createWithArray(bri, comm=comm)
    
    Gcload = np.load("realTest/G.npz")
    Gr = PETSc.Mat().createAIJ(size=tuple(Gcload['shape']), csr=(Gcload['indptr'], Gcload['indices'], np.real(Gcload['data'])))
    Gi = PETSc.Mat().createAIJ(size=tuple(Gcload['shape']), csr=(Gcload['indptr'], Gcload['indices'], np.imag(Gcload['data'])))
    G = PETSc.Mat().createNest([[Gr, -1*Gi], [Gi, Gr]])
    G.assemble()
    
    ## then solve it
    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setType("minres")
    
    pc = ksp.getPC()
    pc.setType("hypre")
    pc.setHYPREType("ams")
    
    pc.setCoordinates(coords)
    print('dg')
    pc.setHYPREDiscreteGradient(G) ## seems to... run out of memory here? for some reason.
    print('starting solve')
    x = b.duplicate()
    ksp.solve(b, x)
    
    print(f'Converged for reason: {ksp.getConvergedReason()}, after {ksp.getIterationNumber()} iterations. Norm: {ksp.getResidualNorm()}')

##MAIN STUFF
if __name__ == '__main__':
    # MPI settings
    comm = MPI.COMM_WORLD
    model_rank = 0 ## rank for printing and definitions, etc.
    verbosity = 2 ## 3 will print everything. 2, most things. 1, just the main process stuff.
    MPInum = comm.size
    
    t1 = timer()
    
    if(len(sys.argv) == 1): ## assume computing on local computer, not cluster. In jobscript for cluster, give a dummy argument
        filename = 'localCompTimesMems.npz'
    else:
        filename = 'prevRuns.npz'
        
    runName = 'testRun' # testing
    folder = 'data3D/'
    
    ts = timer()
    mem_usage = memory_usage((doIt), max_usage = True)/1000 ## track the memory usage here

    print(f'Max. memory: {mem_usage:.3f} GiB -- in {timer()-ts:.1f} s')
    
    
    if(comm.rank == model_rank):
        print(f'real_tryer complete in {timer()-t1:.2f} s, exiting...')
        sys.stdout.flush()