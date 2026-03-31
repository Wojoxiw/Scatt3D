# encoding: utf-8
### This file is for the measurements, continuing from simulations in runScatt3D
#
# Adapted from 2D code started by Daniel Sjoberg, (https://github.com/dsjoberg-git/rotsymsca, https://github.com/dsjoberg-git/ekas3d) approx. 2024-12-13 
# Alexandros Pallaris, after that... and also before that, to some extent
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
import h5py

import psutil
import threading
from memory_profiler import memory_usage
from timeit import default_timer as timer
import time
import sys
import meshMaker
import scatteringProblem
import memTimeEstimation
import postProcessing

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
        matplotlib.use("QtAgg") ## so that plots actually appear
        plt.rc('axes', titlesize=16)     # fontsize of the axes title
        plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
        plt.rc('legend', fontsize=12)    # legend fontsize
        plt.rc('figure', titlesize=30)  # fontsize of the figure title'
        plt.rc('text', usetex=True) ## use latex to generate the font
        plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{bm}') ## load in some packages so I can bold stuff
    else:
        filename = 'prevRuns.npz'
        
    runName = 'testRun' # testing - the default name
    folder = 'data3D/'
    if(verbosity>2):
        print(f"{comm.rank=} {comm.size=}, {MPI.COMM_SELF.rank=} {MPI.COMM_SELF.size=}, {MPI.Get_processor_name()=}")
    if(comm.rank == model_rank):
        print(f'runScatt3D starting with {MPInum} MPI process(es) (main process on {MPI.Get_processor_name()=}):')
    sys.stdout.flush()
            
    def measurementScript(h = 1/3.5, degree = 3, runName=runName, mesh_settings={}, prob_settings={}):
        ## For measurements with patch antennas of a rectangular block. Four antennas, which are rotated by 30 degree steps to cover 360 degrees.
        prevRuns = memTimeEstimation.runTimesMems(folder, comm, filename = filename)
        mesh_settings = {'h': h, 'N_antennas': 4, 'order': degree, 'antenna_type': '6GHz measurement'} | mesh_settings ## uses settings given before those specified here ## settings for the meshMaker
        prob_settings = {'E_ref_anim': True, 'E_dut_anim': False, 'E_anim_allAnts': False, 'ErefEdut': False, 'verbosity': verbosity, 'dataFolder': folder, 'computeBoth': False, 'makeOptVects': True} | prob_settings
        
        if(mesh_settings['antenna_type'].startswith('patch')): ## set the dielectrics for the antennas
            epsrs=[]
            for n in range(mesh_settings['N_antennas']): ## each patch has 3 dielectric zones
                epsrs.append(4.4*(1 - .11/4.4j)) ## susbtrate - patch
                epsrs.append(4.4*(1 - .11/4.4j)) ## substrate under patch
                epsrs.append(2.1*(1 - 0.01j))
            prob_settings = prob_settings | {'antenna_mat_epsrs': epsrs}
        
        refMesh = meshMaker.MeshInfo(comm, folder+runName+'mesh.msh', reference = True, verbosity = verbosity, **mesh_settings)
        #prevRuns.memTimeEstimation(refMesh.ncells, doPrint=True, MPInum = comm.size)
        #refMesh.plotMeshPartition()
        prob = scatteringProblem.Scatt3DProblem(comm, refMesh, MPInum = MPInum, name = runName, fem_degree=degree, **prob_settings)
        #prob.makeOptVectors(skipQs=True)
        
        ## make the opt vects on the rec mesh... try h=1/10
        rec_mesh_settings = {'justInterpolationSubmesh': True, 'interpolationSubmeshSize': 1/10} | mesh_settings ## uses settings given before those specified here ## settings for the meshMaker
        recMesh = meshMaker.MeshInfo(comm, folder+runName+'mesh.msh', reference = True, verbosity = verbosity, **rec_mesh_settings)
        prob.switchToRecMesh(recMesh)
        prob.makeOptVectors(reconstructionMesh=True)
        prevRuns.memTimeAppend(prob)
        
        return prob
    
    
    
    
    runName = 'measurements_test'
    measurementScript(h=1/1, degree=1, runName=runName,
                    mesh_settings={'viewGMSH': False, 'N_antennas': 4, 'f0': 6e9, 'antenna_type': '6GHz measurement', 'antenna_radius': 0.18, 'object_geom': '6GHz measurement', 'domain_height': 1, 'domain_radius': 4.2},
                    prob_settings={'freqs': np.linspace(5.4e9, 6.6e9, 3)})
    #postProcessing.solveFromQs(folder+runName, solutionName='', onlyAPriori=True)
    
    
    
    
    
    if(comm.rank == model_rank):
        print(f'runScatt3D complete in {timer()-t1:.2f} s ({(timer()-t1)/3600:.2f} hours), exiting...')
        sys.stdout.flush()