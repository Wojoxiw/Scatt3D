# encoding: utf-8
## this file will have much of the postprocessing

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
        A_inv = np.linalg.pinv(A)
        x[idx] = np.dot(A_inv, b)
        #x[idx] = np.linalg.lstsq(A, b)
    
                
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, problemName+'testoutput.xdmf', 'w') as f:
        f.write_mesh(mesh)
        cells.x.array[:] = epsr_array_dut + 0j
        f.write_function(cells, 0)
        cells.x.array[:] = epsr_array_ref + 0j
        f.write_function(cells, -1)
        
        cells.x.array[:] = x + 0j
        f.write_function(cells, 1)            
    
    print('do SVD now')