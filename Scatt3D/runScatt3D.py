# encoding: utf-8
### Modification of scatt2d to handle 3d geometry. This is the MAIN FILE
#
# Adapted from 2D code started by Daniel Sjoberg, (https://github.com/dsjoberg-git/rotsymsca, https://github.com/dsjoberg-git/ekas3d) approx. 2024-12-13 
# Alexandros Pallaris, after that... and also before that, to some extent
import os
import numpy as np
import dolfinx, ufl, basix
import dolfinx.fem.petsc
#os.environ["OMP_NUM_THREADS"] = "1" # perhaps needed for MPI speedup if using many processes locally? These do not seem to matter on the cluster, or locally with a spack installation
#os.environ['MKL_NUM_THREADS'] = '1' # maybe also relevent
#os.environ['NUMEXPR_NUM_THREADS'] = '1' # maybe also relevent
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
import threading
from memory_profiler import memory_usage
from timeit import default_timer as timer
import time
import sys
import meshMaker
import scatteringProblem
import memTimeEstimation
import postProcessing

#===============================================================================
# ##line profiling
# import line_profiler
# import atexit
# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats)
#===============================================================================

#===============================================================================
# ##memory profiling
# from memory_profiler import profile
#===============================================================================


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
        
    runName = 'testRun' # testing
    folder = 'data3D/'
    if(verbosity>2):
        print(f"{comm.rank=} {comm.size=}, {MPI.COMM_SELF.rank=} {MPI.COMM_SELF.size=}, {MPI.Get_processor_name()=}")
    if(comm.rank == model_rank):
        print(f'runScatt3D starting with {MPInum} MPI process(es) (main process on {MPI.Get_processor_name()=}):')
    sys.stdout.flush()
            
    def testRun(h = 1/2, degree=1): ## A quick test run to check it works. Default settings make this run in seconds
        prevRuns = memTimeEstimation.runTimesMems(folder, comm, filename = filename)
        refMesh = meshMaker.MeshInfo(comm, folder+runName+'mesh.msh', reference = True, viewGMSH = False, verbosity = verbosity, h=h, object_geom='sphere', domain_radius=0.8, domain_height=0.46, dome_height=0.22, PML_thickness=0.1, antenna_bounding_box_offset=0.05, object_radius=0.2, N_antennas=3, order=degree)
        #prevRuns.memTimeEstimation(refMesh.ncells, doPrint=True, MPInum = comm.size)
        #refMesh.plotMeshPartition()
        prob = scatteringProblem.Scatt3DProblem(comm, refMesh, verbosity = verbosity, MPInum = MPInum, name = runName, computeBoth=True, Nf=1, fem_degree=degree, E_ref_anim=True)
        prevRuns.memTimeAppend(prob)
        
        rmeshInfo = meshMaker.MeshInfo(comm, folder+runName+'mesh.msh', justInterpolationSubmesh=True, reference = True, viewGMSH = False, verbosity = verbosity, h=h, object_geom='sphere', domain_radius=0.8, domain_height=0.22, PML_thickness=0.1, antenna_bounding_box_offset=0.05, object_radius=0.2, N_antennas=3, order=degree)
        prob.makeOptVectors(reconstructionMeshInfo = rmeshInfo)
        
    def testFullExample(h = 1/15, degree = 1, dutOnRefMesh=True, antennaType='waveguide', ErefEdut=False, runName=runName, recMesh=True, mesh_settings={}, prob_settings={}): ## Testing toward a full example. Default settings are to reconstruct with ErefEref, and S-parameters for both cases from the DUT sim.
        prevRuns = memTimeEstimation.runTimesMems(folder, comm, filename = filename)
        mesh_settings = {'h': h, 'N_antennas': 9, 'order': degree, 'object_offset': np.array([.15, .1, 0]), 'viewGMSH': False, 'defect_offset': np.array([-.04, .17, .01]), 'defect_radius': 0.175, 'defect_height': 0.3, 'antenna_type': antennaType} | mesh_settings ## uses settings given before those specified here ## settings for the meshMaker
        prob_settings = {'E_ref_anim': True, 'E_dut_anim': False, 'E_anim_allAnts': False, 'dutOnRefMesh': dutOnRefMesh, 'ErefEdut': ErefEdut, 'verbosity': verbosity, 'Nf': 11, 'dataFolder': folder, 'computeBoth': True, 'makeOptVects': not recMesh} | prob_settings
        
        if(mesh_settings['antenna_type'] == 'patch'): ## set the dielectrics for the antennas
            epsrs=[]
            for n in range(mesh_settings['N_antennas']): ## each patch has 3 dielectric zones
                epsrs.append(4.4*(1 - .11/4.4j)) ## susbtrate - patch
                epsrs.append(4.4*(1 - .11/4.4j)) ## substrate under patch
                epsrs.append(2.1*(1 - 0.01j))
            prob_settings = prob_settings | {'antenna_mat_epsrs': epsrs}
        
        if(dutOnRefMesh):
            refMesh = meshMaker.MeshInfo(comm, folder+runName+'mesh.msh', reference = False, verbosity = verbosity, **mesh_settings)
        else:
            refMesh = meshMaker.MeshInfo(comm, folder+runName+'mesh.msh', reference = True, verbosity = verbosity, **mesh_settings)
        #prevRuns.memTimeEstimation(refMesh.ncells, doPrint=True, MPInum = comm.size)
        #refMesh.plotMeshPartition()
        if(not dutOnRefMesh):
            dutMesh = meshMaker.MeshInfo(comm, folder+runName+'mesh.msh', reference = False, viewGMSH = False, verbosity = verbosity, **mesh_settings)
            prob = scatteringProblem.Scatt3DProblem(comm, refMesh, dutMesh, dutOnRefMesh=False, MPInum = MPInum, name = runName, fem_degree=degree, **prob_settings)
        else:
            prob = scatteringProblem.Scatt3DProblem(comm, refMesh, MPInum = MPInum, name = runName, fem_degree=degree, **prob_settings)
            
        if(recMesh): ## make the opt vects on the rec mesh... try h=1/10
            rec_mesh_settings = {'justInterpolationSubmesh': True, 'interpolationSubmeshSize': 1/10} | mesh_settings ## uses settings given before those specified here ## settings for the meshMaker
            recMesh = meshMaker.MeshInfo(comm, folder+runName+'mesh.msh', reference = True, verbosity = verbosity, **rec_mesh_settings)
            prob.switchToRecMesh(recMesh)
            prob.makeOptVectors(reconstructionMesh=True)
        prevRuns.memTimeAppend(prob)
    
    def testRunDifferentDUTAntennas(h = 1/15, degree = 1): ## Testing what happens when different antennas are used in the ref (simulation) as in the DUT case (unsuccessful reconstruction)
        prevRuns = memTimeEstimation.runTimesMems(folder, comm, filename = filename)
        settings = {'N_antennas': 9, 'order': degree, 'object_offset': np.array([.15, .1, 0]), 'defect_offset': np.array([-.04, .17, .01])} ## settings for the meshMaker
        refMesh = meshMaker.MeshInfo(comm, folder+runName+'mesh.msh', reference = True, viewGMSH = False, verbosity = verbosity, h=h, antenna_type='waveguide', **settings)
        dutMesh = meshMaker.MeshInfo(comm, folder+runName+'mesh.msh', reference = False, viewGMSH = False, verbosity = verbosity, h=h, antenna_type='patch', **settings)
        prob = scatteringProblem.Scatt3DProblem(comm, refMesh, dutMesh, computeBoth=True, verbosity = verbosity, MPInum = MPInum, name = runName, Nf = 11, fem_degree=degree, ErefEdut=True, dutOnRefMesh=False)
        prevRuns.memTimeAppend(prob)
        
    def testShiftedExample(h = 1/15, degree = 1, dutOnRefMesh=False): ## Where the separate dut mesh has the object shifted by some amount
        prevRuns = memTimeEstimation.runTimesMems(folder, comm, filename = filename)
        settings = {'N_antennas': 9, 'order': degree, 'object_offset': np.array([.15, .1, 0]), 'defect_offset': np.array([-.04, .17, .01])} ## settings for the meshMaker
        settingsDut = {'N_antennas': 9, 'order': degree, 'object_offset': np.array([.15, .1, .1]), 'defect_offset': np.array([-.04, .17, .01])} ## settings for the meshMaker
        if(dutOnRefMesh):
            refMesh = meshMaker.MeshInfo(comm, folder+runName+'mesh.msh', reference = False, viewGMSH = False, verbosity = verbosity, h=h, **settings)
        else:
            refMesh = meshMaker.MeshInfo(comm, folder+runName+'mesh.msh', reference = True, viewGMSH = False, verbosity = verbosity, h=h, **settings)
        dutMesh = meshMaker.MeshInfo(comm, folder+runName+'mesh.msh', reference = False, viewGMSH = False, verbosity = verbosity, h=h, **settingsDut)
        #prevRuns.memTimeEstimation(refMesh.ncells, doPrint=True, MPInum = comm.size)
        #refMesh.plotMeshPartition()
        prob = scatteringProblem.Scatt3DProblem(comm, refMesh, dutMesh, computeBoth=True, verbosity = verbosity, MPInum = MPInum, name = runName, Nf = 11, fem_degree=degree, ErefEdut=True, dutOnRefMesh=dutOnRefMesh)
        prevRuns.memTimeAppend(prob)
        
    def testSlightlyShiftedExampleNoDefects(h = 1/15, degree = 1, dutOnRefMesh=False): ## Where the separate dut mesh has the object shifted by some amount
        prevRuns = memTimeEstimation.runTimesMems(folder, comm, filename = filename)
        settings = {'N_antennas': 9, 'order': degree, 'object_offset': np.array([.15, .1, 0]), 'defect_offset': np.array([-.04, .17, .01]), 'defect_geom': ''} ## settings for the meshMaker
        settingsDut = {'N_antennas': 9, 'order': degree, 'object_offset': np.array([.15, .1, .03]), 'defect_offset': np.array([-.04, .17, .01]), 'defect_geom': ''} ## settings for the meshMaker
        if(dutOnRefMesh):
            refMesh = meshMaker.MeshInfo(comm, folder+runName+'mesh.msh', reference = False, viewGMSH = False, verbosity = verbosity, h=h, **settings)
        else:
            refMesh = meshMaker.MeshInfo(comm, folder+runName+'mesh.msh', reference = True, viewGMSH = False, verbosity = verbosity, h=h, **settings)
        dutMesh = meshMaker.MeshInfo(comm, folder+runName+'mesh.msh', reference = False, viewGMSH = False, verbosity = verbosity, h=h, **settingsDut)
        #prevRuns.memTimeEstimation(refMesh.ncells, doPrint=True, MPInum = comm.size)
        #refMesh.plotMeshPartition()
        prob = scatteringProblem.Scatt3DProblem(comm, refMesh, dutMesh, computeBoth=True, verbosity = verbosity, MPInum = MPInum, name = runName, Nf = 11, fem_degree=degree, ErefEdut=True, dutOnRefMesh=dutOnRefMesh)
        prevRuns.memTimeAppend(prob)
        
    def testLargeExample(h = 1/15, degree = 1, dutOnRefMesh=True): ## Testing a large-object example
        prevRuns = memTimeEstimation.runTimesMems(folder, comm, filename = filename)
        settings = {'N_antennas': 9, 'material_epsrs': [3.0*(1 - 0.01j)], 'order': degree, 'object_geom': 'complex1', 'defect_geom': 'complex1', 'h': h, 'defect_height': .63, 'defect_radius': .31, 'object_radius': 1.09, 'antenna_type': 'patch'} ## settings for the meshMaker 'object_offset': np.array([.15, .1, 0])
        if(dutOnRefMesh):
            refMesh = meshMaker.MeshInfo(comm, folder+runName+'mesh.msh', reference = False, viewGMSH = False, verbosity = verbosity, **settings)
        else:
            refMesh = meshMaker.MeshInfo(comm, folder+runName+'mesh.msh', reference = True, viewGMSH = False, verbosity = verbosity, **settings)
        #dutMesh = meshMaker.MeshInfo(comm, folder+runName+'mesh.msh', reference = False, viewGMSH = False, verbosity = verbosity, h=h, N_antennas=9, order=degree)
        #prevRuns.memTimeEstimation(refMesh.ncells, doPrint=True, MPInum = comm.size)
        #refMesh.plotMeshPartition()
        prob = scatteringProblem.Scatt3DProblem(comm, refMesh, computeBoth=True, verbosity = verbosity, MPInum = MPInum, name = runName, Nf = 11, material_epsrs=[3.0*(1 - 0.01j)], fem_degree=degree, ErefEdut=True, dutOnRefMesh=dutOnRefMesh, defect_epsrs=[4.0*(1 - 0.01j), 4.0*(1 - 0.01j), 2.0*(1 - 0.01j)])
        prob.makeOptVectors(skipQs=True)
        prevRuns.memTimeAppend(prob)
        degree = 1
        
    def testSphereScattering(h = 1/12, degree=1, showPlots=False): ## run a spherical domain and object, test the far-field scattering for an incident plane-wave from a sphere vs Mie theoretical result.
        prevRuns = memTimeEstimation.runTimesMems(folder, comm, filename = filename)
        refMesh = meshMaker.MeshInfo(comm, reference = True, viewGMSH = False, verbosity = verbosity, N_antennas=0, object_radius = .33, domain_radius=.9, PML_thickness=0.5, h=h, domain_geom='sphere', object_geom='sphere', FF_surface = True, order=degree)
        #refMesh.plotMeshPartition()
        #prevRuns.memTimeEstimation(refMesh.ncells, doPrint=True, MPInum = comm.size)
        freqs = np.linspace(10e9, 12e9, 1)
        prob = scatteringProblem.Scatt3DProblem(comm, refMesh, verbosity=verbosity, name=runName, MPInum=MPInum, makeOptVects=True, excitation='planewave', freqs = freqs, material_epsrs=[2.0*(1-0.01j)], fem_degree=degree)
        if(showPlots):
            prob.calcNearField()
        prob.calcFarField(reference=True, compareToMie = True, showPlots=showPlots, returnConvergenceVals=False)
        prevRuns.memTimeAppend(prob)
        
    def testPatchPattern(h = 1/12, degree=1, freqs = np.array([10e9]), name='patchPatternTest', showPlots=True): ## run a spherical domain and object, test the far-field pattern from a single patch antenna near the center
        runName = name
        prevRuns = memTimeEstimation.runTimesMems(folder, comm, filename = filename)
        refMesh = meshMaker.MeshInfo(comm, reference = True, viewGMSH = False, verbosity = verbosity, N_antennas=1, domain_radius=1.8, PML_thickness=0.5, h=h, domain_geom='sphere', antenna_type='patchtest', object_geom='', FF_surface = True, order=degree)
        epsrs=[]
        epsrs.append(4.4*(1 - .11/4.4j)) ## susbtrate - patch
        epsrs.append(4.4*(1 - .11/4.4j)) ## substrate under patch
        epsrs.append(2.1*(1 - 0.01j))
        #refMesh.plotMeshPartition()
        #prevRuns.memTimeEstimation(refMesh.ncells, doPrint=True, MPInum = comm.size)
        if(len(freqs) == 1): ## plot the given frequency, if there is only 1
            prob = scatteringProblem.Scatt3DProblem(comm, refMesh, verbosity=verbosity, name=runName, MPInum=MPInum, makeOptVects=False, freqs = freqs, fem_degree=degree, antenna_mat_epsrs=epsrs)
            prob.calcFarField(reference=True, plotFF=True, showPlots=showPlots)
        else: ## save Ss
            prob = scatteringProblem.Scatt3DProblem(comm, refMesh, verbosity=verbosity, name=runName, MPInum=MPInum, makeOptVects=True, freqs = freqs, fem_degree=degree, antenna_mat_epsrs=epsrs)
        prevRuns.memTimeAppend(prob)
 
    def convergenceTestPlots(convergence = 'meshsize', deg=1): ## Runs with reducing mesh size, for convergence plots. Uses the far-field surface test case. If showPlots, show them - otherwise just save them
        if(convergence == 'meshsize'):
            if(deg==1):
                ks = np.linspace(5, 17, 7)
            elif(deg==2):
                ks = np.linspace(3, 7, 7)
            else:
                ks = np.linspace(2, 6, 7)
        elif(convergence == 'pmlR0'): ## result of this is that the value must be below 1e-2, from there further reduction matches the forward-scattering better, the back-scattering less
            ks = np.linspace(2, 15, 10)
            ks = 10**(-ks)
        elif(convergence == 'dxquaddeg'): ## Result of this showed a large increase in time near the end, and an accuracy improvement for increasing from 2 to 3. Not sending any value causes a huge memory cost/error (process gets killed).
            ks = np.arange(1, 20)
            
        ndofs = np.zeros_like(ks) ## to hold problem size
        calcT = np.zeros_like(ks) ## to hold problem size
            
        areaVals = [] ## vals returned from the calculations
        FFrmsRelErrs = np.zeros(len(ks)) ## for the farfields
        FFrmsveryRelErrs = np.zeros(len(ks))
        FFmaxRelErrs = np.zeros(len(ks))
        FFmaxErrRel = np.zeros(len(ks))
        NFrmsErrs = np.zeros(len(ks))
        khatRmsErrs = np.zeros(len(ks))
        khatMaxErrs = np.zeros(len(ks))
        meshOptions = dict()
        probOptions = dict()
        prevRuns = memTimeEstimation.runTimesMems(folder, comm, filename = filename)
        for i in range(len(ks)):
            if(convergence == 'meshsize'):
                meshOptions = dict(h = 1/ks[i])
            elif(convergence == 'pmlR0'):
                probOptions = dict(PML_R0 = ks[i])
            elif(convergence == 'dxquaddeg'):
                probOptions = dict(quaddeg = ks[i])
            
            refMesh = meshMaker.MeshInfo(comm, reference = True, viewGMSH = False, verbosity = verbosity, N_antennas=0, object_radius = .33, PML_thickness=0.5, domain_radius=0.9, domain_geom='sphere', object_geom='sphere', FF_surface = True, order=deg, **meshOptions)
            prob = scatteringProblem.Scatt3DProblem(comm, refMesh, verbosity = verbosity, name=runName, MPInum = MPInum, makeOptVects=False, excitation = 'planewave', material_epsrs=[2.0*(1-0.01j)], Nf=1, fem_degree=deg, **probOptions)
            newval, khats, farfields, mies = prob.calcFarField(reference=True, compareToMie = False, showPlots=False, returnConvergenceVals=True) ## each return is FF surface area, khat integral at each angle, farfields+mies at each angle
            simNF, FEKONF = prob.calcNearField(FEKOcomp=True, showPlots=False)
            prevRuns.memTimeAppend(prob)
            if(comm.rank == model_rank): ## only needed for main process
                areaVals.append(newval)
                ndofs[i] = prob.FEMmesh_ref.ndofs
                calcT[i] = prob.calcTime
                khatRmsErrs[i] = np.sqrt(np.sum(khats**2)/np.size(khats))
                khatMaxErrs[i] = np.max(khats)
                intenss = np.abs(farfields[0,:,0])**2 + np.abs(farfields[0,:,1])**2
                FFrelativeErrors = np.abs( (intenss - mies) ) / np.abs( mies )
                FFrmsRelErrs[i] = np.sqrt(np.sum(FFrelativeErrors**2)/np.size(FFrelativeErrors))
                FFvrelativeErrors = np.abs( (intenss - mies) ) / np.max(np.abs(mies)) ## relative to the max. mie intensity, to make it even more relative
                FFrmsveryRelErrs[i] = np.sqrt(np.sum(FFvrelativeErrors**2)/np.size(FFvrelativeErrors)) ## absolute error, scaled by the max. mie value
                FFmaxRelErrs[i] = np.max(FFrelativeErrors)
                FFmaxErrRel[i] = np.abs( np.max(intenss - mies)) / np.max(np.abs(mies))
                NFrmsErrs[i] = np.sqrt(np.sum(np.abs(simNF-FEKONF)**2)/np.size(simNF)) / np.max(np.abs(FEKONF))
                if(verbosity>1):
                    print(f'Run {i+1}/{len(ks)} completed')
                ## Plot each iteration for case of OOM or such
                real_area = 4*pi*prob.FEMmesh_ref.meshInfo.FF_surface_radius**2
                
                
                if(convergence == 'meshsize'): ## plot 3 times - also vs time and ndofs
                    p=0
                else:
                    p=2
                    
                while p<3: ## just to do the multiple plots
                    fig1 = plt.figure()
                    fig1.set_size_inches(8, 6)
                    ax1 = plt.subplot(1, 1, 1)
                    ax1.grid(True)
                    ax1.set_title('Convergence of Different Values')
                    idx = np.argsort(ks[:i+1]) ## ordered increasing
                    if(convergence == 'meshsize'):
                        if(p==0):
                            ax1.set_xlabel(r'Inverse mesh size ($\lambda / h$)')
                            idx = np.argsort(-ks[:i+1])
                            xs = ks
                        elif(p==1):
                            ax1.set_xlabel(r'Calculation Time [s]')
                            xs = calcT
                        else:
                            ax1.set_xlabel(r'# of dofs')
                            xs = ndofs
                    elif(convergence == 'pmlR0'):
                        ax1.set_xlabel(r'R0')
                        ax1.set_xscale('log')
                    elif(convergence == 'dxquaddeg'):
                        ax1.set_xlabel(r'dx Quadrature Degree')
                    
                    ax1.plot(xs[idx], np.abs((real_area-np.array(areaVals))/real_area)[idx], marker='o', linestyle='--', label = r'$\int dS$ - rel. error')
                    #ax1.plot(xs[idx], khatMaxErrs[idx], marker='o', linestyle='--', label = r'$\int \hat{k}\cdot \vec{n} \, dS$ - max. abs. error')
                    ax1.plot(xs[idx], khatRmsErrs[idx], marker='o', linestyle='--', label = r'$\int \hat{k}\cdot \vec{n} \, dS$ - RMS error')
                    ax1.plot(xs[idx], FFrmsRelErrs[idx], marker='o', linestyle='--', label = r'Farfield cuts RMS rel. error')
                    ax1.plot(xs[idx], FFrmsveryRelErrs[idx], marker='o', linestyle='--', label = r'Farfield cuts RMS. error, normalized')
                    #ax1.plot(xs[idx], FFmaxErrRel[idx], marker='o', linestyle='--', label = r'Farfield max error, rel.')
                    ax1.plot(xs[idx], NFrmsErrs[idx], marker='o', linestyle='--', label = r'Nearfield-FEKO error norm, rel.')
                    
                    ax1.set_yscale('log')
                    ax1.legend()
                    fig1.tight_layout()
                    plt.savefig(prob.dataFolder+prob.name+convergence+f'meshconvergence{p}deg{deg}.png')
                    p+=1
                    if(MPInum == 1 and i==len(ks)-1): ## only show the last one
                        plt.show()
                    plt.close()
                    
    def patchConvergenceTestPlots(convergence = 'meshsize', degree=3): ## Runs with reducing mesh size, for convergence plots. Uses the patch antenna test case, comparing to the FEKO result. If showPlots, show them - otherwise just save them
        if(convergence == 'meshsize'):
            if(degree == 1):
                ks = np.linspace(3, 14, 9)
            elif(degree == 2):
                ks = np.linspace(2.5, 8.5, 9)
            else:
                ks = np.linspace(2, 6.2, 9)
            
        ndofs = np.zeros_like(ks) ## to hold problem size
        calcT = np.zeros_like(ks) ## to hold problem size
            
        rmsS11PhaseDiff = np.zeros(len(ks)) ## adjusted so the first value matches with FEKO's
        maxS11PhaseDiff = np.zeros(len(ks)) ## adjusted so the first value matches with FEKO's
        rmsS11Diff = np.zeros(len(ks))
        
        meshOptions = dict()
        probOptions = dict()
        prevRuns = memTimeEstimation.runTimesMems(folder, comm, filename = filename)
        for i in range(len(ks)):
            if(convergence == 'meshsize'):
                meshOptions = dict(h = 1/ks[i])
            
            refMesh = meshMaker.MeshInfo(comm, reference = True, viewGMSH = False, verbosity = verbosity, N_antennas=1, domain_radius=1.8, PML_thickness=0.5, domain_geom='sphere', antenna_type='patchtest', object_geom='', FF_surface = True, order=degree, **meshOptions)
            prob = scatteringProblem.Scatt3DProblem(comm, refMesh, verbosity=verbosity, name=runName, MPInum=MPInum, makeOptVects=True, Nf=20, fem_degree=degree, material_epsrs=[2.1], **probOptions) ## the first 3 materials per antenna are the antenna's dielectric volume
            fekof = 'TestStuff/FEKO patch S11.dat'
            fekoData = np.transpose(np.loadtxt(fekof, skiprows = 2))
            
            interpReal = np.interp(prob.fvec, fekoData[0], fekoData[1]) ## interpolate the FEKO data to whatever frequency points are calculated here
            interpImag = np.interp(prob.fvec, fekoData[0], fekoData[2]) ## interpolate the FEKO data to whatever frequency points are calculated here
            
            fekoS11 = interpReal+1j*interpImag
            fekoPhase = np.angle(fekoS11)
            
            S11 = prob.S_ref.flatten()
            adjustedPhase = np.angle(S11) + (fekoPhase[0]-np.angle(S11)[0])
            
            prevRuns.memTimeAppend(prob)
            if(comm.rank == model_rank): ## only needed for main process
                ndofs[i] = prob.FEMmesh_ref.ndofs
                calcT[i] = prob.calcTime
                rmsS11PhaseDiff[i] = np.sqrt(np.sum(np.abs(adjustedPhase-fekoPhase)**2)/np.size(S11))
                maxS11PhaseDiff[i] = np.max(np.abs(adjustedPhase-fekoPhase))
                rmsS11Diff[i] = np.sqrt(np.sum(np.abs(S11-fekoS11)**2)/np.size(S11))
                if(verbosity>1):
                    print(f'Run {i+1}/{len(ks)} completed')
                ## Plot each iteration for case of OOM or such
                
                if(convergence == 'meshsize'): ## plot 3 times - also vs time and ndofs
                    p=0
                else:
                    p=2
                    
                while p<3: ## just to do the multiple plots
                    fig1 = plt.figure()
                    fig1.set_size_inches(8, 6)
                    ax1 = plt.subplot(1, 1, 1)
                    ax1.grid(True)
                    ax1.set_title('Convergence of sim. to FEKO (Patch sim.)')
                    idx = np.argsort(ks[:i+1]) ## ordered increasing
                    if(convergence == 'meshsize'):
                        if(p==0):
                            ax1.set_xlabel(r'Inverse mesh size ($\lambda / h$)')
                            idx = np.argsort(-ks[:i+1])
                            xs = ks
                        elif(p==1):
                            ax1.set_xlabel(r'Calculation Time [s]')
                            xs = calcT
                        else:
                            ax1.set_xlabel(r'# of dofs')
                            xs = ndofs
                    elif(convergence == 'pmlR0'):
                        ax1.set_xlabel(r'R0')
                        ax1.set_xscale('log')
                    elif(convergence == 'dxquaddeg'):
                        ax1.set_xlabel(r'dx Quadrature Degree')
                    
                    ax1.plot(xs[idx], rmsS11Diff[idx], marker='o', linestyle='--', label = r'RMS S$_{11}$ Diff.')
                    ax1.plot(xs[idx], rmsS11PhaseDiff[idx], marker='o', linestyle='--', label = r'RMS S$_{11}$ Phase Diff.')
                    ax1.plot(xs[idx], maxS11PhaseDiff[idx], marker='o', linestyle='--', label = r'max. S$_{11}$ Phase Diff.')
                    
                    ax1.set_yscale('log')
                    ax1.legend()
                    fig1.tight_layout()
                    plt.savefig(prob.dataFolder+prob.name+convergence+f'patchconvergence{p}deg{degree}.png')
                    p+=1
                    if(MPInum == 1 and i==len(ks)-1): ## only show the last one
                        plt.show()
                    plt.close()
            
    def testSolverSettings(h = 1/12, deg=1): # Varies settings in the ksp solver/preconditioner, plots the time and iterations a computation takes. Uses the sphere-scattering test case
        refMesh = meshMaker.MeshInfo(comm, reference = True, viewGMSH = False, verbosity = verbosity, N_antennas=3, object_radius = .73, domain_radius=1.9, PML_thickness=0.5, h=h, domain_geom='sphere', object_geom='cylinder', order=deg, FF_surface = False)
        settings = [] ## list of solver settings
        maxTime = 355 ## max solver time in [s], to cut off overly-long runs. Is only checked between iterations, some of which can take minutes...
        
        ## MG tests
        testName = 'mg_testing'
        for mgrtol in [1e-1, 1.5e-1, .5e-1]:
            for maxit in [25, 50, 80]:
                for pctype in ['asm', 'sor', 'gasm']:
                        settings.append( {'mg_coarse_ksp_rtol': mgrtol, 'mg_coarse_ksp_max_it': maxit, 'mg_levels_pc_type': pctype, 'mg_coarse_pc_type': pctype} )
        
        
        #=======================================================================
        # ## GASM tests
        # for subDs in [MPInum*1, MPInum*2, MPInum*3]:
        #     for overlap in [2, 3, 4, 5]:
        #         for tryit in [{'sub_pc_type':'lu', 'sub_pc_factor_mat_solver_type': 'mumps'}, {'sub_pc_type':'ilu', 'sub_pc_factor_levels': 1, 'sub_pc_factor_mat_solver_type': 'petsc'}, {'sub_pc_type':'ilu', 'sub_pc_factor_levels': 2, 'sub_pc_factor_mat_solver_type': 'petsc'}, {'sub_pc_type':'ilu', 'sub_pc_factor_levels': 3, 'sub_pc_factor_mat_solver_type': 'petsc'}]:
        #             for try2 in [{'sub_pc_factor_mat_ordering_type': 'nd'}, {}]:
        #                 settings.append( {'pc_gasm_total_subdomains': subDs, 'pc_gasm_overlap': overlap, **tryit, **try2} )
        #=======================================================================
        
        #=======================================================================
        # ## composite PC tests
        # for type in ['additive', 'mutiplicative']:
        #     for pc1 in ['gamg', 'asm', 'sor', 'bcgs', 'gmres', 'gasm']:
        #         for pc2 in ['gasm', 'asm', 'sor', 'bcgs', 'gmres', 'gamg']:
        #             if(pc1 != pc2):
        #                 def pc1stuff(pc1, pc2):
        #                     if(pc1 == 'gasm'):
        #                         pc1t = {'pc_gasm_total_subdomains': MPInum, 'pc_gasm_overlap': 4, 'sub_pc_type': 'bjacobi', 'sub_pc_factor_levels': 1, 'sub_pc_factor_mat_solver_type': 'petsc', 'sub_pc_factor_mat_ordering_type': 'nd'}
        #                         settings.append( {'pc_composite_type': type, 'pc_composite_pcs': pc1+','+pc2, **pc1t, **pc2t} )
        #                         pc1t = {'pc_gasm_total_subdomains': MPInum*2, 'pc_gasm_overlap': 3, 'sub_pc_type': 'bjacobi', 'sub_pc_factor_levels': 1, 'sub_pc_factor_mat_solver_type': 'petsc', 'sub_pc_factor_mat_ordering_type': 'nd'}
        #                         settings.append( {'pc_composite_type': type, 'pc_composite_pcs': pc1+','+pc2, **pc1t, **pc2t} )
        #                         pc1t = {'pc_gasm_total_subdomains': MPInum, 'pc_gasm_overlap': 3, 'sub_pc_type': 'lu', 'sub_pc_factor_mat_solver_type': 'mumps'}
        #                         settings.append( {'pc_composite_type': type, 'pc_composite_pcs': pc1+','+pc2, **pc1t, **pc2t} )
        #                     elif(pc1 == 'asm'):
        #                         pc1t = {'sub_pc_type':'lu', 'sub_pc_factor_mat_solver_type': 'mumps'}
        #                         settings.append( {'pc_composite_type': type, 'pc_composite_pcs': pc1+','+pc2, **pc1t, **pc2t} )
        #                     elif(pc1 == 'sor'):
        #                         pc1t = {}
        #                         settings.append( {'pc_composite_type': type, 'pc_composite_pcs': pc1+','+pc2, **pc1t, **pc2t} )
        #                     elif(pc1 == 'gamg'):
        #                         pc1t = {'pc_gamg_type': 'agg', 'pc_gamg_sym_graph': 1, 'matptap_via': 'scalable', 'pc_gamg_square_graph': 1, 'pc_gamg_reuse_interpolation': 1}
        #                         settings.append( {'pc_composite_type': type, 'pc_composite_pcs': pc1+','+pc2, **pc1t, **pc2t} )
        #                         pc1t = {'mg_levels_pc_type': 'jacobi', 'pc_gamg_agg_nsmooths': 1, 'pc_mg_cycle_type': 'v', 'pc_gamg_aggressive_coarsening': 2, 'pc_gamg_theshold': 0.01, 'mg_levels_ksp_max_it': 5, 'mg_levels_ksp_type': 'chebyshev', 'pc_gamg_repartition': False, 'pc_gamg_square_graph': True, 'pc_mg_type': 'additive'}
        #                         settings.append( {'pc_composite_type': type, 'pc_composite_pcs': pc1+','+pc2, **pc1t, **pc2t} )
        #                     elif(pc1 == 'bcgs'):
        #                         pc1t = {'ksp_ksp_type': 'bcgs', 'ksp_ksp_max_it': 100, 'ksp_pc_type': 'jacobi'}
        #                         settings.append( {'pc_composite_type': type, 'pc_composite_pcs': 'ksp'+','+pc2, **pc1t, **pc2t} )
        #                     elif(pc1 == 'gmres'):
        #                         pc1t = {'pc_ksp_type': 'gmres', 'ksp_max_it': 1, 'pc_ksp_rtol' : 1e-1, "pc_ksp_pc_type": "sor"}
        #                         settings.append( {'pc_composite_type': type, 'pc_composite_pcs': 'ksp'+','+pc2, **pc1t, **pc2t} )
        #                         
        #                 if(pc2 == 'gasm'):
        #                     pc2t = {'pc_gasm_total_subdomains': MPInum, 'pc_gasm_overlap': 4, 'sub_pc_type': 'bjacobi', 'sub_pc_factor_levels': 1, 'sub_pc_factor_mat_solver_type': 'petsc', 'sub_pc_factor_mat_ordering_type': 'nd'}
        #                     pc1stuff(pc1, pc2)
        #                     pc2t = {'pc_gasm_total_subdomains': MPInum*2, 'pc_gasm_overlap': 3, 'sub_pc_type': 'bjacobi', 'sub_pc_factor_levels': 1, 'sub_pc_factor_mat_solver_type': 'petsc', 'sub_pc_factor_mat_ordering_type': 'nd'}
        #                     pc1stuff(pc1, pc2)
        #                     pc2t = {'pc_gasm_total_subdomains': MPInum, 'pc_gasm_overlap': 3, 'sub_pc_type': 'lu', 'sub_pc_factor_mat_solver_type': 'mumps'}
        #                     pc1stuff(pc1, pc2)
        #                 elif(pc2 == 'asm'):
        #                     pc2t = {'sub_pc_type':'lu', 'sub_pc_factor_mat_solver_type': 'mumps'}
        #                     pc1stuff(pc1, pc2)
        #                 elif(pc2 == 'sor'):
        #                     pc2t = {}
        #                     pc1stuff(pc1, pc2)
        #                 elif(pc2 == 'gamg'):
        #                     pc2t = {'pc_gamg_type': 'agg', 'pc_gamg_sym_graph': 1, 'matptap_via': 'scalable', 'pc_gamg_square_graph': 1, 'pc_gamg_reuse_interpolation': 1}
        #                     pc1stuff(pc1, pc2)
        #                     pc2t = {'mg_levels_pc_type': 'jacobi', 'pc_gamg_agg_nsmooths': 1, 'pc_mg_cycle_type': 'v', 'pc_gamg_aggressive_coarsening': 2, 'pc_gamg_theshold': 0.01, 'mg_levels_ksp_max_it': 5, 'mg_levels_ksp_type': 'chebyshev', 'pc_gamg_repartition': False, 'pc_gamg_square_graph': True, 'pc_mg_type': 'additive'}
        #                     pc1stuff(pc1, pc2)
        #                 elif(pc2 == 'bcgs'):
        #                     pc2t = {'ksp_ksp_type': 'bcgs', 'ksp_ksp_max_it': 100, 'ksp_pc_type': 'jacobi'}
        #                     pc1stuff(pc1, 'ksp')
        #                 elif(pc2 == 'gmres'):
        #                     pc2t = {'pc_ksp_type': 'gmres', 'ksp_max_it': 1, 'pc_ksp_rtol' : 1e-1, "pc_ksp_pc_type": "sor"}
        #                     pc1stuff(pc1, 'ksp')
        #=======================================================================
                                                 
        num = len(settings)
        if(comm.rank == model_rank):
            print(f'Expected max time: approximately {num*maxTime} seconds')
            for i in range(num):
                print(f'Settings {i}:', settings[i])
        sys.stdout.flush()
        
        omegas = np.arange(num) ## Number of the setting being varied, if it is not a numerical quantity
        ts = np.zeros(num)
        its = np.zeros(num)
        norms = np.zeros(num)
        mems = np.zeros(num) ## to get the memories for each run, use psutil on each process with sampling every 0.5 seconds
        for i in range(num):
            if(comm.rank == model_rank):
                print('\033[94m' + f'Run {i+1}/{num} with settings:' + '\033[0m', settings[i])
            sys.stdout.flush()
            
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
                prob = scatteringProblem.Scatt3DProblem(comm, refMesh, verbosity=0.5, name=runName, MPInum=MPInum, makeOptVects=False, excitation='planewave', material_epsrs=[2.0*(1 - 0.01j)], Nf=1, fem_degree=deg, solver_settings=settings[i], max_solver_time=maxTime)
                memGather = comm.gather(process_mem, root=model_rank)
                if(comm.rank == model_rank):
                    memTotal = sum(memGather) ## keep the total usage. Only the master rank should be used, so this should be fine
                if(comm.rank == model_rank):
                    ts[i] = prob.calcTime
                    its[i] = prob.solver_its
                    norms[i] = prob.solver_norm
                    mems[i] = memTotal #prob.memCost
            except Exception as error: ## if the solver isn't defined or something, try skipping it
                if(comm.rank == model_rank):
                    print('\033[31m' + 'Warning: solver failed' + '\033[0m', error)
                    ts[i] = np.nan
                    its[i] = np.nan
                    norms[i] = np.nan
                    mems[i] = np.nan
                sys.stdout.flush()
            process_done = True
                    
        if(comm.rank == model_rank):
            fig, ax1 = plt.subplots()
            fig.subplots_adjust(right=0.45)
            fig.set_size_inches(29.5, 14.5)
            ax2 = ax1.twinx()
            ax3 = ax1.twinx()
            ax3.spines.right.set_position(("axes", 1.2))
            ax3.set_yscale('log')
            ax1.grid()
            
            l1, = ax1.plot(omegas, its, label = 'Iterations', linewidth = 2, color='tab:red')
            l2, = ax2.plot(omegas, ts, label = 'Time [s]', linewidth = 2, color='tab:blue')
            l3, = ax3.plot(omegas, norms, label = 'norms', linewidth = 2, color = 'orange')
            
            plt.title(f'Solver Time by Setting (fem_degree={prob.fem_degree}, h={h:.2f}, dofs={prob.FEMmesh_ref.ndofs:.2e}), ({testName})')
            ax1.set_xlabel(r'Setting (composite try #)')
            ax1.set_ylabel('#')
            ax2.set_ylabel('Time [s]')
            ax3.set_ylabel('log10(Norms)')
            
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
            plt.savefig(prob.dataFolder+prob.name+testName+'_solversettingsplot.png')
            
            if(num>9):
                print('Top 10 Options #s:') ## fastest options that seemed to converge
                ts[norms>4e-4] = ts[norms>4e-4] + 10000
                idxsort = np.argsort(ts)
                for k in range(10):
                    print(f'#{idxsort[k]+1}: t={ts[idxsort[k]]:.3e}, norm={norms[idxsort[k]]}, mem={mems[idxsort[k]]:.3e}GB --- ')
                    print(settings[idxsort[k]])
                    print()
                    
            
            plt.show()
            
    def reconstructionErrorTestPlots(sims = True): ## Runs some basic simulations, comparing the reconstructions errors with different FEM degrees and mesh sizes. If sims, compute results. If not, postprocess and plot
        errs3 = []; dofs3 = []
        for oh in np.linspace(2, 3.8, 7): ## degree 3
            runName = f'degree3ho{oh:.1f}'
            if(sims):
                if(os.path.exists(folder+runName+'output.npz')):
                    if(comm.rank == model_rank):
                        print(f'{runName} already completed...') ## if it already exists, skip it
                else:
                    if(comm.rank == model_rank):
                        print(f'running {runName}...') ## if it already exists, skip it
                    testFullExample(h=1/oh, degree=3, runName=runName)
            else:
                errs3.append(postProcessing.solveFromQs(folder+runName, onlyAPriori=False, returnResults=[3,4,25,28]))
                load = np.load(folder+runName+'output.npz')
                dofs3.append(load['ndofs'])
                 
        errs2 = []; dofs2 = []
        for oh in np.linspace(2.5, 6.4, 7): ## degree 2
            runName = f'degree2ho{oh:.1f}'
            if(sims):
                if(os.path.exists(folder+runName+'output.npz')):
                    if(comm.rank == model_rank):
                        print(f'{runName} already completed...') ## if it already exists, skip it ## if it already exists, skip it
                else:
                    testFullExample(h=1/oh, degree=2, runName=runName)
                if(comm.rank == model_rank):
                        print(f'running {runName}...') ## if it already exists, skip it
            else:
                errs2.append(postProcessing.solveFromQs(folder+runName, onlyAPriori=False, returnResults=[3,4,25,28]))
                load = np.load(folder+runName+'output.npz')
                dofs2.append(load['ndofs'])
                 
        errs1 = []; dofs1 = []
        for oh in np.linspace(4, 11, 7): ## degree 1
            runName = f'degree1ho{oh:.1f}'
            if(sims):
                if(os.path.exists(folder+runName+'output.npz')):
                    if(comm.rank == model_rank):
                        print(f'{runName} already completed...') ## if it already exists, skip it ## if it already exists, skip it
                else:
                    testFullExample(h=1/oh, degree=1, runName=runName)
                if(comm.rank == model_rank):
                        print(f'running {runName}...') ## if it already exists, skip it
            else:
                errs1.append(postProcessing.solveFromQs(folder+runName, onlyAPriori=False, returnResults=[3,4,25,28]))
                load = np.load(folder+runName+'output.npz')
                dofs1.append(load['ndofs'])
        
        if(not sims): ## make the plot(s)
            dofs = {'1': dofs1, '2': dofs2, '3': dofs3} ## should be [meshsize]
            errs = {'1': np.array(errs1), '2': np.array(errs2), '3': np.array(errs3)} ## should be [meshsize, result]
            if(comm.rank == model_rank):
                for degree in [1, 2, 3]:
                    fig = plt.figure()
                    ax1 = plt.subplot(1, 1, 1)
                    
                    ax1.plot(dofs[f'{degree}'], errs[f'{degree}'][:, 0], label='SVD_ap')
                    ax1.plot(dofs[f'{degree}'], errs[f'{degree}'][:, 1], label='SVD')
                    ax1.plot(dofs[f'{degree}'], errs[f'{degree}'][:, 2], label='spgl lasso_ap')
                    ax1.plot(dofs[f'{degree}'], errs[f'{degree}'][:, 3], label='spgl lasso')
                     
                    ax1.legend()
                    ax1.grid(True)
                    plt.xlabel('# dofs')
                    plt.ylabel('Reconstruction Error')
                    plt.title(f'Degree {degree} reconstruction errors')
                    plt.savefig(folder+runName+f'reconstructioncomparisonsdeg{degree}.png')
                    
    def reconstructionMeshSizeTesting(sim = 1, dutOnRefMesh=True):
        '''
        Runs the basic simulation, performing the reconstruction with different mesh sizes to see what seems best.
        :param sim: =0 to run the simulation, =1 for reconstruction to a new mesh by interpolating saved As, =2 for reconstruction to a new mesh by interpolating E-fields
        '''
        runName='reconstructionMeshSizeTesting'
        Nants = 9
        degree = 3
        mesh_settings={'h': 1/3.5, 'N_antennas': Nants, 'viewGMSH': False, 'antenna_type': 'patch', 'object_geom': 'simple1', 'defect_geom': 'simple1', 'defect_radius': 0.475, 'object_radius': 5, 'domain_radius': 4, 'domain_height': 1.5}
        epsrs = []
        for n in range(Nants): ## each patch has 3 dielectric zones
            epsrs.append(4.4*(1 - .11/4.4j)) ## susbtrate - patch
            epsrs.append(4.4*(1 - .11/4.4j)) ## substrate under patch
            epsrs.append(2.1*(1 - 0j)) ## coax
        if(sim==0):
            prevRuns = memTimeEstimation.runTimesMems(folder, comm, filename = filename)
            mesh_settings = {'N_antennas': 9, 'order': degree, 'object_offset': np.array([.15, .1, 0]), 'viewGMSH': False, 'defect_offset': np.array([-.04, .17, .01]), 'defect_radius': 0.175, 'defect_height': 0.3, 'antenna_type': 'patch'} | mesh_settings ## uses settings given before those specified here ## settings for the meshMaker
            prob_settings = {'E_ref_anim': True, 'E_dut_anim': False, 'E_anim_allAnts': False, 'dutOnRefMesh': dutOnRefMesh, 'ErefEdut': True, 'verbosity': verbosity, 'Nf': 11, 'computeBoth': True, 'antenna_mat_epsrs': epsrs}
            if(dutOnRefMesh):
                refMesh = meshMaker.MeshInfo(comm, folder+runName+'mesh.msh', reference = False, verbosity = verbosity, **mesh_settings)
            else:
                refMesh = meshMaker.MeshInfo(comm, folder+runName+'mesh.msh', reference = True, verbosity = verbosity, **mesh_settings)
            if(not dutOnRefMesh):
                dutMesh = meshMaker.MeshInfo(comm, folder+runName+'mesh.msh', reference = False, viewGMSH = False, verbosity = verbosity, **mesh_settings)
                prob = scatteringProblem.Scatt3DProblem(comm, refMesh, dutMesh, dutOnRefMesh=False, MPInum = MPInum, name = runName, fem_degree=degree, **prob_settings)
            else:
                prob = scatteringProblem.Scatt3DProblem(comm, refMesh, MPInum = MPInum, name = runName, fem_degree=degree, **prob_settings)
            prevRuns.memTimeAppend(prob)
            if(comm.rank == model_rank):
                print('Simulation part complete.')
                
        elif(sim==1): ## reconstruct with interpolated As, then make the plot(s)
            errs = []
            oh = np.linspace(2, 10, (10-2)*4+1)
            for h in 1/oh: ## do the reconstructions, then plot each time in case it crashes
                rec_mesh_settings = {'justInterpolationSubmesh': True, 'interpolationSubmeshSize': h, 'N_antennas': 9, 'order': 1, 'object_offset': np.array([.15, .1, 0]), 'defect_offset': np.array([-.04, .17, .01]), 'defect_radius': 0.175, 'defect_height': 0.3} | mesh_settings ## uses settings given before those specified here ## settings for the meshMaker
                recMesh = meshMaker.MeshInfo(comm, folder+runName+'mesh.msh', reference = True, verbosity = verbosity, **rec_mesh_settings)
                errs.append(postProcessing.solveFromQs(folder+runName, solutionName=f'recMeshSize_ho{1/h:.2f}', onlyAPriori=False, returnResults=[4,28], reconstructionMeshInfo=recMesh))   
        elif(sim==2): ## reconstruct with interpolated Es, then make the plot(s)
            mesh_settings = {'N_antennas': 9, 'order': degree, 'object_offset': np.array([.15, .1, 0]), 'viewGMSH': False, 'defect_offset': np.array([-.04, .17, .01]), 'defect_radius': 0.175, 'defect_height': 0.3, 'antenna_type': 'patch'} | mesh_settings ## uses settings given before those specified here ## settings for the meshMaker
            prob_settings = {'E_ref_anim': True, 'E_dut_anim': False, 'E_anim_allAnts': False, 'dutOnRefMesh': dutOnRefMesh, 'ErefEdut': True, 'verbosity': verbosity, 'Nf': 11, 'computeBoth': True, 'antenna_mat_epsrs': epsrs, 'computeImmediately': False}
            errs = []
            oh = np.linspace(2, 10, (10-2)*4+1)
            for h in 1/oh: ## do the reconstructions, then plot each time in case it crashes
                rec_mesh_settings = {'justInterpolationSubmesh': True, 'interpolationSubmeshSize': h, 'N_antennas': 9, 'order': 1, 'object_offset': np.array([.15, .1, 0]), 'defect_offset': np.array([-.04, .17, .01]), 'defect_radius': 0.175, 'defect_height': 0.3} | mesh_settings ## uses settings given before those specified here ## settings for the meshMaker
                recMesh = meshMaker.MeshInfo(comm, folder+runName+'mesh.msh', reference = True, verbosity = verbosity, **rec_mesh_settings)
                prob = scatteringProblem.Scatt3DProblem(comm, recMesh, MPInum = MPInum, name = runName, fem_degree=degree, dataFolder=folder, justInterping=True, **prob_settings)
                prob.makeOptVectors(reconstructionMesh=True)
                errs.append(postProcessing.solveFromQs(folder+runName, solutionName=f'recMeshSize_ho{1/h:.2f}', onlyAPriori=False, returnResults=[4,28]))
                
        if(comm.rank == model_rank and sim>0):
            errs = np.transpose(np.array(errs))
            fig = plt.figure()
            ax1 = plt.subplot(1, 1, 1)
            
            ax1.plot(oh, errs[0], label='SVD')
            ax1.plot(oh, errs[1], label='spgl lasso')
             
            ax1.legend()
            ax1.grid(True)
            plt.xlabel(r'1/Mesh Size (in $\lambda$)')
            plt.ylabel('Reconstruction Error')
            plt.title(f'Reconstruction errors by Mesh Size')
            plt.savefig(folder+runName+f'reconstructionMeshSizes.png')
            np.savez(f'{folder}{runName}reconstructionMeshSizeData.npz', oh=oh, errs=errs)
            print('Plotting complete.')
            plt.show()
            
    def patchSsPlot(hols): ## Makes a plot of the patch S11 vs the FEKO S11, for some given h/lambdas
        for ho in hols:
            name = f'patchPatternTest_ho{ho:.1f}'
            data = np.load(folder+name+'output.npz')
            S11 = data['S_ref'][:, 0, 0]
            fvec = data['fvec']
            
            plt.plot(fvec/1e9, 20*np.log10(np.abs(S11)), label=f'sim. ($\lambda/h={ho:.1f}$'+f')', linewidth=2)
        
        fekof = 'TestStuff/FEKO patch S11 new.dat'
        fekoData = np.transpose(np.loadtxt(fekof, skiprows = 2))
        plt.plot(fekoData[0]/1e9, 20*np.log10(np.abs(fekoData[1]+1j*fekoData[2])), label='FEKO')
        plt.grid()
        plt.ylabel(r'$|$S$_{11}|$ [dB]')
        plt.xlabel(r'Frequency [GHz]')
        plt.title(r'Patch Antenna Reflection Coefficient')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    def plotMeshSizeByErrors(plotting=False): ## plots the mesh size vs sphere-scattering near-field error, and reconstruction accuracy for the basic case (ErefEref and ErefEdut)
        meshSizes = [1/1, 1/1.5, 1/2, 1/2.5, 1/3, 1/3.5, 1/4, 1/4.5] ## h/lambda
        meshSizes = [1/1, 1/2, 1/3,  1/4, 1/5] ## h/lambda
        meshSizes = [1/1.5, 1/2.5, 1/3.5, 1/4.5, 1/5.5]
        if(plotting): ## make the plots, assuming data already made
            NFerrs = []
            ErefErefErrs = []
            ErefEdutErrs = []
            for hol in meshSizes: ## first, load in the data
                err = 1 ## first calculate the near-field error
                for index, name in [(0, 'x'), (1, 'y'), (2, 'z')]: ## scattering along the x-, y-, and z- axes
                    E_load = np.load(f'{folder}{runName}_SimulatedEs_{name}-axis.npz')['E_values']
                NFerrs.append(err)
                
                ## then calculate the ErefEref err
                #ErefErefErrs.append(postProcessing.reconstructionError(delta_epsr_rec, epsr_ref, epsr_dut, cell_volumes, indices='defect'))
                ## then the ErefEdut err
                
            
            plt.plot(meshSizes, NFerrs, label='SS N-F E')
            plt.plot(meshSizes, ErefErefErrs, label='ErefEref')
            plt.plot(meshSizes, ErefEdutErrs, label='ErefEdut')
        else:
            degree = 3
            mesh_setts = {'viewGMSH': False, 'N_antennas': 9, 'antenna_type': 'patch', 'object_geom': 'simple1', 'defect_geom': 'simple1', 'defect_radius': 0.475, 'object_radius': 4, 'domain_radius': 3, 'domain_height': 1.3, 'object_offset': np.array([.15, .1, 0]), 'defect_offset': np.array([-.04, .17, 0])}
            prob_setts = {'Nf': 13, 'material_epsrs' : [3*(1 - 0.01j)], 'defect_epsrs' : [3.1*(1 - 0.01j)]}
            for hol in meshSizes:
                runName = f'meshSizeErrRun_ho{hol}'
                if(os.path.isfile(f'{folder}{runName}_ErefEdutpost-process.xdmf')): ## check if this mesh size has already been run
                    pass ## if it has, dont run again
                else:
                    # start with SSNFEprevRuns = memTimeEstimation.runTimesMems(folder, comm, filename = filename)
                    refMesh = meshMaker.MeshInfo(comm, reference = True, viewGMSH = False, verbosity = verbosity, N_antennas=0, object_radius = .33, domain_radius=.9, PML_thickness=0.5, h=hol, domain_geom='sphere', object_geom='sphere', FF_surface = True, order=degree)
                    prob = scatteringProblem.Scatt3DProblem(comm, refMesh, verbosity=verbosity, name=runName, MPInum=MPInum, makeOptVects=True, excitation='planewave', freqs = np.array([10e9]), material_epsrs=[2.0*(1-0.01j)], fem_degree=degree)
                    prob.calcNearField(showPlots=False) ## saves the data to a file
                    
                    # then ErefEref basic case
                    testFullExample(h=hol, degree=3, runName=runName+'ErefEref', ErefEdut=False,
                                    mesh_settings=mesh_setts,
                                    prob_settings=prob_setts)
                    postProcessing.solveFromQs(folder+runName+'ErefEref', solutionName='', onlyAPriori=True)
                    
                    # then ErefEdut basic case
                    testFullExample(h=hol, degree=3, runName=runName+'ErefEdut', ErefEdut=True,
                                    mesh_settings=mesh_setts,
                                    prob_settings=prob_setts)
                    postProcessing.solveFromQs(folder+runName+'ErefEdut', solutionName='', onlyAPriori=True)
            
    #testRun(h=1/2)
    #folder = 'data3DLUNARC/'
    #reconstructionErrorTestPlots()
    #reconstructionErrorTestPlots(False)
    
    #reconstructionMeshSizeTesting(0)
    #reconstructionMeshSizeTesting(1)
    #reconstructionMeshSizeTesting(2)
    
    plotMeshSizeByErrors()
    
    
    #testFullExample(h=1/6, degree=1, antennaType='patch')
    
    #runName = 'testRunTiny' ## h=1/2
    #testFullExample(h=1/2, degree=1, runName=runName, mesh_settings={'N_antennas': 2, 'viewGMSH': False}, prob_settings={'Nf': 2})
    #postProcessing.solveFromQs(folder+runName, solutionName='', onlyAPriori=True)
    
    #runName = 'testRunDeg2' ## h=1/9.5
    #runName = 'testRunDeg2Smaller' ## h=1/6
    #runName = 'testRunSmall' ## h=1/3.5, degree 3
    #testFullExample(h=1/3.5, degree=3, runName=runName, mesh_settings={'N_antennas': 9, 'viewGMSH': False}, prob_settings={'Nf': 11})
    
    
    #===========================================================================
    # runName = 'testRunComplex2Obj'
    # testFullExample(h=1/3.5, degree=3, runName=runName,
    #                 mesh_settings={ 'viewGMSH': False, 'N_antennas': 9, 'antenna_type': 'patch', 'object_geom': 'complex2', 'defect_geom': 'complex2', 'defect_radius': 0.475, 'object_radius': 4, 'domain_radius': 3, 'domain_height': 1.3, 'object_offset': np.array([.15, .1, 0]), 'defect_offset': np.array([-.04, .17, 0])},
    #                 prob_settings={'Nf': 13, 'material_epsrs' : [3*(1 - 0.01j)], 'defect_epsrs': [2.9*(1 - 0.01j), 3.2*(1 - 0.01j), 3.1*(1 - 0.01j)]})
    #===========================================================================
    
    #===========================================================================
    # runName = 'testRunD3.3'
    # testFullExample(h=1/3, degree=3, runName=runName,
    #                 mesh_settings={'viewGMSH': False, 'N_antennas': 9, 'antenna_type': 'patch', 'object_geom': 'simple1', 'defect_geom': 'simple1', 'defect_radius': 0.475, 'object_radius': 4, 'domain_radius': 3, 'domain_height': 1.3, 'object_offset': np.array([.15, .1, 0]), 'defect_offset': np.array([-.04, .17, 0])},
    #                 prob_settings={'Nf': 26})
    #===========================================================================
    
    #===========================================================================
    # runName = 'testRunD3LowContrast'
    # testFullExample(h=1/3.5, degree=3, runName=runName,
    #                 mesh_settings={'viewGMSH': False, 'N_antennas': 9, 'antenna_type': 'patch', 'object_geom': 'simple1', 'defect_geom': 'simple1', 'defect_radius': 0.475, 'object_radius': 4, 'domain_radius': 3, 'domain_height': 1.3, 'object_offset': np.array([.15, .1, 0]), 'defect_offset': np.array([-.04, .17, 0])},
    #                 prob_settings={'Nf': 26, 'material_epsrs' : [3*(1 - 0.01j)], 'defect_epsrs' : [3.3*(1 - 0.01j)]})
    #===========================================================================
    
    #===========================================================================
    # runName = 'testRunD3LowerContrast'
    # testFullExample(h=1/3.5, degree=3, runName=runName,
    #                 mesh_settings={'viewGMSH': False, 'N_antennas': 9, 'antenna_type': 'patch', 'object_geom': 'simple1', 'defect_geom': 'simple1', 'defect_radius': 0.475, 'object_radius': 4, 'domain_radius': 3, 'domain_height': 1.3, 'object_offset': np.array([.15, .1, 0]), 'defect_offset': np.array([-.04, .17, 0])},
    #                 prob_settings={'Nf': 26, 'material_epsrs' : [3*(1 - 0.01j)], 'defect_epsrs' : [3.1*(1 - 0.01j)]})
    #===========================================================================
    
    #===========================================================================
    # runName = 'testRunD3LowerContrastQsView'
    # testFullExample(h=1/3.5, degree=3, runName=runName, recMesh=False,
    #                 mesh_settings={'viewGMSH': False, 'N_antennas': 9, 'antenna_type': 'patch', 'object_geom': 'simple1', 'defect_geom': 'simple1', 'defect_radius': 0.475, 'object_radius': 4, 'domain_radius': 3, 'domain_height': 1.3, 'object_offset': np.array([.15, .1, 0]), 'defect_offset': np.array([-.04, .17, 0])},
    #                 prob_settings={'Nf': 13, 'material_epsrs' : [3*(1 - 0.01j)], 'defect_epsrs' : [3.1*(1 - 0.01j)]})
    #===========================================================================
    
    #postProcessing.solveFromQs(folder+runName, solutionName='', onlyAPriori=True)#, frequenciesToUse=[2, 4, 6, 8, 12, 14, 16, 18, 20, 22], returnResults=[3, 25])
    
    #runName = 'testRunLargeAsPossible2'
    #testFullExample(h=1/3, degree=3, runName=runName, mesh_settings = {'domain_radius': 9, })
    
    #postProcessing.solveFromQs(folder+runName, solutionName='', onlyAPriori=True, returnResults=[99])
    #runName = 'testRunSmall_ypol' ## h=1/3.5, degree 3
    #testFullExample(h=1/3.5, degree=3, runName=runName)
    
    #runName = 'testRunPatchesEFields' ## test run to save all E-fields
    #testFullExample(h=1/3, degree=3, runName=runName, mesh_settings={'N_antennas': 5, 'object_geom': '', 'defect_geom': '', 'antenna_type': 'patch'}, prob_settings={'freqs': [10e9], 'makeOptVects':False, 'E_ref_anim':True, 'E_dut_anim':True, 'E_anim_allAnts':True, 'computeBoth': False})
    
    #runName = 'testRunPatches' ## h=1/3.5, degree 3
    #testFullExample(h=1/3.5, degree=3, antennaType='patch', runName=runName)
    #postProcessing.solveFromQs(folder+runName, solutionName='', onlyAPriori=True)
    
    #postProcessing.solveFromQs(folder+'testRunSmall_ypol', folder+'testRunPatches', solutionName='SsFromPatches', onlyAPriori=True) ## aka testRunDifferentDUTAntennas2
    
    #runName = 'testRunDifferentDUTAntennas' ## h=1/3.6, d3
    #testRunDifferentDUTAntennas(h=1/3.6, degree=3)
    
    #testFullExample(h=1/8, degree=1)
    #postProcessing.solveFromQs(folder+runName, solutionName='', onlyAPriori=False)
    
    #postProcessing.solveFromQs(folder+runName, solutionName='4antennas', antennasToUse=[1, 3, 5, 7])
    #postProcessing.solveFromQs(folder+runName, solutionName='just2antennas', onlyNAntennas=2)
    #postProcessing.solveFromQs(folder+runName, solutionName='just4antennas', onlyNAntennas=4)
    #postProcessing.solveFromQs(folder+runName, solutionName='4freqs', frequenciesToUse=[2, 4, 6, 8])
    #postProcessing.solveFromQs(folder+runName, solutionName='4freqs4antennas', antennasToUse=[1, 3, 5, 7], frequenciesToUse=[2, 4, 6, 8])
    
    #patchConvergenceTestPlots(degree=1)
    
    #testSphereScattering(h=1/3.5, degree=3, showPlots=True)
    #convergenceTestPlots('pmlR0')
    #convergenceTestPlots('meshsize', deg=3)
    #convergenceTestPlots('dxquaddeg')
    #testSolverSettings(h=1/6)
    
    #runName = 'patchPatternTest' #'patchPatternTest_ho8.0' #patchPatternTestd2small', h=1/10 'patchPatternTestd2', h=1/5.6 #'patchPatternTestd1' , h=1/15  #'patchPatternTestd3'#, h=1/3.4 #'patchPatternTestd3smaller'#, h=1/6
    #testPatchPattern(h=1/3.5, degree=3, freqs = np.linspace(8e9, 12e9, 50), name=runName, showPlots=False)
    #testPatchPattern(h=1/1, degree=1, name=runName, showPlots=True) ## plot the FF comp. with Feko
    #postProcessing.solveFromQs(folder+runName, solutionName='', onlyAPriori=True, plotSs=True) ## inspect the S11
    
    #patchSsPlot([3.5, 8]) ## plot S11 comp. with Feko
    
    #runName = 'testingComplexObject' ## h=1/8
    #testLargeExample(h=1/6, degree=2)
    #postProcessing.solveFromQs(folder+runName)
    
    #runName = 'testingShiftedDut' ## h=1/12
    #testShiftedExample(h=1/12, degree=1)
    #postProcessing.solveFromQs(folder+runName
    
    #===========================================================================
    # runName = 'testingSlightlyShiftedDutNoDefects' ## h=1/12
    # testSlightlyShiftedExampleNoDefects(h=1/12, degree=1)
    # postProcessing.solveFromQs(folder+runName)
    #===========================================================================
    
    
    #===========================================================================
    # for k in np.arange(10, 35, 4):
    #     runName = 'test-fem3h'+str(k)
    #     testSphereScattering(h=1/k)
    #===========================================================================
    
    #===========================================================================
    # for k in range(15, 40, 2):
    #     runName = 'testRunbiggerdomainfaraway(10hawayFF)FF'+str(k)
    #     testSphereScattering(h=1/k)
    #===========================================================================
    
    #filename = 'prevRuns.npz'
    otherprevs = [] ## if adding other files here, specify here (i.e. prevRuns.npz.old)
    #prevRuns = memTimeEstimation.runTimesMems(folder, comm, otherPrevs = otherprevs, filename = filename)
    #prevRuns.makePlots(MPInum = comm.size)
    
    if(comm.rank == model_rank):
        print(f'runScatt3D complete in {timer()-t1:.2f} s ({(timer()-t1)/3600:.2f} hours), exiting...')
        sys.stdout.flush()