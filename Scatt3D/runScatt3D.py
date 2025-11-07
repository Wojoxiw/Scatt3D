# encoding: utf-8
### Modification of scatt2d to handle 3d geometry
# Stripped down and rewritten for DD2358 course
#
# Adapted from 2D code started by Daniel Sjoberg, (https://github.com/dsjoberg-git/rotsymsca, https://github.com/dsjoberg-git/ekas3d) approx. 2024-12-13 
# Alexandros Pallaris, after that
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
    else:
        filename = 'prevRuns.npz'
        
    runName = 'testRun' # testing
    folder = 'data3D/'
    if(verbosity>2):
        print(f"{comm.rank=} {comm.size=}, {MPI.COMM_SELF.rank=} {MPI.COMM_SELF.size=}, {MPI.Get_processor_name()=}")
    if(comm.rank == model_rank):
        print(f'runScatt3D starting with {MPInum} MPI process(es) (main process on {MPI.Get_processor_name()=}):')
    sys.stdout.flush()
    
    def profilingMemsTimes(): ## as used to make plots for the report
        prevRuns = memTimeEstimation.runTimesMems(folder, comm, filename = '8nodes24MPI1threads2b.npz') ## make sure to change to filename so it doesn't get overwritten - the data is stored here
        numRuns = 1 ## run these 10 times to find averages/stds
        hs = [1/10, 1/11, 1/12, 1/13, 1/14, 1/15, 1/16, 1/17, 1/18, 1/19, 1/20] ## run it for different mesh sizes
        for i in range(numRuns):
            if(comm.rank == model_rank):
                print('############')
                print(f'  RUN {i+1}/{numRuns} ')
                print('############')
            for h in hs:
                refMesh = meshMaker.MeshInfo(comm, folder+runName+'mesh.msh', reference = True, viewGMSH = False, verbosity = verbosity, h=h)
                prob = scatteringProblem.Scatt3DProblem(comm, refMesh, verbosity = verbosity, MPInum = MPInum)
                prevRuns.memTimeAppend(prob, '8nodes24MPI1threads2b')
    
    def actualProfilerRunning(): # Here I call more things explicitly in order to more easily profile the code in separate methods (profiling activated in the methods themselves also).
        refMesh = meshMaker.MeshInfo(comm, folder+runName+'mesh.msh', reference = True, viewGMSH = False, verbosity = verbosity, h=1/10) ## this will have around 190000 elements
        prob = scatteringProblem.Scatt3DProblem(comm, refMesh, verbosity = verbosity, MPInum = MPInum)
            
    def testRun(h = 1/2): ## A quick test run to check it works. Default settings make this run in a second
        prevRuns = memTimeEstimation.runTimesMems(folder, comm, filename = filename)
        refMesh = meshMaker.MeshInfo(comm, folder+runName+'mesh.msh', reference = True, viewGMSH = False, verbosity = verbosity, h=h, object_geom='sphere', N_antennas=1)
        prevRuns.memTimeEstimation(refMesh.ncells, doPrint=True, MPInum = comm.size)
        #refMesh.plotMeshPartition()
        prob = scatteringProblem.Scatt3DProblem(comm, refMesh, verbosity = verbosity, MPInum = MPInum, name = runName, Nf=1)
        #prob.saveEFieldsForAnim()
        prevRuns.memTimeAppend(prob)
        
    def testFullExample(h = 1/15, degree = 1, dutOnRefMesh=True): ## Testing toward a full example
        prevRuns = memTimeEstimation.runTimesMems(folder, comm, filename = filename)
        settings = {'N_antennas': 9, 'order': degree, 'object_offset': np.array([.15, .1, 0]), 'defect_offset': np.array([-.04, .17, .01])} ## settings for the meshMaker
        if(dutOnRefMesh):
            refMesh = meshMaker.MeshInfo(comm, folder+runName+'mesh.msh', reference = False, viewGMSH = False, verbosity = verbosity, h=h, **settings)
        else:
            refMesh = meshMaker.MeshInfo(comm, folder+runName+'mesh.msh', reference = True, viewGMSH = False, verbosity = verbosity, h=h, **settings)
        #dutMesh = meshMaker.MeshInfo(comm, folder+runName+'mesh.msh', reference = False, viewGMSH = False, verbosity = verbosity, h=h, **settings)
        #prevRuns.memTimeEstimation(refMesh.ncells, doPrint=True, MPInum = comm.size)
        #refMesh.plotMeshPartition()
        prob = scatteringProblem.Scatt3DProblem(comm, refMesh, computeBoth=True, verbosity = verbosity, MPInum = MPInum, name = runName, Nf = 11, fem_degree=degree, ErefEdut=True, dutOnRefMesh=dutOnRefMesh)
        #prob.saveEFieldsForAnim(True)
        #prob.saveEFieldsForAnim(False)
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
        prob.saveEFieldsForAnim(True)
        prob.saveEFieldsForAnim(False)
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
        prob.saveEFieldsForAnim(True)
        prob.saveEFieldsForAnim(False)
        prevRuns.memTimeAppend(prob)
        
    def testLargeExample(h = 1/15, degree = 1, dutOnRefMesh=True): ## Testing a large-object example
        prevRuns = memTimeEstimation.runTimesMems(folder, comm, filename = filename)
        settings = {'N_antennas': 9, 'order': degree, 'object_geom': 'complex1', 'defect_geom': 'complex1', 'h': h, 'defect_height': .37, 'defect_radius': .17, 'object_radius': 1.06} ## settings for the meshMaker 'object_offset': np.array([.15, .1, 0])
        if(dutOnRefMesh):
            refMesh = meshMaker.MeshInfo(comm, folder+runName+'mesh.msh', reference = False, viewGMSH = False, verbosity = verbosity, **settings)
        else:
            refMesh = meshMaker.MeshInfo(comm, folder+runName+'mesh.msh', reference = True, viewGMSH = False, verbosity = verbosity, **settings)
        #dutMesh = meshMaker.MeshInfo(comm, folder+runName+'mesh.msh', reference = False, viewGMSH = False, verbosity = verbosity, h=h, N_antennas=9, order=degree)
        #prevRuns.memTimeEstimation(refMesh.ncells, doPrint=True, MPInum = comm.size)
        #refMesh.plotMeshPartition()
        prob = scatteringProblem.Scatt3DProblem(comm, refMesh, computeBoth=True, verbosity = verbosity, MPInum = MPInum, name = runName, Nf = 11, fem_degree=degree, ErefEdut=True, dutOnRefMesh=dutOnRefMesh)
        prob.makeOptVectors(skipQs=True)
        prob.saveEFieldsForAnim(True)
        prob.saveEFieldsForAnim(False)
        prevRuns.memTimeAppend(prob)
        degree = 1
        
        
    def testSphereScattering(h = 1/12, degree=1, showPlots=False): ## run a spherical domain and object, test the far-field scattering for an incident plane-wave from a sphere vs Mie theoretical result.
        prevRuns = memTimeEstimation.runTimesMems(folder, comm, filename = filename)
        refMesh = meshMaker.MeshInfo(comm, reference = True, viewGMSH = False, verbosity = verbosity, N_antennas=0, object_radius = .33, domain_radius=.9, PML_thickness=0.5, h=h, domain_geom='sphere', object_geom='sphere', FF_surface = True, order=degree)
        #refMesh.plotMeshPartition()
        #prevRuns.memTimeEstimation(refMesh.ncells, doPrint=True, MPInum = comm.size)
        freqs = np.linspace(10e9, 12e9, 1)
        prob = scatteringProblem.Scatt3DProblem(comm, refMesh, verbosity=verbosity, name=runName, MPInum=MPInum, makeOptVects=True, excitation='planewave', freqs = freqs, material_epsr=2.0*(1-0.01j), fem_degree=degree)
        #prob.saveEFieldsForAnim()
        if(showPlots):
            prob.calcNearField(direction='side')
        prob.calcFarField(reference=True, compareToMie = True, showPlots=showPlots, returnConvergenceVals=False)
        prevRuns.memTimeAppend(prob)
        
    def testPatchPattern(h = 1/12, degree=1): ## run a spherical domain and object, test the far-field pattern from a single patch antenna near the center
        runName = 'patchPatternTest'
        prevRuns = memTimeEstimation.runTimesMems(folder, comm, filename = filename)
        refMesh = meshMaker.MeshInfo(comm, reference = True, viewGMSH = False, verbosity = verbosity, N_antennas=1, domain_radius=1.8, PML_thickness=0.5, h=h, domain_geom='sphere', antenna_type='patch', antenna_depth=.5, antenna_height=.05, antenna_width=.2, object_geom='', FF_surface = True, order=degree)
        #refMesh.plotMeshPartition()
        #prevRuns.memTimeEstimation(refMesh.ncells, doPrint=True, MPInum = comm.size)
        freqs = np.linspace(10e9, 12e9, 1)
        prob = scatteringProblem.Scatt3DProblem(comm, refMesh, verbosity=verbosity, name=runName, MPInum=MPInum, makeOptVects=True, excitation='patch', freqs = freqs, fem_degree=degree)
        prob.saveEFieldsForAnim()
        prob.calcFarField(reference=True, plotFF=True, showPlots=True)
        prevRuns.memTimeAppend(prob)
 
    def convergenceTestPlots(convergence = 'meshsize', deg=1): ## Runs with reducing mesh size, for convergence plots. Uses the far-field surface test case. If showPlots, show them - otherwise just save them
        if(convergence == 'meshsize'):
            ks = np.linspace(5, 14, 13)
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
            
            refMesh = meshMaker.MeshInfo(comm, reference = True, viewGMSH = False, verbosity = verbosity, N_antennas=0, object_radius = .33, PML_thickness=0.5, domain_radius=0.9, domain_geom='sphere', FF_surface = True, order=deg, **meshOptions)
            prob = scatteringProblem.Scatt3DProblem(comm, refMesh, verbosity = verbosity, name=runName, MPInum = MPInum, makeOptVects=False, excitation = 'planewave', material_epsr=2.0*(1-0.01j), Nf=1, fem_degree=deg, **probOptions)
            newval, khats, farfields, mies = prob.calcFarField(reference=True, compareToMie = False, showPlots=False, returnConvergenceVals=True) ## each return is FF surface area, khat integral at each angle, farfields+mies at each angle
            simNF, FEKONF = prob.calcNearField(direction='side', FEKOcomp=True, showPlots=False)
            prevRuns.memTimeAppend(prob)
            if(comm.rank == model_rank): ## only needed for main process
                areaVals.append(newval)
                ndofs[i] = prob.ndofs
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
                real_area = 4*pi*prob.refMeshdata.FF_surface_radius**2
                
                
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
                prob = scatteringProblem.Scatt3DProblem(comm, refMesh, verbosity=0.5, name=runName, MPInum=MPInum, makeOptVects=False, excitation='planewave', material_epsr=2.0*(1 - 0.01j), Nf=1, fem_degree=deg, solver_settings=settings[i], max_solver_time=maxTime)
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
    
    
    #runName = 'testRunDeg2' ## h=1/9.5
    #runName = 'testRunDeg2Smaller' ## h=1/6
    #runName = 'testRunSmall' ## h=1/8
    #runName = 'testRunLarger' ## h=1/18
    
    #testRun(h=1/2)
    #profilingMemsTimes()
    #actualProfilerRunning()
    
    #testFullExample(h=1/13, degree=1)
    #postProcessing.solveFromQs(folder+runName, MPInum, solutionName='')
    
    #postProcessing.solveFromQs(folder+runName, MPInum, solutionName='4antennas', antennasToUse=[1, 3, 5, 7])
    #postProcessing.solveFromQs(folder+runName, MPInum, solutionName='4freqs', frequenciesToUse=[2, 4, 6, 8])
    #postProcessing.solveFromQs(folder+runName, MPInum, solutionName='4freqs4antennas', antennasToUse=[1, 3, 5, 7], frequenciesToUse=[2, 4, 6, 8])
    
    #testSphereScattering(h=1/3.66, degree=3, showPlots=True)
    #convergenceTestPlots('pmlR0')
    #convergenceTestPlots('meshsize', deg=3)
    #convergenceTestPlots('dxquaddeg')
    #testSolverSettings(h=1/6)
    
    #testPatchPattern(h=1/10, degree=1)
    
    #===========================================================================
    # runName = 'testingComplexObject' ## h=1/12
    # testLargeExample(h=1/12, degree=1)
    # postProcessing.solveFromQs(folder+runName, MPInum)
    #===========================================================================
    
    #runName = 'testingShiftedDut' ## h=1/12
    #testShiftedExample(h=1/12, degree=1)
    #postProcessing.solveFromQs(folder+runName, MPInum)
    
    #===========================================================================
    # runName = 'testingSlightlyShiftedDutNoDefects' ## h=1/12
    # testSlightlyShiftedExampleNoDefects(h=1/12, degree=1)
    # postProcessing.solveFromQs(folder+runName, MPInum)
    #===========================================================================
    
    for oh in np.arange(2.5, 5, 7): ## degree 3
        h = 1/oh
        runName = f'degree3ho{oh:.1f}'
        testFullExample(h=1/oh, degree=3)
         
    #===========================================================================
    # for oh in np.arange(3, 7.45, 7): ## degree 2
    #     h = 1/oh
    #     runName = f'degree3ho{oh:.1f}'
    #     testFullExample(h=1/oh, degree=3)
    #      
    # for oh in np.arange(3, 17, 7): ## degree 1
    #     h = 1/oh
    #     runName = f'degree3ho{oh:.1f}'
    #     testFullExample(h=1/oh, degree=3)
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
    #prevRuns.makePlotsSTD()
    
    if(comm.rank == model_rank):
        print(f'runScatt3D complete in {timer()-t1:.2f} s, exiting...')
        sys.stdout.flush()