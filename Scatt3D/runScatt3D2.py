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
        folder = 'data3DLUNARC/'
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
        folder = 'data3D/'
        
    runName = 'testRun' # testing - the default name
    if(verbosity>2):
        print(f"{comm.rank=} {comm.size=}, {MPI.COMM_SELF.rank=} {MPI.COMM_SELF.size=}, {MPI.Get_processor_name()=}")
    if(comm.rank == model_rank):
        print(f'runScatt3D starting with {MPInum} MPI process(es) (main process on {MPI.Get_processor_name()=}):')
    sys.stdout.flush()
            
    def measurementScript(h = 1/3.5, degree = 3, runName=runName, angles = np.arange(0, 360, 20, dtype=float), mesh_settings={}, prob_settings={}, dutForSimSolution=False):
        ## For measurements with patch antennas of a rectangular block. Four antennas, which are rotated by 30 degree steps to cover 360 degrees.
        prevRuns = memTimeEstimation.runTimesMems(folder, comm, filename = filename)
        mesh_settings = {'h': h, 'N_antennas': 4, 'order': degree, 'antenna_type': '6GHz measurement'} | mesh_settings ## uses settings given before those specified here ## settings for the meshMaker
        prob_settings = {'E_ref_anim': True, 'E_dut_anim': True, 'E_anim_allAnts': False, 'ErefEdut': False, 'verbosity': verbosity, 'dataFolder': folder, 'computeBoth': False, 'makeOptVects': False} | prob_settings
        
        epsrs=[]
        for n in range(mesh_settings['N_antennas']): ## each patch has 3 dielectric zones
            epsrs.append(4.4*(1 - .11/4.4j)) ## box
            epsrs.append(4.4*(1 - .11/4.4j)) ## patch
            epsrs.append(2.1*(1 - 0.01j)) ## coax dielectric
            epsrs.append(2.7*(1 - 0.01j)) ## PLA printed holder
        prob_settings = prob_settings | {'antenna_mat_epsrs': epsrs}
        
        if(dutForSimSolution): ## test-case simulation
            for angle in angles:
                if(os.path.isfile(folder+runName+f'_angle{angle}'+'output-qs.xdmf')): ## check if the angle has already been run
                    if(comm.rank == model_rank):
                        print(f'Already computed run with {angle=}, skipping...')
                    continue
                else:
                    if(comm.rank == model_rank):
                        print(f'Computing run with angle={angle}:')
                testMesh = meshMaker.MeshInfo(comm, folder+runName+f'_angle{angle}'+'mesh.msh', reference = False, verbosity = verbosity, phi_antennas=-angle, **mesh_settings)
                prob = scatteringProblem.Scatt3DProblem(comm, testMesh, testMesh, MPInum = MPInum, name = runName+f'_angle{angle}', fem_degree=degree, computeImmediately=False, dutOnRefMesh=False, **prob_settings)
                prob.compute(computeRef=False, makeOptVects=False)
                prob.makeOptVectors(DUTMesh=True, skipQs=True)
        else: ## reference simulation
            rec_mesh_settings = {'justInterpolationSubmesh': True, 'interpolationSubmeshSize': 1/10} | mesh_settings | {'viewGMSH': False} ## uses settings given before those specified here ## settings for the meshMaker
            recMesh = meshMaker.MeshInfo(comm, folder+runName+'mesh.msh', reference = True, verbosity = verbosity, **rec_mesh_settings)
            for angle in angles: ## 20 degree spacing. Should rotate in opposite direction to measurements, since this rotates the antennas while measurements rotate the object. (this way the E-fields in the object are all aligned)
                if(os.path.isfile(folder+runName+f'_angle{angle}'+'output-qs.xdmf')): ## check if the angle has already been run
                    if(comm.rank == model_rank):
                        print(f'Already computed run with {angle=}, skipping...')
                    continue
                else:
                    if(comm.rank == model_rank):
                        print(f'Computing run with angle={angle}:')
                refMesh = meshMaker.MeshInfo(comm, folder+runName+f'_angle{angle}'+'mesh.msh', reference = True, verbosity = verbosity, phi_antennas=-angle, **mesh_settings)
                #prevRuns.memTimeEstimation(refMesh.ncells, doPrint=True, MPInum = comm.size)
                #refMesh.plotMeshPartition()
                prob = scatteringProblem.Scatt3DProblem(comm, refMesh, MPInum = MPInum, name = runName+f'_angle{angle}', fem_degree=degree, **prob_settings)
                prob.makeOptVectors(reconstructionMesh=False, saveName=prob.name+'_regMesh') ## to check if everything is correct
                ## make the opt vects on the rec mesh
                prob.switchToRecMesh(recMesh)
                prob.makeOptVectors(reconstructionMesh=True)
                prob.deleteSol() ## remove saved E-fields afterward, since this generates too much data to store on the cluster easily
                prevRuns.memTimeAppend(prob)
        return prob
    
    def testPatchPattern(h = 1/3.5, degree=3, freqs = np.array([6e9]), name='6GHzpatchPatternTest', showPlots=True): ## run a spherical domain and object, test the far-field pattern from a single patch antenna near the center
        runName = name
        prevRuns = memTimeEstimation.runTimesMems(folder, comm, filename = filename)
        refMesh = meshMaker.MeshInfo(comm, reference = True, viewGMSH = True, verbosity = verbosity, N_antennas=1, domain_radius=1.8, PML_thickness=0.5, h=h, domain_geom='sphere', antenna_radius=0, antenna_type='6GHz measurement', object_geom='', defect_geom='', FF_surface = True, order=degree)
        epsrs=[]
        epsrs.append(4.4*(1 - .11/4.4j)) ## susbtrate - patch
        epsrs.append(4.4*(1 - .11/4.4j)) ## substrate under patch
        epsrs.append(2.1*(1 - 0.01j))
        epsrs.append(2.7*(1 - 0.01j))
        #refMesh.plotMeshPartition()
        #prevRuns.memTimeEstimation(refMesh.ncells, doPrint=True, MPInum = comm.size)
        if(len(freqs) == 1): ## plot the given frequency, if there is only 1
            prob = scatteringProblem.Scatt3DProblem(comm, refMesh, verbosity=verbosity, name=runName, MPInum=MPInum, makeOptVects=False, freqs = freqs, fem_degree=degree, antenna_mat_epsrs=epsrs)
            prob.calcFarField(reference=True, plotFF=True, showPlots=showPlots)
        else: ## save Ss
            prob = scatteringProblem.Scatt3DProblem(comm, refMesh, verbosity=verbosity, name=runName, MPInum=MPInum, makeOptVects=True, freqs = freqs, fem_degree=degree, antenna_mat_epsrs=epsrs)
        prevRuns.memTimeAppend(prob)
    
    def patchSsPlot(hols): ## Makes a plot of the patch S11 vs the FEKO S11, for some given h/lambdas
        colors = ['tab:blue', 'tab:orange']
        markers = ['o', 'v']
        i=0
        measFolder = '/mnt/c/Users/al8032pa/Work Folders/Documents/antenna measurements/Microwave Imaging/Patch Data/'
        
        for ho in hols:
            name = f'6GHzpatchPatternTest_ho{ho:.1f}'
            data = np.load(folder+name+'output.npz')
            S11 = data['S_ref'][:, 0, 0]
            fvec = data['fvec']
             
            plt.plot(fvec/1e9, 20*np.log10(np.abs(S11)), label=rf'sim. ($\lambda/h={ho:.1f}$'+f')', linewidth=2, color=colors[i], marker=markers[i], markevery=10-i, markersize=8)
            i = i+1
        
        fekof = measFolder+'feko patch S11.dat'
        fekoData = np.transpose(np.loadtxt(fekof, skiprows = 2))
        plt.plot(fekoData[0]/1e9, 20*np.log10(np.abs(fekoData[1]+1j*fekoData[2])), label='FEKO', color='tab:purple')#, marker='+', markevery=8, markersize=10)
        
        
        for patch in ['1', '2', '3', '4']:
            measData = np.transpose(np.loadtxt(measFolder+'Patches S11 before holders/'+patch+'.csv', skiprows = 3))
            plt.plot(measData[0]/1e9, 20*np.log10(np.abs(measData[1]+1j*measData[2])), label='Meas.'+patch)#, color='tab:green', marker='+', markevery=8, markersize=10)
        
        plt.grid()
        plt.ylabel(r'$|$S$_{11}|$ [dB]')
        plt.xlabel(r'Frequency [GHz]')
        plt.title(r'Patch Antenna Reflection Coefficient')
        plt.legend()
        plt.tight_layout()
        
        for patch in ['1', '2', '3', '4']:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            i=0
            
            for ho in hols:
                name = f'6GHzpatchPatternTest_ho{ho:.1f}'
                data = np.load(folder+name+'output.npz')
                S11 = data['S_ref'][:, 0, 0]
                fvec = data['fvec']
                 
                ax1.plot(fvec/1e9, np.unwrap(np.angle(S11)), label=rf'sim. ($\lambda/h={ho:.1f}$'+f')', linewidth=2, color=colors[i], marker=markers[i], markevery=10-i, markersize=8)
                ax2.plot(fvec/1e9, 20*np.log10(np.abs(S11)), label=rf'sim. ($\lambda/h={ho:.1f}$'+f')', linewidth=2, color=colors[i], marker=markers[i], markevery=10-i, markersize=8)
                i = i+1
            
            fekof = measFolder+'feko patch S11.dat'
            fekoData = np.transpose(np.loadtxt(fekof, skiprows = 2))
            ax1.plot(fekoData[0]/1e9, np.unwrap(np.angle(fekoData[1]+1j*fekoData[2])), label='FEKO', color='tab:purple')#, marker='+', markevery=8, markersize=10)
            ax2.plot(fekoData[0]/1e9, 20*np.log10(np.abs(fekoData[1]+1j*fekoData[2])), label='FEKO', color='tab:purple')
            
            if(patch != '1'):
                measData = np.transpose(np.loadtxt(measFolder+'Patch S11s in holders/6GHzPatch'+patch+'a_InSetup.csv', dtype=complex, skiprows = 3, delimiter=','))
                ax1.plot(measData[0]/1e9, np.unwrap(np.angle(measData[int(patch)])), label='Meas.'+patch+'a')
                ax2.plot(measData[0]/1e9, 20*np.log10(np.abs(measData[int(patch)])), label='Meas.'+patch+'a')#, color='tab:green', marker='+', markevery=8, markersize=10)
            measData = np.transpose(np.loadtxt(measFolder+'Patch S11s in holders/6GHzPatch'+patch+'b_InSetup.csv', dtype=complex, skiprows = 3, delimiter=','))
            ax1.plot(measData[0]/1e9, np.unwrap(np.angle(measData[int(patch)])), linestyle=':', label='Meas.'+patch+'b')#, color='tab:green', marker='+', markevery=8, markersize=10)
            ax2.plot(measData[0]/1e9, 20*np.log10(np.abs(measData[int(patch)])), linestyle=':', label='Meas.'+patch+'b')
        
            plt.grid()
            ax1.set_ylabel(r'Angle(S$_{11}$) [rad.]')
            ax2.set_ylabel(r'Mag(S$_{11}$) [dB]')
            plt.xlabel(r'Frequency [GHz]')
            plt.title(r'Patch Antenna Reflection Coefficient')
            plt.legend()
            plt.tight_layout()
        plt.show()
        
    ###
    ###
    
    #folder = 'data3D/' ## if running something locally
    
    #runName = f'measurements_init_'
    #runName = f'measurements_init_actuallMeasuredFreqs_'
    runName = f'measurements_corrected_'
    angles = np.arange(0, 360, 40, dtype=float) ## measured with 20-degree spacing, simulate 40 degree so its faster
    measFreqs = np.linspace(5.4e9, 7.2e9, 201) ## the measured frequencies
    freqs = [measFreqs[i] for i in np.arange(len(measFreqs)) if i%10==0] ## simulate these 21 frequencies
    
    #===========================================================================
    # measurementScript(h=1/3.5, degree=3, runName=runName, angles=angles,
    #                 mesh_settings={'viewGMSH': False, 'N_antennas': 4, 'f0': 6e9, 'antenna_type': '6GHz measurement', 'antenna_radius': 0.18, 'object_geom': '6GHz measurement', 'domain_height': 1, 'domain_radius': 4.2},
    #                 prob_settings={'freqs': freqs, 'material_epsrs' : [2.73 - .014j]}) # epsr of POM taken from Complex Permittivity Measurements of Common Plastics Over Variable Temperatures, Bill Riddle
    #===========================================================================
    
    testrunName = f'{runName}dut_POMfill_' ## the test case where there is a hole partially filled with a POM cylinder
    measurementScript(h=1/3.5, degree=3, runName=testrunName, angles=angles, dutForSimSolution=True,
                    mesh_settings={'viewGMSH': False, 'N_antennas': 4, 'f0': 6e9, 'antenna_type': '6GHz measurement', 'antenna_radius': 0.18, 'object_geom': '6GHz measurement', 'defect_geom': '6GHz measurement POM cyl', 'domain_height': 1, 'domain_radius': 4.2},
                    prob_settings={'freqs': freqs, 'material_epsrs' : [2.73 - .014j], 'defect_epsrs' : [1.0 - .0j]}) # epsr of POM taken from Complex Permittivity Measurements of Common Plastics Over Variable Temperatures, Bill Riddle
    
    ## try the postprocessing with just sim. stuff:
    #postProcessing.solveFromQs(folder+runName+f'_angle{angles[0]}', solutionName=f'_Sdutfrom{testrunName}', onlyAPriori=True, SparamName=f'{folder}{testrunName}_angle{angles[0]}', returnResults=[3])
    
    #######
    ### Measurement Stuff
    #######
    #measFolder = '/mnt/c/Users/al8032pa/Work Folders/Documents/antenna measurements/Microwave Imaging/Datasets/Attempt 1 (16-4-2026)/'
    measFolder = '/mnt/c/Users/al8032pa/Work Folders/Documents/antenna measurements/Microwave Imaging/Datasets/Attempt 2 (4-5-2026)/'
    
    Sref = f'{measFolder}solidPOMblock'
    #Stest = f'{measFolder}solidPOMblock+hole_near_A1_filledwithPLA'
    Stest = f'{measFolder}solidPOMblock+hole_near_A2_filledwithPOM'
    #Stest = f'{measFolder}solidPOMblock+hole_near_A1'
    
    Ssangle = angles[0]
    measfnames = [f'{measFolder}solidPOMblock/', f'{measFolder}emptySetup/', f'{measFolder}emptySetup+foam/']
    #measfnames = [f'{measFolder}solidPOMblock/angle{Ssangle:.2f}.csv', f'{measFolder}solidPOMblock+hole_near_A1/angle{Ssangle:.2f}.csv', f'{measFolder}solidPOMblock+hole_near_A1_filledwithPLA/angle{Ssangle:.2f}.csv']
    simfname = f'{folder}{runName}_angle{Ssangle:.1f}output.npz'
    #postProcessing.measCompareSs(simfname, measfnames)
    
    angles = [0.0, 40.0, 80.0, 120.0]#np.arange(0, 360, 80, dtype=float) ## try using only a few for analysis
    frequenciesToUse=[]#[i for i in np.arange(20) if i%2==0]
    #postProcessing.solveFromQs(folder+runName+f'_angle{angles[0]}', SparamMeas=[Sref, Stest, angles, freqs], extraProbs = [folder+runName+f'_angle{angle}' for angle in angles[1:]], solutionName='', onlyAPriori=True, frequenciesToUse=frequenciesToUse, returnResults=[3])
    #postProcessing.solveFromQs(folder+runName+f'_angle{angles[0]}_regMesh', SparamMeas=[Sref, Stest, angles, freqs], extraProbs = [folder+runName+f'_angle{angle}' for angle in angles[1:]], solutionName='', onlyAPriori=False, frequenciesToUse=frequenciesToUse, returnResults=[3, 4])
    
    
    #testPatchPattern(h=1/8, name=f'6GHzpatchPatternTest_ho{8:.1f}', degree=3, freqs = np.linspace(5e9, 7e9, 50), showPlots=False)
    #testPatchPattern(h=1/3.5, name=f'6GHzpatchPatternTest_ho{3.5:.1f}', degree=3, freqs = np.linspace(5e9, 7e9, 50), showPlots=False)
    
    #patchSsPlot([3.5, 8]) ## plot S11 comp. with Feko
    
    
    if(comm.rank == model_rank):
        print(f'runScatt3D complete in {timer()-t1:.2f} s ({(timer()-t1)/3600:.2f} hours), exiting...')
        sys.stdout.flush()