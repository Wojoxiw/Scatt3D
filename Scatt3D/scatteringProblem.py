# encoding: utf-8
## this file computes the simulation

from mpi4py import MPI
import numpy as np
import dolfinx
import ufl
import basix
import functools
from timeit import default_timer as timer
from memory_profiler import memory_usage
import gmsh
import sys
from scipy.constants import c as c0, mu_0 as mu0, epsilon_0 as eps0, pi
from petsc4py import PETSc
import memTimeEstimation
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.collections import _MeshData
import miepython
import miepython.field
import resource
eta0 = np.sqrt(mu0/eps0)

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

class Scatt3DProblem():
    """Class to hold definitions and functions for simulating scattering or transmission of electromagnetic waves for a rotationally symmetric structure."""
    def __init__(self,
                 comm, # MPI communicator
                 refMeshdata, # Mesh and metadata for the reference case
                 DUTMeshdata = None, # Mesh and metadata for the DUT case - should just include defects into the object (this will change the mesh)
                 verbosity = 0,   ## if > 0, I print more stuff
                 f0=10e9,             # Frequency of the problem
                 epsr_bkg=1,          # Permittivity of the background medium
                 mur_bkg=1,           # Permeability of the background medium
                 material_epsr=3.0*(1 - 0.01j),  # Permittivity of object
                 material_mur=1+0j,   # Permeability of object
                 defect_epsr=6.5*(1 - 0.01j),      # Permittivity of defect
                 defect_mur=1+0j,       # Permeability of defect
                 fem_degree=1,            # Degree of finite elements
                 model_rank=0,        # Rank of the master model - for saving, plotting, etc.
                 MPInum = 1,          # Number of MPI processes
                 freqs = [],          # Optionally, pass in the exact frequencies to calculate  
                 Nf = 10,              # Number of frequency-points to calculate
                 BW = 4e9,              # Bandwidth (equally spaced around f0)
                 dataFolder = 'data3D/', # Folder where data is stored
                 name = 'testRun', # Name of this particular simulation
                 pol = 'vert', ## Polarization of the antenna excitation - can be either 'vert' or 'horiz' (untested)
                 computeImmediately = True, ## compute solutions at the end of initialization
                 computeRef = True, # If computing immediately, computes the reference simulation, where defects are not included
                 ErefEdut = False, # compute optimization vectors with Eref*Edut, a less-approximated version of the equation. Should provide better results, but can only be used in simulation
                 excitation = 'antennas', # if 'planewave', sends in a planewave from the +x-axis, otherwise antenna excitation as normal
                 PW_dir = np.array([0, 0, 1]), ## incident direction of the plane-wave, if used above. Default is coming in from the z-axis, to align with miepython
                 PW_pol = np.array([1, 0, 0]), ## incident polarization of the plane-wave, if used above. Default is along the x-axis
                 makeOptVects = True, ## if True, compute and saves the optimization vectors. Turn False if not needed
                 computeBoth = False, ## if True and computeImmediately is True, computes both ref and dut cases.
                 PML_R0 = 1e-11, ## 'intended damping for reflections from the PML', or something similar...
                 quaddeg = 5, ## quadrature degree for dx, default to 5 to avoid slowdown with pml if it defaults to higher?
                 solver_settings = {}, ## dictionary of additional solver settings
                 max_solver_time = -1, ## If an iteration finishes after this time, the solver aborts - only used for tests, currently. Disabled if negative
                 ):
        """Initialize the problem."""
        
        self.dataFolder = dataFolder
        self.name = name
        self.MPInum = MPInum                      # Number of MPI processes (used for estimating computational costs)
        self.comm = comm
        self.model_rank = model_rank
        self.verbosity = verbosity
        
        self.makeOptVects = makeOptVects
        self.ErefEdut = ErefEdut
        
        self.tdim = 3                             # Dimension of triangles/tetraedra. 3 for 3D
        self.fdim = self.tdim - 1                      # Dimension of facets
        
        self.lambda0 = c0/f0                      # Vacuum wavelength, used to define lengths in the mesh
        self.k0 = 2*np.pi*f0/c0                   # Vacuum wavenumber
        
        if(len(freqs) > 0): ## if given frequency points, use those
            self.Nf = len(freqs)
            self.fvec = freqs  # Vector of simulation frequencies
        else:
            self.Nf = Nf
            self.fvec = np.linspace(f0-BW/2, f0+BW/2, Nf)  # Vector of simulation frequencies
            
        self.PML_R0 = PML_R0
        self.dxquaddeg = quaddeg
        self.solver_settings = solver_settings
        self.max_solver_time = max_solver_time
            
        self.epsr_bkg = epsr_bkg
        self.mur_bkg = mur_bkg
        self.material_epsr = material_epsr
        self.material_mur = material_mur
        self.defect_epsr = defect_epsr
        self.defect_mur = defect_mur
        self.fem_degree = fem_degree
        self.antenna_pol = pol
        self.excitation = excitation
        
        self.PW_dir = PW_dir
        self.PW_pol = PW_pol

        # Set up mesh information
        self.refMeshdata = refMeshdata
        self.refMeshdata.mesh.topology.create_connectivity(self.tdim, self.tdim)
        self.refMeshdata.mesh.topology.create_connectivity(self.fdim, self.tdim) ## required when there are no antennas, for some reason
        if(DUTMeshdata != None):
            self.DUTMeshdata = DUTMeshdata
            self.DUTMeshdata.mesh.topology.create_connectivity(self.tdim, self.tdim)
            self.DUTMeshdata.mesh.topology.create_connectivity(self.fdim, self.tdim) ## required when there are no antennas, for some reason
            
        # Calculate solutions
        if(computeImmediately):
            if(computeBoth): ## compute both cases, then opt vectors if asked for
                self.compute(False, makeOptVects=False)
                self.makeOptVectors(True) ## makes an xdmf of the DUT mesh/epsrs
                self.saveDofsView(self.DUTMeshdata, self.dataFolder+self.name+'DUTDofsview.xdmf') ## to see that all groups are assigned appropriately
                self.saveDofsView(self.refMeshdata, self.dataFolder+self.name+'refDofsview.xdmf')
                self.compute(True, makeOptVects=self.makeOptVects)
            else: ## just compute the ref case, and make opt vects if asked for
                self.compute(computeRef, makeOptVects=self.makeOptVects)
            
                
    
    #@profile
    def compute(self, computeRef=True, makeOptVects=True):
        '''
        Sets up and runs the simulation. All the setup is set to reflect the current mesh, reference or dut. Solutions are saved.
        :param computeRef: If True, computes on the reference mesh
        '''
        if(computeRef):
            meshData = self.refMeshdata
        else:
            meshData = self.DUTMeshdata
        # Initialize function spaces, boundary conditions, and PML - for the reference mesh
        self.InitializeFEM(meshData)
        self.InitializeMaterial(meshData)
        self.CalculatePML(meshData, self.k0) ## this is recalculated for each frequency, in ComputeSolutions - run it here just to initialize variables (not sure if needed)
        t1 = timer()
        #mem_usage = memory_usage((self.ComputeSolutions, (meshData,), {'computeRef':computeRef,}), max_usage = True)/1000 ## track the memory usage here
        self.ComputeSolutions(meshData, computeRef=True)
        mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2 ## should give max. RSS for the process in GB - possibly this is slightly less than the memory required
        self.calcTime = timer()-t1 ## Time it took to solve the problem. Given to mem-time estimator 
        if(self.verbosity > 2):
            print(f'Max. memory: {mem_usage:.3f} GiB -- '+f"{self.comm.rank=} {self.comm.size=}")
        mems = self.comm.gather(mem_usage, root=self.model_rank)
        if (self.comm.rank == self.model_rank):
            self.memCost = sum(mems) ## keep the total usage. Only the master rank should be used, so this should be fine
            if(self.verbosity>0):
                print(f'Total memory: {self.memCost:.3f} GiB ({mem_usage*self.MPInum:.3f} GiB for this process, MPInum={self.MPInum} times)')
                print(f'Computations for {self.name} completed in ' + '\033[31m' + f' {self.calcTime:.2e} s ({self.calcTime/3600:.2e} hours, or {self.calcTime/self.Nf:.2e} s/freq) ' + '\033[0m')
        sys.stdout.flush()
        if(makeOptVects):
            self.makeOptVectors()
                
    def InitializeFEM(self, meshData):
        # Set up some FEM function spaces and boundary condition stuff.
        curl_element = basix.ufl.element('N1curl', meshData.mesh.basix_cell(), self.fem_degree)
        self.Vspace = dolfinx.fem.functionspace(meshData.mesh, curl_element)
        self.ndofs = self.Vspace.dofmap.index_map.size_global * self.Vspace.dofmap.index_map_bs ## from https://github.com/jpdean/maxwell/blob/master/solver.py#L108-L162 - presumably this is accurate? Only the first coefficient is nonone
        self.ScalarSpace = dolfinx.fem.functionspace(meshData.mesh, ('CG', 1)) ## this is just used for plotting+post-computations, so use degree... 1?
        self.Wspace = dolfinx.fem.functionspace(meshData.mesh, ("DG", 0))
        # Create measures for subdomains and surfaces
        self.dx = ufl.Measure('dx', domain=meshData.mesh, subdomain_data=meshData.subdomains, metadata={'quadrature_degree': self.dxquaddeg})
        self.dx_dom = self.dx((meshData.domain_marker, meshData.mat_marker, meshData.defect_marker))
        self.dx_pml = self.dx(meshData.pml_marker)
        self.ds = ufl.Measure('ds', domain=meshData.mesh, subdomain_data=meshData.boundaries) ## changing quadrature degree on ds/dS doesn't seem to have any effect
        self.dS = ufl.Measure('dS', domain=meshData.mesh, subdomain_data=meshData.boundaries) ## capital S for internal facets (shared between two cells?)
        self.ds_antennas = [self.ds(m) for m in meshData.antenna_surface_markers]
        self.ds_pec = self.ds(meshData.pec_surface_marker)
        self.Ezero = dolfinx.fem.Function(self.Vspace)
        self.Ezero.x.array[:] = 0.0
        self.pec_dofs = dolfinx.fem.locate_dofs_topological(self.Vspace, entity_dim=self.fdim, entities=meshData.boundaries.find(meshData.pec_surface_marker))
        self.bc_pec = dolfinx.fem.dirichletbc(self.Ezero, self.pec_dofs)
        if(meshData.FF_surface): ## if there is a farfield surface
            self.dS_farfield = self.dS(meshData.farfield_surface_marker)
            cells = []
            ff_facets = meshData.boundaries.find(meshData.farfield_surface_marker)
            facets_to_cells = meshData.mesh.topology.connectivity(self.fdim, self.tdim)
            for facet in ff_facets:
                for cell in facets_to_cells.links(facet):
                    if cell not in cells:
                        cells.append(cell)
            cells.sort()
            self.farfield_cells = np.array(cells)
        else:
            self.farfield_cells = []
        
    def InitializeMaterial(self, meshData):
        # Set up material parameters. Not chancing mur for now, need to edit this if doing so
        self.epsr = dolfinx.fem.Function(self.Wspace)
        self.mur = dolfinx.fem.Function(self.Wspace)
        self.epsr.x.array[:] = self.epsr_bkg
        self.mur.x.array[:] = self.mur_bkg
        mat_cells = meshData.subdomains.find(meshData.mat_marker)
        self.mat_dofs = dolfinx.fem.locate_dofs_topological(self.Wspace, entity_dim=self.tdim, entities=mat_cells)
        defect_cells = meshData.subdomains.find(meshData.defect_marker)
        self.defect_dofs = dolfinx.fem.locate_dofs_topological(self.Wspace, entity_dim=self.tdim, entities=defect_cells)
        pml_cells = meshData.subdomains.find(meshData.pml_marker)
        self.pml_dofs = dolfinx.fem.locate_dofs_topological(self.Wspace, entity_dim=self.tdim, entities=pml_cells)
        domain_cells = meshData.subdomains.find(meshData.domain_marker)
        self.domain_dofs = dolfinx.fem.locate_dofs_topological(self.Wspace, entity_dim=self.tdim, entities=domain_cells)
        self.epsr.x.array[self.mat_dofs] = self.material_epsr
        self.mur.x.array[self.mat_dofs] = self.material_mur
        self.epsr.x.array[self.defect_dofs] = self.material_epsr
        self.mur.x.array[self.defect_dofs] = self.material_mur
        self.epsr_array_ref = self.epsr.x.array.copy()
        self.epsr.x.array[self.defect_dofs] = self.defect_epsr
        self.mur.x.array[self.defect_dofs] = self.defect_mur
        self.epsr_array_dut = self.epsr.x.array.copy()
        
    def CalculatePML(self, meshData, k):
        '''
        Set up the PML - stretched coordinates to form a perfectly-matched layer which absorbs incoming (perpendicular) waves
         Since we calculate at many frequencies, recalculate this for each freq. (should also just work for many frequencies without recalculating, but the calculation is quick)
        :param k: Frequency used for coordinate stretching.
        '''
        # Set up the PML
        def pml_stretch(y, x, k, x_dom=0, x_pml=1, n=3, R0=self.PML_R0):
            '''
            Calculates the PML stretching of a coordinate
            :param y: the coordinate to be stretched
            :param x: the coordinate the stretching is based on
            :param k: wavenumber
            :param x_dom: size of domain
            :param x_pml: size of pml
            :param n: order
            :param R0: intended damping (based on relative strength of reflection?) According to 'THE_ELECTRICAL_ENGINEERING_HANDBOOKS', section 9.7: Through extensive numerical experimentation, Gedney
            (1996) and He (1997) found that, for a broad range of applications, an optimal choice for a 10-cell-thick, polynomial-graded PML is R(0) = e^-16. For a 5-cell-thick PML, R(0) = e^-8 is optimal.
            '''
            return y*(1 - 1j*(n + 1)*np.log(1/R0)/(2*k*np.abs(x_pml - x_dom))*((x - x_dom)/(x_pml - x_dom))**n)

        def pml_epsr_murinv(pml_coords):
            '''
            Transforms epsr, mur, using the given stretched coordinates (this implements the pml)
            :param pml_coords: the coordinates
            '''
            J = ufl.grad(pml_coords)
            A = ufl.inv(J)
            epsr_pml = ufl.det(J) * A * self.epsr * ufl.transpose(A)
            mur_pml = ufl.det(J) * A * self.mur * ufl.transpose(A)
            murinv_pml = ufl.inv(mur_pml)
            return epsr_pml, murinv_pml
        
        x, y, z = ufl.SpatialCoordinate(meshData.mesh)
        if(meshData.domain_geom == 'domedCyl'): ## implement it for this geometry
            r = ufl.real(ufl.sqrt(x**2 + y**2)) ## cylindrical radius. need to set this to real because I compare against it later
            domain_height_spheroid = ufl.conditional(ufl.ge(r, meshData.domain_radius), meshData.domain_height/2, (meshData.domain_height/2+meshData.dome_height)*ufl.real(ufl.sqrt(1-(r/(meshData.domain_radius+meshData.domain_spheroid_extraRadius))**2)))   ##start the z-stretching at, at minmum, the height of the domain cylinder
            PML_height_spheroid = (meshData.PML_height/2+meshData.dome_height)*ufl.sqrt(1-(r/(meshData.PML_radius+meshData.PML_spheroid_extraRadius))**2) ##should always be ge the pml cylinder's height
            x_stretched = pml_stretch(x, r, k, x_dom=meshData.domain_radius, x_pml=meshData.PML_radius)
            y_stretched = pml_stretch(y, r, k, x_dom=meshData.domain_radius, x_pml=meshData.PML_radius)
            z_stretched = pml_stretch(z, abs(z), k, x_dom=domain_height_spheroid, x_pml=PML_height_spheroid) ## /2 since the height is from - to +
            x_pml = ufl.conditional(ufl.ge(abs(r), meshData.domain_radius), x_stretched, x) ## stretch when outside radius of the domain
            y_pml = ufl.conditional(ufl.ge(abs(r), meshData.domain_radius), y_stretched, y) ## stretch when outside radius of the domain
            z_pml = ufl.conditional(ufl.ge(abs(z), domain_height_spheroid), z_stretched, z) ## stretch when outside the height of the cylinder of the domain (or oblate spheroid roof with factor a_dom/a_pml - should only be higher/lower inside the domain radially)
        elif(meshData.domain_geom == 'sphere'):
            r = ufl.real(ufl.sqrt(x**2 + y**2 + z**2))
            x_stretched = pml_stretch(x, r, k, x_dom=meshData.domain_radius, x_pml=meshData.PML_radius)
            y_stretched = pml_stretch(y, r, k, x_dom=meshData.domain_radius, x_pml=meshData.PML_radius)
            z_stretched = pml_stretch(z, r, k, x_dom=meshData.domain_radius, x_pml=meshData.PML_radius)
            x_pml = ufl.conditional(ufl.ge(abs(r), meshData.domain_radius), x_stretched, x) ## stretch when outside radius of the domain
            y_pml = ufl.conditional(ufl.ge(abs(r), meshData.domain_radius), y_stretched, y) ## stretch when outside radius of the domain
            z_pml = ufl.conditional(ufl.ge(abs(r), meshData.domain_radius), z_stretched, z) ## stretch when outside radius of the domain
        else:
            print('nonvalid meshData.domain_geom')
            exit()
        pml_coords = ufl.as_vector((x_pml, y_pml, z_pml))
        self.epsr_pml, self.murinv_pml = pml_epsr_murinv(pml_coords)
    
    #@profile
    def ComputeSolutions(self, meshData, computeRef = True):
        '''
        Computes the solutions
        
        :param computeRef: if True, computes the reference case (this is always needed for reconstruction). if False, computes the DUT case (needed for simulation-only stuff).
        Since things need to be initialized for each mesh, that should be done first.
        '''
        def Eport(x): # Set up the excitation - on antenna faces
            """
            Compute the normalized electric field distribution in all ports.
            :param x: Some vector of positions you want to find the field on
            """
            Ep = np.zeros((3, x.shape[1]), dtype=complex)
            for p in range(meshData.N_antennas):
                center = meshData.pos_antennas[p]
                phi = -meshData.rot_antennas[p] # Note rotation by the negative of antenna rotation
                Rmat = np.array([[np.cos(phi), -np.sin(phi), 0],
                                 [np.sin(phi), np.cos(phi), 0],
                                 [0, 0, 1]]) ## rotation around z
                y = np.transpose(x.T - center)
                loc_x = np.dot(Rmat, y) ### position vector, [x, y, z] presumably, rotated to be in the coordinates the antenna was defined in
                if (self.antenna_pol == 'vert'): ## vertical (z-) pol, field varies along x
                    Ep_loc = np.vstack((0*loc_x[0], 0*loc_x[0], np.cos(meshData.kc*loc_x[0])))/np.sqrt(meshData.antenna_width/2)
                else: ## horizontal (x-) pol, field varies along z
                    Ep_loc = np.vstack((np.cos(meshData.kc*loc_x[2])), 0*loc_x[2], 0*loc_x[2])/np.sqrt(meshData.antenna_height/2)
                    
                #simple, inexact confinement conditions
                #Ep_loc[:,np.sqrt(loc_x[0]**2 + loc_x[1]**2) > antenna_width] = 0 ## no field outside of the antenna's width (circular)
                ##if I confine it to just the 'empty face' of the waveguide thing. After testing, this seems to make no difference to just selecting the entire antenna via a sphere, with the above line
                Ep_loc[:, np.abs(loc_x[0])  > meshData.antenna_width*.54] = 0 ## no field outside of the antenna's width
                Ep_loc[:, np.abs(loc_x[1])  > meshData.antenna_depth*.04] = 0 ## no field outside of the antenna's depth - origin should be on this face - it is a face so no depth
                #for both
                Ep_loc[:,np.abs(loc_x[2]) > meshData.antenna_height*.54] = 0 ## no field outside of the antenna's height.. plus a small extra (no idea if that matters)
                
                Ep_global = np.dot(Rmat, Ep_loc)
                Ep = Ep + Ep_global
            return Ep
        def planeWave(x, k):
            '''
            Set up the excitation for a background plane-wave. Uses the problem's PW parameters. Needs the frequency, so I do it inside the freq. loop
            :param x: some given position you want to find the field on
            :param k: wavenumber
            '''
            E_pw = np.zeros((3, x.shape[1]), dtype=complex)
            if(self.excitation == 'planewave'): # only make an excitation if we actually want one
                E_pw[0, :] = self.PW_pol[0] ## just use the same amplitude as the polarization has
                E_pw[1, :] = self.PW_pol[1] ## just use the same amplitude as the polarization has
                E_pw[2, :] = self.PW_pol[2] ## just use the same amplitude as the polarization has
                k_pw = k*self.PW_dir ## direction (should be given normalized)
                E_pw[:] = E_pw[:]*np.exp(-1j*np.dot(k_pw, x))
            return E_pw
    
        Ep = dolfinx.fem.Function(self.Vspace)
        Ep.interpolate(lambda x: Eport(x))
        Eb = dolfinx.fem.Function(self.Vspace) ## background/plane wave excitation
        
        # Set up simulation
        E = ufl.TrialFunction(self.Vspace)
        v = ufl.TestFunction(self.Vspace)
        curl_E = ufl.curl(E)
        curl_v = ufl.curl(v)
        nvec = ufl.FacetNormal(meshData.mesh)
        Zrel = dolfinx.fem.Constant(meshData.mesh, 1j)
        k00 = dolfinx.fem.Constant(meshData.mesh, 1j)
        a = [dolfinx.fem.Constant(meshData.mesh, 1.0 + 0j) for n in range(meshData.N_antennas)]
        F_antennas_str = '0' ## seems to give an error when evaluating an empty string
        for n in range(meshData.N_antennas):
            F_antennas_str += f"""+ 1j*k00/Zrel*ufl.inner(ufl.cross(E, nvec), ufl.cross(v, nvec))*self.ds_antennas[{n}] - 1j*k00/Zrel*2*a[{n}]*ufl.sqrt(Zrel*eta0)*ufl.inner(ufl.cross(Ep, nvec), ufl.cross(v, nvec))*self.ds_antennas[{n}]"""
        F = ufl.inner(1/self.mur*curl_E, curl_v)*self.dx_dom \
            - ufl.inner(k00**2*self.epsr*E, v)*self.dx_dom \
            + ufl.inner(self.murinv_pml*curl_E, curl_v)*self.dx_pml \
            - ufl.inner(k00**2*self.epsr_pml*E, v)*self.dx_pml \
            - ufl.inner(k00**2*(self.epsr - 1/self.mur*self.mur_bkg*self.epsr_bkg)*Eb, v)*self.dx_dom + eval(F_antennas_str) ## background field and antenna terms
        bcs = [self.bc_pec]
        lhs, rhs = ufl.lhs(F), ufl.rhs(F)
        max_its = 10000
        conv_sets = {"ksp_rtol": 1e-6, "ksp_atol": 1e-15, "ksp_max_it": max_its} ## convergence settings
        petsc_options = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"} ## the basic option - fast, robust/accurate, but takes a lot of memory
        
        #petsc_options={"ksp_type": "lgmres", "pc_type": "sor", **self.solver_settings, **conv_sets} ## (https://petsc.org/release/manual/ksp/)
        #petsc_options={"ksp_type": "lgmres", 'pc_type': 'asm', 'sub_pc_type': 'sor', **conv_sets} ## is okay
        
        #petsc_options={"ksp_type": "lgmres", "pc_type": "ksp", "pc_ksp_type":"gmres", 'ksp_max_it': 1, 'pc_ksp_rtol' : 1e-1, "pc_ksp_pc_type": "sor", **conv_sets}
        #petsc_options={'ksp_type': 'fgmres','ksp_gmres_restart': 1000, 'pc_type': 'ksp', "ksp_ksp_type": 'bcgs', "ksp_ksp_max_it": 100, 'ksp_pc_type': 'jacobi', **conv_sets, **self.solver_settings} ## one of the best so far... tfqmr or bcgs
        #petsc_options={'ksp_type': 'gmres', 'ksp_gmres_restart': 1000, 'pc_type': 'gamg', 'pc_gamg_type': 'agg', 'pc_gamg_sym_graph': 1, 'matptap_via': 'scalable', 'pc_gamg_square_graph': 1, 'pc_gamg_reuse_interpolation': 1, **conv_sets, **self.solver_settings}
        #petsc_options={'ksp_type': 'fgmres', 'ksp_gmres_restart': 1000, 'pc_type': 'gamg', 'mg_levels_pc_type': 'jacobi', 'pc_gamg_agg_nsmooths': 1, 'pc_mg_cycle_type': 'v', 'pc_gamg_aggressive_coarsening': 2, 'pc_gamg_theshold': 0.01, 'mg_levels_ksp_max_it': 5, 'mg_levels_ksp_type': 'chebyshev', 'pc_gamg_repartition': False, 'pc_gamg_square_graph': True, 'pc_mg_type': 'additive', **conv_sets, **self.solver_settings}
        
        ## GASM attempts
        #petsc_options={'ksp_type': 'fgmres', 'ksp_gmres_restart': 1200, 'pc_type': 'gasm', **conv_sets, **self.solver_settings}
        #petsc_options={'ksp_type': 'fgmres', 'ksp_gmres_restart': 1000, 'pc_type': 'gasm', 'sub_ksp_type': 'preonly', 'pc_gasm_total_subdomains': self.MPInum*2, 'pc_gasm_overlap': 4, 'sub_pc_type': 'ilu', 'sub_pc_factor_levels': 1, 'sub_pc_factor_mat_solver_type': 'petsc', 'sub_pc_factor_mat_ordering_type': 'nd', **conv_sets, **self.solver_settings}
        
        
        #petsc_options = {"ksp_type": "fgmres", 'ksp_gmres_restart': 1000, "pc_type": "composite", **conv_sets, **self.solver_settings}
        
        #petsc_options = {"ksp_type": "fgmres", 'ksp_gmres_restart': 1000, "pc_type": "composite", 'pc_composite_type': 'additive', 'pc_composite_pcs': 'sor,gasm', **conv_sets, **self.solver_settings}
        #self.max_solver_time = 30
        
        ## BDDC
        #petsc_options={'ksp_type': 'fgmres', 'ksp_gmres_restart': 1200, 'pc_type': 'bddc', **conv_sets, **self.solver_settings}
        
        ## HPDDM stuff
        #petsc_options={'ksp_type': 'hpddm', 'ksp_hpddm_type': 'gmres', **conv_sets, **self.solver_settings}
        #petsc_options={'ksp_type': 'fgmres', 'ksp_gmres_restart': 1000, 'pc_type': 'hpddm', 'pc_hpddm_type': 'hcurl', 'sub_pc_type': 'lu', 'sub_ksp_type': 'preonly', 'pc_hpddm_coarse_correction': 'galerkin', 'pc_hpddm_levels_1_overlap': 2, **conv_sets, **self.solver_settings}
        #petsc_options={'ksp_type': 'fgmres', 'ksp_gmres_restart': 1000, 'pc_type': 'hpddm', 'pc_hpddm_type': 'hcurl', 'sub_pc_type': 'lu', 'sub_ksp_type': 'preonly', 'pc_hpddm_coarse_correction': 'deflated', 'pc_hpddm_levels_1_overlap': 2, 'coarse_pc_type': 'gamg', 'pc_hpddm_levels_1_eps_nev': 10, **conv_sets, **self.solver_settings}


        
        cache_dir = f"{str(Path.cwd())}/.cache"
        jit_options={}
        jit_options= {"cffi_extra_compile_args": ['-O3', "-march=native"], "cache_dir": cache_dir, "cffi_libraries": ["m"]} ## possibly this speeds things up a little.
        
        problem = dolfinx.fem.petsc.LinearProblem(lhs, rhs, bcs=bcs, petsc_options=petsc_options, jit_options=jit_options)
        
        #=======================================================================
        # a = dolfinx.fem.form(problem.a)
        # A = dolfinx.fem.petsc.assemble_matrix(a, bcs=bcs)
        # A.assemble()
        # A_matis = PETSc.Mat().create(comm=A.getComm())
        # A_matis.setSizes(A.getSizes())
        # A_matis.setType('is')
        # A_matis.setUp()
        # A_matis.assemble()
        # b = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(problem.L))
        # dolfinx.fem.petsc.apply_lifting(b, [a], [bcs])
        # b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        # dolfinx.fem.petsc.set_bc(b, bcs)
        # ksp = PETSc.KSP().create(A_matis.getComm())
        # ksp.setOperators(A_matis)
        # ksp.setType("cg")
        # pc = ksp.getPC()
        # pc.setType("bddc")
        # x = A_matis.createVecLeft()
        # x.set(0)
        # # ksp.solve(b, x)
        # 
        #=======================================================================
        
        ksp = problem.solver
        pc = ksp.getPC()
        #print(ksp.view()) ## gives the settings
        class TimeAbortMonitor:
            def __init__(self, max_time, comm):
                self.maxT = max_time
                self.start_time = timer()
                self.comm = comm
            
            def __call__(self, ksp, its, rnorm):
                if(self.comm.rank == 0):
                    if(its%101 == 100): ## print some progress, in case a run is taking extremely long
                        print(f'Iteration {its}, norm {rnorm:.3e}...')
                if (self.maxT>0) and (timer() - self.start_time > self.maxT): ## if using a max time, call out when it is reached
                    if(self.comm.rank == 0):
                        PETSc.Sys.Print(f"Aborting solve after {its} iterations due to maximum solver time ({self.maxT} s)")
                    ksp.setConvergedReason(PETSc.KSP.ConvergedReason.DIVERGED_NULL)
                                  
        ksp.setMonitor(TimeAbortMonitor(self.max_solver_time, self.comm))
        
        
        
        ### try Laguerre transform stuff (https://arxiv.org/pdf/2309.11023)
        #Assemble K, M
        
        #Laguerre Setup Stuff
        eta = 1
        M_lag = 8
        beta1 = mu0/(4*eps0) * self.epsr*eta0**2
        
        # for m in N, build then solve elliptic systems/operators
        
        # Then sum them to obtain the actual solution
        
        ###
        
        
        #=======================================================================
        # ### try nullspace stuff
        # nullvec = dolfinx.fem.Function(self.Vspace)
        # with nullvec.vector.localForm() as loc:
        #     loc.set(1.0)  # Uniform constant vector field (approximate near-nullspace)
        # for bc in bcs:
        #     dofs = bc.dof_indices # The PETSc dof index for this BC.
        #     with nullvec.vector.localForm() as loc:
        #         loc.array[dofs] = 0.0
        # norm = nullvec.vector.norm()
        # nullvec.vector.scale(1.0/norm)
        #     
        # E_nulls = [] # Or try directional components
        # for i in range(self.Vspace.mesh.geometry.dim):
        #     v = dolfinx.fem.Function(self.Vspace)
        #     dofmap = self.Vspace.dofmap
        #     for cell in range(self.Vspace.mesh.topology.index_map(self.Vspace.mesh.topology.dim).size_local):
        #         dofs = dofmap.cell_dofs(cell)
        #         for dof in dofs:
        #             v.vector[dof] = 1.0  # crude approximation
        #     E_nulls.append(v)
        #     
        # vecs = [v.vector for v in E_nulls] + nullvec
        # nullspace = dolfinx.la.create_petsc_nullspace(vecs, comm=self.comm, near=True)
        # 
        # A = problem.A  # from LinearProblem
        # A.setNearNullSpace(nullspace)
        #=======================================================================
        
        
        #=======================================================================
        # def monitor(ksp, its, rnorm): ## inner KSP iterations monitor
        #     if(its%1001 == 0):
        #         print(f"[Inner KSP] Iteration {its}, residual = {rnorm}")
        # ksp_inner = pc.getKSP()
        # ksp_inner.setMonitor(monitor)
        # #ksp_inner.setTolerances(rtol=1e-2, atol=1e-10, max_it=102)
        #=======================================================================
        
        #=======================================================================
        # ## save bilinear/linear parts to try with real solvers - takes too much memory to save/load like this, also strange mem maxing when using discrete gradient
        # a = dolfinx.fem.form(problem.a)
        # A = dolfinx.fem.petsc.assemble_matrix(a, bcs=bcs)
        # A.assemble()
        # b = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(problem.L))
        # dolfinx.fem.petsc.apply_lifting(b, [a], [bcs])
        # b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        # dolfinx.fem.petsc.set_bc(b, bcs)
        #  
        # indptr, indices, data = A.getValuesCSR() ## hopefully the correct way to get the data
        # np.savez("realTest/A.npz", indptr=indptr, indices=indices, data=data, shape=A.getSize()) ## save with numpy since otherwise it would be PETSc complex, not sure if there's a better way
        # np.savez("realTest/b.npz", data=b.getArray(), shape=b.getSize())
        # # need coords and discrete gradient also for hypre ams
        # coords = meshData.mesh.geometry.x
        # np.savez("realTest/coords.npz", coords=coords)
        # G = dolfinx.fem.petsc.discrete_gradient(self.ScalarSpace, self.Vspace)
        # G.assemble()
        # indptr, indices, data = G.getValuesCSR()
        # np.savez("realTest/G.npz", indptr=indptr, indices=indices, data=data, shape=G.getSize())
        # exit()
        # ##
        #=======================================================================
        
        #=======================================================================
        # ## try plotting the array, just to see it. This... may require only 1 MPI process as is?
        # A_dense = A.convert('dense')
        # if(self.comm.rank == 0):
        #     rows, cols = A_dense.getSize()
        #     A_np = np.zeros((rows, cols))
        #     for i in range(rows):
        #         row_vals = A_dense.getRow(i)[1]  # getRow returns (cols, values)
        #         A_np[i, :len(row_vals)] = row_vals
        # plt.imshow(np.log10(np.abs(A_np)), cmap="inferno")
        # plt.colorbar()
        # plt.title('log10(Abs. of problem matrix)')
        # plt.show()
        #=======================================================================
        
            
        def ComputeFields():
            '''
            Computes the fields. There are two cases: one with antennas, and one without (PW excitation)
            Returns solutions, a list of Es for each frequency and exciting antenna, and S (0 if no antennas), a list of S-parameters for each frequency, exciting antenna, and receiving antenna
            '''
            S = np.zeros((self.Nf, meshData.N_antennas, meshData.N_antennas), dtype=complex)
            solutions = []
            for nf in range(self.Nf):
                if( (self.verbosity >= 1 and self.comm.rank == self.model_rank) or (self.verbosity > 2) ):
                    print(f'Rank {self.comm.rank}: Frequency {nf+1} / {self.Nf}')
                sys.stdout.flush()
                k0 = 2*np.pi*self.fvec[nf]/c0
                k00.value = k0
                Zrel.value = k00.value/np.sqrt(k00.value**2 - meshData.kc**2)
                self.CalculatePML(meshData, k0)  ## update PML to this freq.
                Eb.interpolate(functools.partial(planeWave, k=k0))
                sols = []
                if(meshData.N_antennas == 0): ## if no antennas:
                    E_h = problem.solve()
                    sols.append(E_h.copy())
                else:
                    for n in range(meshData.N_antennas):
                        for m in range(meshData.N_antennas):
                            a[m].value = 0.0
                        a[n].value = 1.0
                        E_h = problem.solve()
                        for m in range(meshData.N_antennas):
                            factor = dolfinx.fem.assemble.assemble_scalar(dolfinx.fem.form(2*ufl.sqrt(Zrel*eta0)*ufl.inner(ufl.cross(Ep, nvec), ufl.cross(Ep, nvec))*self.ds_antennas[m]))
                            factors = self.comm.gather(factor, root=self.model_rank)
                            if self.comm.rank == self.model_rank:
                                factor = sum(factors)
                            else:
                                factor = None
                            factor = self.comm.bcast(factor, root=self.model_rank)
                            b = dolfinx.fem.assemble.assemble_scalar(dolfinx.fem.form(ufl.inner(ufl.cross(E_h, nvec), ufl.cross(Ep, nvec))*self.ds_antennas[m] + Zrel/(1j*k0)*ufl.inner(ufl.curl(E_h), ufl.cross(Ep, nvec))*self.ds_antennas[m]))/factor
                            bs = self.comm.gather(b, root=self.model_rank)
                            if self.comm.rank == self.model_rank:
                                b = sum(bs)
                            else:
                                b = None
                            b = self.comm.bcast(b, root=self.model_rank)
                            S[nf,m,n] = b
                        sols.append(E_h.copy())
                solutions.append(sols)
            return S, solutions
        
        if(computeRef):
            if( (self.verbosity >= 1 and self.comm.rank == self.model_rank) or (self.verbosity > 2) ):
                print(f'Rank {self.comm.rank}: Computing REF solutions (ndofs={self.ndofs})')
            sys.stdout.flush()
            self.epsr.x.array[:] = self.epsr_array_ref
            self.S_ref, self.solutions_ref = ComputeFields()    
        else:
            if( (self.verbosity >= 1 and self.comm.rank == self.model_rank) or (self.verbosity > 2) ):
                print(f'Rank {self.comm.rank}: Computing DUT solutions')
            sys.stdout.flush()
            self.epsr.x.array[:] = self.epsr_array_dut
            self.S_dut, self.solutions_dut = ComputeFields()
            
        solver = problem.solver
        fname=self.dataFolder+self.name+"solver_output.info"
        viewer = PETSc.Viewer().createASCII(fname)
        solver.view(viewer)
        self.solver_its = solver.its
        self.solver_norm = solver.norm
        if( (self.verbosity > 0 and self.comm.rank == self.model_rank)): ## print solver info for the final solve - presumably this is representative of all solves
            print(f'Converged for reason: {solver.reason}, after {solver.its} iterations. Norm: {solver.norm}') ## if reason is negative, it diverged (see https://petsc.org/release/manualpages/KSP/KSPConvergedReason/)
            if(self.solver_its == max_its):
                print('\033[93m' + 'Warning: solver failed to converge.' + '\033[0m')
            if(self.verbosity > 3):
                solver_output = open(fname, "r") ## this prints to console
                for line in solver_output.readlines():
                    print(line)
            sys.stdout.flush()
           
    #@profile
    def makeOptVectors(self, DUTMesh=False):
        '''
        Computes the optimization vectors from the E-fields and saves to .xdmf - this is done on the reference mesh.
        This function also saves various other parameters needed for later postprocessing
        :param meshData: Should be the refMeshData, since that is what the reconstruction is on. If justMesh, can be the DUT mesh
        :param justMesh: No optimization vectors, just cell volumes and epsrs
        '''
        ## First, save mesh to xdmf
        if(DUTMesh):
            meshData = self.DUTMeshdata
            xdmf = dolfinx.io.XDMFFile(comm=self.comm, filename=self.dataFolder+self.name+'DUTmesh.xdmf', file_mode='w')
        else:
            meshData = self.refMeshdata
            xdmf = dolfinx.io.XDMFFile(comm=self.comm, filename=self.dataFolder+self.name+'output-qs.xdmf', file_mode='w')
        xdmf.write_mesh(meshData.mesh)
        
        ## Then, compute opt. vectors, and save data
        if( (self.verbosity > 0 and self.comm.rank == self.model_rank) or (self.verbosity > 2) ):
            print(f'Rank {self.comm.rank}: Computing optimization vectors')
            sys.stdout.flush()
        
        # Create function space for temporary interpolation
        q = dolfinx.fem.Function(self.Wspace)
        bb_tree = dolfinx.geometry.bb_tree(meshData.mesh, meshData.mesh.topology.dim)
        cell_volumes = dolfinx.fem.assemble_vector(dolfinx.fem.form(ufl.conj(ufl.TestFunction(self.Wspace))*ufl.dx)).array
        def q_func(x, Em, En, k0):
            '''
            Calculates the 'optimization vector' at each position in the reference meshData. Since the DUT mesh is different,
            this requires interpolation to find the E-fields at each point
            :param x: positions/points
            :param Em: first E-field
            :param En: second E-field
            :param k0: wavenumber at this frequency
            '''
            cells = []
            cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, x.T)
            colliding_cells = dolfinx.geometry.compute_colliding_cells(meshData.mesh, cell_candidates, x.T)
            for i, point in enumerate(x.T):
                if len(colliding_cells.links(i)) > 0:
                    cells.append(colliding_cells.links(i)[0])
            Em_vals = Em.eval(x.T, cells)
            En_vals = En.eval(x.T, cells)
            values = -1j*k0/eta0/2*(Em_vals[:,0]*En_vals[:,0] + Em_vals[:,1]*En_vals[:,1] + Em_vals[:,2]*En_vals[:,2])*cell_volumes
            return values
        
        
        ## save some problem/mesh data
        self.epsr.x.array[:] = cell_volumes
        #self.epsr.name = 'Cell Volumes' ## can use names, but makes looking in paraview more annoying
        xdmf.write_function(self.epsr, -3)
        self.epsr.x.array[:] = self.epsr_array_ref
        #self.epsr.name = 'epsr_ref'
        xdmf.write_function(self.epsr, -2)
        self.epsr.x.array[:] = self.epsr_array_dut
        #self.epsr.name = 'epsr_dut'
        xdmf.write_function(self.epsr, -1)
        
        
        if(not DUTMesh): ## Do the interpolation to find qs, then save them
            b = np.zeros(self.Nf*meshData.N_antennas*meshData.N_antennas, dtype=complex)
            for nf in range(self.Nf):
                if( (self.verbosity > 0 and self.comm.rank == self.model_rank) or (self.verbosity > 2) and (meshData.N_antennas > 0) ):
                    print(f'Rank {self.comm.rank}: Frequency {nf+1} / {self.Nf}')
                    sys.stdout.flush()
                k0 = 2*np.pi*self.fvec[nf]/c0
                for m in range(meshData.N_antennas):
                    Em_ref = self.solutions_ref[nf][m]
                    for n in range(meshData.N_antennas):
                        if(self.ErefEdut): ## only using Eref*Eref right now. Eref*Edut should provide a superior reconstruction with fully simulated data, though
                            En = self.solutions_dut[nf][n] 
                        else:
                            En = self.solutions_ref[nf][n]
                        q.interpolate(functools.partial(q_func, Em=Em_ref, En=En, k0=k0))
                        # Each function q is one row in the A-matrix, save it to file
                        #q.name = f'freq{nf}m={m}n={n}'
                        xdmf.write_function(q, nf*meshData.N_antennas*meshData.N_antennas + m*meshData.N_antennas + n)
                if(meshData.N_antennas < 1): # if no antennas, still save something
                    q.interpolate(functools.partial(q_func, Em=self.solutions_ref[nf][0], En=self.solutions_ref[nf][0], k0=k0))
                    xdmf.write_function(q, nf)
        xdmf.close()
        
        if (self.comm.rank == self.model_rank): # Save some other values for postprocessing
            if( hasattr(self, 'solutions_dut') and hasattr(self, 'solutions_ref')): ## need both computed - otherwise, do not save
                b = np.zeros(self.Nf*meshData.N_antennas*meshData.N_antennas, dtype=complex) ## the array of S-parameters
                for nf in range(self.Nf):
                    for m in range(meshData.N_antennas):
                        for n in range(meshData.N_antennas):
                            b[nf*meshData.N_antennas*meshData.N_antennas + m*meshData.N_antennas + n] = self.S_dut[nf, m, n] - self.S_ref[nf, n, m]
                np.savez(self.dataFolder+self.name+'output.npz', b=b, fvec=self.fvec, S_ref=self.S_ref, S_dut=self.S_dut, epsr_mat=self.material_epsr, epsr_defect=self.defect_epsr, N_antennas=meshData.N_antennas)    
    
    def saveEFieldsForAnim(self, ref=True, Nframes = 50, removePML = True):
        '''
        Saves the E-field magnitudes for the final solution into .xdmf, for a number of different phase factors to create an animation in paraview
        Uses the reference mesh and fields. If removePML, set the values within the PML to 0 (can also be NaN, etc.)
        
        :param ref: If True, plot for the reference case. If False, for the DUT case
        :param Nframes: Number of frames in the anim. Each frame is a different phase from 0 to 2*pi
        :param removePML: If True, sets all values in the PML to something different
        '''
        ## This is presumably an overdone method of finding these already-computed fields - I doubt this is needed
        meshData = self.refMeshdata # use the ref case
        
        E = dolfinx.fem.Function(self.ScalarSpace)
        bb_tree = dolfinx.geometry.bb_tree(meshData.mesh, meshData.mesh.topology.dim)
        def q_abs(x, Es, pol = 'z'): ## similar to the one in makeOptVectors
            cells = []
            cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, x.T)
            colliding_cells = dolfinx.geometry.compute_colliding_cells(meshData.mesh, cell_candidates, x.T)
            for i, point in enumerate(x.T):
                if len(colliding_cells.links(i)) > 0:
                    cells.append(colliding_cells.links(i)[0])
            if(pol == 'z-pol'): ## it is not simple to save the vector itself for some reason...
                E_vals = Es.eval(x.T, cells)[:, 2]
            elif(pol == 'x-pol'):
                E_vals = Es.eval(x.T, cells)[:, 0]
            elif(pol == 'y-pol'):
                E_vals = Es.eval(x.T, cells)[:, 1]
            return E_vals
        pols = ['x-pol', 'y-pol', 'z-pol']
        if(ref):
            sol = self.solutions_ref[0][0] ## fields for the first frequency/antenna combo
        else:
            sol = self.solutions_dut[0][0]
        pml_cells = meshData.subdomains.find(meshData.pml_marker)
        pml_dofs = dolfinx.fem.locate_dofs_topological(self.ScalarSpace, entity_dim=self.tdim, entities=pml_cells)
        xdmf = dolfinx.io.XDMFFile(comm=self.comm, filename=self.dataFolder+self.name+'outputPhaseAnimation.xdmf', file_mode='w')
        xdmf.write_mesh(meshData.mesh)
        for pol in pols:
            E.interpolate(functools.partial(q_abs, Es=sol, pol=pol))
            E.name = pol
            if(removePML):
                E.x.array[pml_dofs] = 0#np.nan ## use the ScalarSpace one
            for i in range(Nframes):
                E.x.array[:] = E.x.array*np.exp(1j*2*pi/Nframes)
                xdmf.write_function(E, i)
        xdmf.close()
        if(self.verbosity>0 and self.comm.rank == self.model_rank):
            print(self.name+' E-fields animation complete')
            sys.stdout.flush()
        
    def saveDofsView(self, meshData, fname):
        '''
        Saves the dofs with different numbers for viewing in ParaView. This hangs on the cluster, for unknown reasons
        :param meshData: Whichever meshData to use
        '''
        self.InitializeFEM(meshData) ## so that this can be done before calculations
        self.InitializeMaterial(meshData) ## so that this can be done before calculations
        vals = dolfinx.fem.Function(self.Wspace)
        xdmf = dolfinx.io.XDMFFile(comm=self.comm, filename=fname, file_mode='w')
        xdmf.write_mesh(meshData.mesh)
        pec_dofs = dolfinx.fem.locate_dofs_topological(self.Wspace, entity_dim=self.fdim, entities=meshData.boundaries.find(meshData.pec_surface_marker)) ## should be empty since these surfaces are not 3D, I think
        vals.x.array[:] = np.nan
        vals.x.array[self.domain_dofs] = 0
        vals.x.array[self.pml_dofs] = -1
        vals.x.array[self.defect_dofs] = 3
        vals.x.array[self.mat_dofs] = 2
        vals.x.array[pec_dofs] = 5
        vals.x.array[self.farfield_cells] = 1
        xdmf.write_function(vals, 0)
        xdmf.close()
        if(self.verbosity>0 and self.comm.rank == self.model_rank):
            print(fname+' saved')
            sys.stdout.flush()
            
    def calcFarField(self, reference, compareToMie = False, showPlots=False, returnConvergenceVals=False, angles = None):
        '''
        Calculates the farfield at each frequency point at at given angles, using the farfield boundary in the mesh - must have mesh.FF_surface = True
        Returns an array of [E_theta, E_phi] at each angle, to the master process only
        :param reference: Whether this is being computed for the DUT case or the reference
        :param compareToMie: If True, plots a comparison against predicted Mie scattering (assuming spherical object)
        :param showPlots: If True, plt.show(). Plots are still saved, though. This must be False for cluster use
        :param returnConvergenceVals: If True, returns some convergence values instead of the regular Mie scattering comparison or anything else. Angles should be default for this, so first/second are forward/backward
        :param angles: Is an array of theta and phi angles to calculate at [in degrees]. Incoming plane waves should be from (90, 0). Default asks for forward and backward scattering, depending on what is being asked for.
        '''
        t1 = timer()
        if( (self.verbosity > 0 and self.comm.rank == self.model_rank)):
                print(f'Calculating farfield values... ', end='')
                sys.stdout.flush()
        if(reference):
            meshData = self.refMeshdata
            sols = self.solutions_ref
        else:
            meshData = self.DUTMeshdata
            sols = self.solutions_dut
        ## check what angles to compute
        if(self.Nf > 2 and not returnConvergenceVals):
            angles = np.array([[0, 0], [180, 0]]) ## forward + backward
        elif(returnConvergenceVals or compareToMie):
            nvals = 2*int(360/4) ## must be divisible by 2
            angles = np.zeros((nvals*2, 2))
            angles[:nvals, 0] = np.linspace(-180, 180, nvals) ## first half is the H-plane
            angles[:nvals, 1] = 90
            angles[nvals:, 0] = np.linspace(-180, 180, nvals) ## second half is the E-plane
            angles[nvals:, 1] = 0
            
        
        numAngles = np.shape(angles)[0]
        prefactor = dolfinx.fem.Constant(meshData.mesh, 0j)
        theta = dolfinx.fem.Constant(meshData.mesh, 0j)
        phi = dolfinx.fem.Constant(meshData.mesh, 0j)
        khat = ufl.as_vector([ufl.sin(theta)*ufl.cos(phi), ufl.sin(theta)*ufl.sin(phi), ufl.cos(theta)]) ## in cartesian coordinates 
        phiHat = ufl.as_vector([-ufl.sin(phi), ufl.cos(phi), 0])
        thetaHat = ufl.as_vector([ufl.cos(theta)*ufl.cos(phi), ufl.cos(theta)*ufl.sin(phi), -ufl.sin(theta)])
        n = ufl.FacetNormal(meshData.mesh)('+')
        signfactor = ufl.sign(ufl.inner(n, ufl.SpatialCoordinate(meshData.mesh))) # Enforce outward pointing normal
        exp_kr = dolfinx.fem.Function(self.ScalarSpace)
        
        #eta0 = float(np.sqrt(self.mur_bkg/self.epsr_bkg)) # following Daniel's script, this should really be etar here. The factor would be 1 and gets cancelled out anyway, though
        eta0 = float(np.sqrt(mu0/eps0)) ## must convert to float first
        
        farfields = np.zeros((self.Nf, numAngles, 2), dtype=complex) ## for each frequency and angle, E_theta and E_phi
        
        for b in range(self.Nf):
            freq = self.fvec[b]
            k = 2*np.pi*freq/c0
            E = sols[b][0]('+')
            H = -1/(1j*k*eta0)*ufl.curl(E) ## or possibly B = 1/w k x E, 2*pi/freq*k*ufl.cross(khat, E)
            
            ## can only integrate scalars
            F = signfactor*prefactor*ufl.cross(khat, ( ufl.cross(E, n) + eta0*ufl.cross(khat, ufl.cross(n, H))))*exp_kr
            self.F_theta = dolfinx.fem.form(ufl.dot(thetaHat, F)*self.dS_farfield)
            self.F_phi = dolfinx.fem.form(ufl.dot(phiHat, F)*self.dS_farfield)
                
            for i in range(numAngles):
                theta.value = angles[i,0]*pi/180 # convert to radians first
                phi.value = angles[i,1]*pi/180
            
                def evalFs(): ## evaluates the farfield in some given direction khat
                    khatnp = [np.sin(angles[i,0]*pi/180)*np.cos(angles[i,1]*pi/180), np.sin(angles[i,0]*pi/180)*np.sin(angles[i,1]*pi/180), np.cos(angles[i,0]*pi/180)] ## so I can use it in evalFs as regular numbers - not sure how else to do this
                    exp_kr.interpolate(lambda x: np.exp(1j*k*(khatnp[0]*x[0] + khatnp[1]*x[1] + khatnp[2]*x[2])), self.farfield_cells) ## not sure how to use ufl for this expression.
                    prefactor.value = 1j*k/(4*pi)
                    F_theta = dolfinx.fem.assemble.assemble_scalar(self.F_theta)
                    F_phi = dolfinx.fem.assemble.assemble_scalar(self.F_phi)
                    return np.array((F_theta, F_phi))
                
                farfieldpart = evalFs()
                farfieldparts = self.comm.gather(farfieldpart, root=self.model_rank)
                if(self.comm.rank == 0): ## assemble each part as it is made
                    farfields[b, i] = sum(farfieldparts)
        if(self.comm.rank == 0): ## plotting and returning
            if(self.verbosity > 1):
                print(f'done, in {timer()-t1:.3f} s')
                sys.stdout.flush()
                    
        if(returnConvergenceVals): ## calculate and print some tests
            khatResults = np.zeros((numAngles), dtype=complex) ## should really be a real number, but somehow isnt?
            khatCalc = dolfinx.fem.form(ufl.dot(khat, n)*self.dS_farfield) ## calculate zero from khat . n, for each angle
            for i in range(numAngles):
                theta.value = angles[i,0]*pi/180 # convert to radians first
                phi.value = angles[i,1]*pi/180
                khatPart = dolfinx.fem.assemble.assemble_scalar(khatCalc)
                khatParts = self.comm.gather(khatPart, root=self.model_rank)
                if(self.comm.rank == 0): ## assemble each part as it is made
                    khatResults[i] = sum(khatParts)
            
            areaCalc = 1*self.dS_farfield ## calculate area
            areaPart = dolfinx.fem.assemble.assemble_scalar(dolfinx.fem.form(areaCalc))
            areaParts = self.comm.gather(areaPart, root=self.model_rank)
            
            if(self.comm.rank == 0): ## assemble each part as it is made
                areaResult = sum(areaParts)
                real_area = 4*pi*meshData.FF_surface_radius**2
                if(self.verbosity>2):
                    print(f'khat calc first angle (should be zero): {np.abs(khatResults[0]):.5e}')
                    print(f'Farfield-surface area, calculated vs real (expected): {np.abs(areaResult)} vs {real_area}. Error: {np.abs(areaResult-real_area):.3e}, rel. error: {np.abs((areaResult-real_area)/real_area):.3e}')
                      
                m = np.sqrt(self.material_epsr) ## complex index of refraction - if it is not PEC
                mies = np.zeros_like(angles[:, 1])
                lambdat = c0/freq
                x = 2*pi*meshData.object_radius/lambdat
                for i in range(nvals*2): ## get a miepython error if I use a vector of x, so:
                    if(angles[i, 1] == 90): ## if theta=90, then this is H-plane/perpendicular
                        mies[i] = miepython.i_per(m, x, np.cos((angles[i, 0]*pi/180)), norm='qsca')*pi*meshData.object_radius**2 ## +pi since it seems backwards => forwards
                    else: ## if not, we are changing theta angles and in the parallel plane
                        mies[i] = miepython.i_par(m, x, np.cos((angles[i, 0]*pi/180)), norm='qsca')*pi*meshData.object_radius**2 ## +pi/2 since it seems backwards => forwards
                        
                vals = areaResult, khatResults, farfields, mies # [FF surface area, khat integral], scattering along planes, mie intensities in the scattering directions
                return vals
            else: ## for non-main processes, just return zeros
                return 0, 0, 0, 0
                    
        if(self.comm.rank == 0): ## plotting and returning
                                        
            if(compareToMie and self.Nf < 3): ## make some plots by angle if few freqs. (assuming here that we have many angles)
                for b in range(self.Nf):
                    fig = plt.figure()
                    ax1 = plt.subplot(1, 1, 1)
                    #print('theta',np.abs(farfields[b,:,0]))
                    #print('phi',np.abs(farfields[b,:,1]))
                    #print('intensity',np.abs(farfields[b,:,0])**2 + np.abs(farfields[b,:,1])**2)
                    #plt.plot(angles[:, 1], np.abs(farfields[b,:,0]), label = 'theta-pol')
                    #plt.plot(angles[:, 1], np.abs(farfields[b,:,1]), label = 'phi-pol')'
                    mag = np.abs(farfields[b,:,0])**2 + np.abs(farfields[b,:,1])**2
                    ax1.plot(angles[:nvals, 0], mag[:nvals], label = 'Integrated (H-plane)', linewidth = 1.2, color = 'blue', linestyle = '-') ## -180 so 0 is the forward direction
                    ax1.plot(angles[nvals:, 0], mag[nvals:], label = 'Integrated (E-plane)', linewidth = 1.2, color = 'red', linestyle = '-') ## -90 so 0 is the forward direction
                    
                    ##Calculate Mie scattering
                    lambdat = c0/freq
                    m = np.sqrt(self.material_epsr) ## complex index of refraction - if it is not PEC
                    mie = np.zeros_like(angles[:, 1])
                    x = 2*pi*meshData.object_radius/lambdat
                    for i in range(nvals*2): ## get a miepython error if I use a vector of x, so:
                        if(angles[i, 1] == 90): ## if theta=90, then this is H-plane/perpendicular
                            mie[i] = miepython.i_per(m, x, np.cos((angles[i, 0]*pi/180)), norm='qsca')*pi*meshData.object_radius**2 ## +pi since it seems backwards => forwards
                        else: ## if not, we are changing theta angles and in the parallel plane
                            mie[i] = miepython.i_par(m, x, np.cos((angles[i, 0]*pi/180)), norm='qsca')*pi*meshData.object_radius**2 ## +pi/2 since it seems backwards => forwards
                    
                    ax1.plot(angles[:nvals, 0], mie[:nvals], label = 'Miepython (H-plane)', linewidth = 1.2, color = 'blue', linestyle = '--') ## first part should be H-plane ## -180 so 0 is the forward direction
                    ax1.plot(angles[nvals:, 0], mie[nvals:], label = 'Miepython (E-plane)', linewidth = 1.2, color = 'red', linestyle = '--') ## -90 so 0 is the forward direction
                    
                    ##plot error
                    ax1.plot(angles[:nvals, 0], np.abs(mag[:nvals] - mie[:nvals]), label = 'H-plane Error', linewidth = 1.2, color = 'blue', linestyle = ':')
                    ax1.plot(angles[:nvals, 0], np.abs(mag[nvals:] - mie[nvals:]), label = 'E-plane Error', linewidth = 1.2, color = 'red', linestyle = ':')
                    print(f'Forward-scattering intensity relative error: {np.abs(mag[int(nvals/2)] - mie[int(nvals/2)])/mie[int(nvals/2)]:.2e}, backward: {np.abs(mag[0] - mie[0])/mie[0]:.2e}')
                    plt.title(f'Scattered E-field Intensity Comparison ($\lambda/h=${lambdat/meshData.h:.1f})')
                    ax1.legend()
                    #ax1.set_yscale('log')
                    ax1.grid(True)
                    plt.savefig(self.dataFolder+self.name+'miecomp.png')
                    if(showPlots):
                        plt.show()
                    plt.clf()
            
            if(compareToMie and self.Nf > 2): ## do plots by frequency for forward+backward scattering
                ##Calculate Mie scattering
                m = np.sqrt(self.material_epsr) ## complex index of refraction - if it is not PEC
                mieForward = np.zeros_like(self.fvec)
                mieBackward = np.zeros_like(self.fvec)
                for i in range(len(self.fvec)): ## get a miepython error if I use a vector of x, so:
                    lambdat = c0/self.fvec[i]
                    x = 2*pi*meshData.object_radius/lambdat
                    mieForward[i] = miepython.i_par(m, x, np.cos(pi), norm='qsca') 
                    mieBackward[i] = miepython.i_par(m, x, np.cos(0), norm='qsca')
                
                for i in range(len(angles)):
                    plt.plot(self.fvec/1e9, np.abs(farfields[:, i, 0])**2 + np.abs(farfields[:, i, 1])**2, label = r'sim, $\theta=$'+f'{angles[i, 0]:.0f}, $\phi={angles[i, 1]:.0f}$')
                
                plt.xlabel('Frequency [GHz]')
                plt.ylabel('Intensity')
                plt.plot(self.fvec/1e9, mieForward, linestyle='--', label = r'Mie forward-scattering')
                plt.plot(self.fvec/1e9, mieBackward, linestyle='--', label = r'Mie backward-scattering')
                plt.legend()
                plt.grid()
                plt.tight_layout()
                if(showPlots):
                    plt.show()
            
            return farfields
        else: ## return nan for non-main processes, just in case
            return np.nan
        
    def calcNearField(self, reference=True, direction = 'forward', FEKOcomp=True, showPlots = True):
        '''
        Computes + plots the near-field along a line, using interpolation. Compares to miepython's e_near, which is untested. Should still be similar (for farfield test case)
        Returns array of near-field values calculated and corresponding FEKO values. For epsr = 2 and other matching settings
        :param reference: Whether this is being computed for the DUT case or the reference
        :param direction: 'forward' for on-axis (z) scattering, 'side' for sideways (y) scattering
        :param FEKOcomp: Also plots a comparison with FEKO results
        '''
        t1 = timer()
        if( (self.verbosity > 1 and self.comm.rank == self.model_rank)):
                print(f'Calculating near-field values... ', end='')
                sys.stdout.flush()
        if(reference):
            meshData = self.refMeshdata
            sols = self.solutions_ref
        else:
            meshData = self.DUTMeshdata
            sols = self.solutions_dut

        ## also compare internal electric fields inside the sphere, at distances rs
        numpts = 1001
        rs = np.linspace(0, meshData.PML_radius*.99, numpts)
        negrs = np.linspace(-1*meshData.PML_radius*.99, 0, numpts)
        enears = np.zeros((np.size(rs), 3), dtype=complex)
        enearsbw = np.zeros((np.size(rs), 3), dtype=complex) ## backward (I think)
        
        ### find the simulated values for those radii, at some angle
        points = np.zeros((3, numpts*2))
        if(direction == 'forward'):
            points[2] = np.hstack((negrs, rs)) ## this should be the forward scattering
            if(FEKOcomp):
                FEKOdat = np.loadtxt('TestStuff/ForwardScatt.efe', skiprows=16) #[Xpos, Ypos, Zpos, Exreal, Exim, Eyreal, Eyim, Ezreal, Ezim]
                FEKOpos = FEKOdat[:, 2]
        elif(direction == 'side'):
            points[1] = np.hstack((negrs, rs)) ## this should be the side-scattering (y-axis)
            if(FEKOcomp):
                FEKOdat = np.loadtxt('TestStuff/SidewaysScatt.efe', skiprows=16) #[Xpos, Ypos, Zpos, Exreal, Exim, Eyreal, Eyim, Ezreal, Ezim]
                FEKOpos = FEKOdat[:, 1]
        
        if(FEKOcomp):
            FEKO_Es = np.zeros((np.shape(FEKOdat)[0], 3), dtype=complex)
            FEKO_Es[:, 0] = FEKOdat[:, 3] + 1j*FEKOdat[:, 4] - 1 ## minus 1 to remove incident field?
            FEKO_Es[:, 1] = FEKOdat[:, 5] + 1j*FEKOdat[:, 6]
            FEKO_Es[:, 2] = FEKOdat[:, 7] + 1j*FEKOdat[:, 8]
        
            
        bb_tree = dolfinx.geometry.bb_tree(meshData.mesh, meshData.mesh.topology.dim)
        cells = []
        points_on_proc = [] ## points on this processor
        cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, points.T) # Find cells whose bounding-box collide with the the points
        colliding_cells = dolfinx.geometry.compute_colliding_cells(meshData.mesh, cell_candidates, points.T) # Choose one of the cells that contains the point
        idx_on_proc_list = [] ## list of indices on the MPI process
        for i, point in enumerate(points.T):
            if len(colliding_cells.links(i)) > 0:
                points_on_proc.append(point)
                idx_on_proc_list.append(i)
                cells.append(colliding_cells.links(i)[0])

        E_part = self.solutions_ref[0][0].eval(points_on_proc, cells) ## less/non-interpolated
        E_parts = self.comm.gather(E_part, root=self.model_rank) ### list of lists of E-fields at the indices
        indices = self.comm.gather(idx_on_proc_list, root=self.model_rank) ## list of lists of indices of contained points
        idx_found = [] ## indices that have already been found
        E_values = np.zeros((numpts*2, 3), dtype=complex)
        if (self.comm.rank == 0): ## use the first-found E_values where processes meet (have not checked, but presumably the values should be the same for all processes) - discard any indices that have been found before
            for b in range(len(indices)): ## iterate over the lists of lists
                if(np.size(indices[b]) == 1): ## if it's not a list (presumably this can happen if only 1 element is in, and not sure how the list of E-fields would look like, might not need this conditional - never seen it yet)
                    print('onlyone', indices[b])
                    ##try this stuff if I get an error
                    #idx_found = idx_found + indices[b] ## add it to found list
                    #E_values[indices[b][0], :] = E_parts[b][:] ## use this value for the electric field
                    print(E_parts[b])
                elif(np.size(indices[b]) == 0): # if no elements on a process
                    continue
                for k in range(len(indices[b])): ## check each index in the list
                    idx = indices[b][k] # the index
                    if(idx not in idx_found): ## if the index had not already been found
                        idx_found = idx_found + [idx] ## add it to found list
                        E_values[idx, :] = E_parts[b][k][:] ## use this value for the electric field
        
        if(self.comm.rank == 0 and self.verbosity>1):
            print(f'done, in {timer()-t1:.3f} s')
            sys.stdout.flush()
        
        if(showPlots and self.comm.rank == self.model_rank): ## generate miepython values if necessary
            fig1 = plt.figure()
            ax1 = plt.subplot(1, 1, 1)
            ax1.grid(True)
            fig2 = plt.figure()
            ax2 = plt.subplot(1, 1, 1)
            ax2.grid(True)
            fig3 = plt.figure()
            ax3 = plt.subplot(1, 1, 1)
            ax3.grid(True)
            
            m = np.sqrt(self.material_epsr) ## complex index of refraction - if it is not PEC
            freq = self.fvec[0]
            lambdat = c0/freq
            k = 2*pi/lambdat
            x = k*meshData.object_radius
            coefs = miepython.core.coefficients(m, x, internal=True)
            def planeWaveSph(x):
                '''
                Plane wave to be subtracted from miepython, so only the scattered field is left. Converts to r-, theta-, phi- pols
                :param x: some given position you want to find the field on
                :param k: wavenumber
                '''
                if(direction == 'forward'): ## along the z-axis
                    E_pw = np.array([self.PW_pol[1], self.PW_pol[2], self.PW_pol[0]])
                elif(direction == 'side'): ## along x
                    E_pw = np.array([self.PW_pol[2], self.PW_pol[1], self.PW_pol[0]])
                k_pw = k*self.PW_dir ## z-directed
                E_pw = E_pw*np.exp(-1j*np.dot(k_pw, x))
                return E_pw
            for q in range(len(rs)):
                r = rs[q]
                nr = -1*negrs[q] ## still positive, just in other direction due to angle
                if(np.abs(r)<meshData.object_radius): ## miepython seems to include the incident field inside the sphere
                    t = 1
                else:
                    t = 0
                if(np.abs(nr)<meshData.object_radius): ## miepython seems to include the incident field inside the sphere
                    t2 = 1
                else:
                    t2 = 0
                if(direction == 'forward'):
                    enears[q] = miepython.field.e_near(coefs, 2*pi/k, 2*meshData.object_radius, m, r, pi, pi/2) - t*planeWaveSph(points[:, q])
                    enearsbw[q] = miepython.field.e_near(coefs, 2*pi/k, 2*meshData.object_radius, m, nr, 0, pi/2) - t2*planeWaveSph(points[:, q+numpts])
                elif(direction == 'side'):
                    enears[q] = miepython.field.e_near(coefs, 2*pi/k, 2*meshData.object_radius, m, r, pi/2, pi/2) - t*planeWaveSph(points[:, q])
                    enearsbw[q] = miepython.field.e_near(coefs, 2*pi/k, 2*meshData.object_radius, m, nr, -pi/2, pi/2) - t2*planeWaveSph(points[:, q+numpts])
                
            enears = np.vstack((enearsbw, enears))
            ## plot components
            ax1.plot(np.hstack((negrs, rs)), np.abs(enears[:, 0]), label='r-comp. miepython', linestyle = ':', color = 'red')
            ax1.plot(np.hstack((negrs, rs)), np.abs(enears[:, 1]), label='theta-comp. miepython', linestyle = ':', color = 'blue')
            ax1.plot(np.hstack((negrs, rs)), np.abs(enears[:, 2]), label='phi-comp. miepython', linestyle = ':', color = 'green')
            ax1.plot(np.hstack((negrs, rs)), np.abs(E_values[:, 0]), label='x-comp.', color = 'red')
            ax1.plot(np.hstack((negrs, rs)), np.abs(E_values[:, 1]), label='y-comp.', color = 'blue')
            ax1.plot(np.hstack((negrs, rs)), np.abs(E_values[:, 2]), label='z-comp.', color = 'green')
            ## plot magnitudes
            ax2.plot(np.hstack((negrs, rs)), np.sqrt(np.abs(enears[:, 0])**2+np.abs(enears[:, 1])**2+np.abs(enears[:, 2])**2), label='miepython')
            ax2.plot(np.hstack((negrs, rs)), np.sqrt(np.abs(E_values[:, 0])**2+np.abs(E_values[:, 1])**2+np.abs(E_values[:, 2])**2), label='simulation')
            #ax2.plot(np.hstack((negrs, rs)), np.sqrt(np.abs(E_values2[:, 0])**2+np.abs(E_values2[:, 1])**2+np.abs(E_values2[:, 2])**2), label='simulation, interpolated')
            ## plot real/imags
            ax3.plot(np.hstack((negrs, rs)), np.real(E_values[:, 0]), label='sim., x-pol real')
            ax3.plot(np.hstack((negrs, rs)), np.imag(E_values[:, 0]), label='sim., x-pol imag.')
            ax3.plot(np.hstack((negrs, rs)), np.real(enears[:, 2]), label='miepython, phi-pol real', linestyle = ':')
            ax3.plot(np.hstack((negrs, rs)), np.imag(enears[:, 2]), label='miepython, phi-pol imag.', linestyle = ':')
            ax3.plot(FEKOpos, np.imag(FEKO_Es[:, 0]), label='FEKO, x-pol imag.', linestyle = '--')
            ax3.plot(FEKOpos, np.real(FEKO_Es[:, 0]), label='FEKO, x-pol real', linestyle = '--')
            
            ax1.legend()
            ax2.legend()
            ax3.legend()
            ax1.set_xlabel('Radius [m]')
            ax2.set_xlabel('Radius [m]')
            ax3.set_xlabel('Radius [m]')
            ax1.set_ylabel('E-field Components')
            ax2.set_ylabel('E-field Magnitude')
            ax3.set_ylabel('E-field Components')
            ax1.set_title('Absolute values of components')
            ax2.set_title('E-field magnitudes')
            ax3.set_title('Real/imag. parts of incident pol.')
            
            ax1.axvline(meshData.object_radius, label = 'radius', color = 'black')
            ax1.axvline(-1*meshData.object_radius, color = 'black')
            ax2.axvline(meshData.object_radius, label = 'radius', color = 'black')
            ax2.axvline(-1*meshData.object_radius, color = 'black')
            ax3.axvline(meshData.object_radius, label = 'radius', color = 'black')
            ax3.axvline(-1*meshData.object_radius, color = 'black')
            ax1.axvline(meshData.domain_radius, label = 'radius', color = 'gray')
            ax1.axvline(-1*meshData.domain_radius, color = 'gray')
            ax2.axvline(meshData.domain_radius, label = 'radius', color = 'gray')
            ax2.axvline(-1*meshData.domain_radius, color = 'gray')
            ax3.axvline(meshData.domain_radius, label = 'radius', color = 'gray')
            ax3.axvline(-1*meshData.domain_radius, color = 'gray')
            fig1.tight_layout()
            fig2.tight_layout()
            fig3.tight_layout()
            #plt.show()
            
        if(FEKOcomp and not showPlots and self.comm.rank==self.model_rank): ## can't find bounding boxes for the exact points in the FEKO values, so interpolate
            Erealsx = np.real(E_values[:, 0])
            Eimagsx = np.imag(E_values[:, 0])
            Ereturns = np.interp(FEKOpos, points[1], Erealsx) + 1j*np.interp(FEKOpos, points[1], Eimagsx)
            return Ereturns, FEKO_Es[:, 0]
        elif(FEKOcomp and not showPlots): ## return NaNs to the other ranks
            return np.nan, np.nan