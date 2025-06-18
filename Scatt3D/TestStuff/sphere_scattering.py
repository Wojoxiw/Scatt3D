# Simulate EM scattering of a sphere and compare to Mie solution.
#
# Daniel Sjöberg, 2025-05-21

from mpi4py import MPI
import numpy as np
import dolfinx, dolfinx.fem.petsc
import ufl, basix
import sys
import gmsh
from matplotlib import pyplot as plt

# Scipy not installed in my lunarc account, define constants explicitly
#from scipy.constants import c as c0, mu_0 as mu0, epsilon_0 as eps0, pi
c0 = 299792458.0
mu0 = 4*np.pi*1e-7
eps0 = 1/c0**2/mu0
pi = np.pi
eta0 = np.sqrt(mu0/eps0)


class ScattSphereProblem():
    """Class to hold definitions and functions for simulating scattering against a dielectric sphere."""
    def __init__(self,
        comm=None,    # MPI communicator
        model_rank=0, # Rank on which to build model
        a=1e-2,       # Radius of sphere
        epsr=2-0j,    # Complex permittivity of sphere
        f0=10e9,      # Simulation frequency
        h=3e-3,       # Mesh size
        fem_degree=3, # Degree of finite elements
        ):
        self.comm = comm
        self.model_rank = model_rank
        self.sphere_radius = a
        self.material_epsr = epsr
        self.material_mur = 1.0
        self.f0 = f0
        self.lambda0 = c0/f0
        self.h = h
        self.fem_degree = fem_degree
        self.tdim = 3
        self.fdim = self.tdim - 1
        self.epsr_bkg = 1.0
        self.mur_bkg = 1.0

        # Set some geometry parameters
        self.domain_radius = self.sphere_radius + 0.5*self.lambda0
        self.farfield_radius = self.sphere_radius + 0.25*self.lambda0
        self.pml_radius = self.domain_radius + 0.5*self.lambda0

        # Create the mesh
        self.CreateMesh()
        self.mesh.topology.create_connectivity(self.tdim, self.tdim)
        self.mesh.topology.create_connectivity(self.fdim, self.tdim)
        self.InitializeFEM()
        self.InitializeMaterial()
        self.CalculatePML()
        
    def CreateMesh(self):
        gmsh.initialize()
        if self.comm.rank == self.model_rank:
            gmsh.option.setNumber('General.Verbosity', 1)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", self.h)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.h)

            # Create the spheres making up the geometry
            sphere = [(self.tdim, gmsh.model.occ.addSphere(0, 0, 0, self.sphere_radius))]
            domain = [(self.tdim, gmsh.model.occ.addSphere(0, 0, 0, self.domain_radius))]
            farfield = [(self.tdim, gmsh.model.occ.addSphere(0, 0, 0, self.farfield_radius))]
            pml = [(self.tdim, gmsh.model.occ.addSphere(0, 0, 0, self.pml_radius))]
            
            # Create fragments and dimtags
            outdimTags, outDimTagsMap = gmsh.model.occ.fragment(pml, domain + farfield + sphere)
            sphereDimTags = [x for x in outDimTagsMap[3]]
            domainDimTags = [x[0] for x in outDimTagsMap[1:3]]
            pmlDimTags = [x for x in outDimTagsMap[0] if x not in domainDimTags + sphereDimTags]

            gmsh.model.occ.synchronize()
            
            # Make physical groups for domains and PML
            sphere_marker = gmsh.model.addPhysicalGroup(self.tdim, [x[1] for x in sphereDimTags])
            domain_marker = gmsh.model.addPhysicalGroup(self.tdim, [x[1] for x in domainDimTags])
            pml_marker = gmsh.model.addPhysicalGroup(self.tdim, [x[1] for x in pmlDimTags])

            # Identify far field surface
            farfield_surface = []
            for boundary in gmsh.model.occ.getEntities(dim=self.fdim):
                bbox = gmsh.model.getBoundingBox(boundary[0], boundary[1])
                xmin = bbox[0]
                if np.isclose(xmin, -self.farfield_radius):
                    farfield_surface.append(boundary[1])
            farfield_surface_marker = gmsh.model.addPhysicalGroup(self.fdim, farfield_surface)

            gmsh.model.occ.synchronize()
            gmsh.model.mesh.generate(self.tdim)

            if False:
                gmsh.fltk.run()
                exit()

        else:
            sphere_marker = None
            domain_marker = None
            pml_marker = None
            farfield_surface_marker = None
            
        self.sphere_marker = self.comm.bcast(sphere_marker, root=self.model_rank)
        self.domain_marker = self.comm.bcast(domain_marker, root=self.model_rank)
        self.pml_marker = self.comm.bcast(pml_marker, root=self.model_rank)
        self.farfield_surface_marker = self.comm.bcast(farfield_surface_marker, root=self.model_rank)

        self.mesh, self.subdomains, self.boundaries = dolfinx.io.gmshio.model_to_mesh(gmsh.model, comm=self.comm, rank=self.model_rank, gdim=self.tdim, partitioner=dolfinx.mesh.create_cell_partitioner(dolfinx.cpp.mesh.GhostMode.shared_facet))
            
        gmsh.finalize()


    def InitializeFEM(self):
        # Set up some FEM function spaces and boundary condition stuff.
        curl_element = basix.ufl.element('N1curl', self.mesh.basix_cell(), self.fem_degree)
        self.Vspace = dolfinx.fem.functionspace(self.mesh, curl_element)
        self.ScalarSpace = dolfinx.fem.functionspace(self.mesh, ('CG', self.fem_degree))
        self.Wspace = dolfinx.fem.functionspace(self.mesh, ("DG", 0))

        # Create measures for subdomains and surfaces
        self.dx = ufl.Measure('dx', domain=self.mesh, subdomain_data=self.subdomains, metadata={'quadrature_degree': 5})
        self.dx_dom = self.dx((self.domain_marker, self.sphere_marker))
        self.dx_pml = self.dx(self.pml_marker)
        self.ds = ufl.Measure('ds', domain=self.mesh, subdomain_data=self.boundaries)
        self.dS = ufl.Measure('dS', domain=self.mesh, subdomain_data=self.boundaries) ## capital S for internal facets

        self.dS_farfield = self.dS(self.farfield_surface_marker)
        cells = []
        ff_facets = self.boundaries.find(self.farfield_surface_marker)
        facets_to_cells = self.mesh.topology.connectivity(self.fdim, self.tdim)
        for facet in ff_facets:
            for cell in facets_to_cells.links(facet):
                if cell not in cells:
                    cells.append(cell)
        cells.sort()
        self.farfield_cells = np.array(cells)
        
    def InitializeMaterial(self):
        self.epsr = dolfinx.fem.Function(self.Wspace)
        self.mur = dolfinx.fem.Function(self.Wspace)
        self.epsr.x.array[:] = self.epsr_bkg
        self.mur.x.array[:] = self.mur_bkg

        sphere_cells = self.subdomains.find(self.sphere_marker)
        self.sphere_dofs = dolfinx.fem.locate_dofs_topological(self.Wspace, entity_dim=self.tdim, entities=sphere_cells)
        pml_cells = self.subdomains.find(self.pml_marker)
        self.pml_dofs = dolfinx.fem.locate_dofs_topological(self.Wspace, entity_dim=self.tdim, entities=pml_cells)
        domain_cells = self.subdomains.find(self.domain_marker)
        self.domain_dofs = dolfinx.fem.locate_dofs_topological(self.Wspace, entity_dim=self.tdim, entities=domain_cells)

        self.epsr.x.array[self.sphere_dofs] = self.material_epsr
        self.mur.x.array[self.sphere_dofs] = self.material_mur
        
    def CalculatePML(self):
        '''
        Set up the PML - stretched coordinates to form a perfectly-matched layer which absorbs incoming (perpendicular) waves
         Since we calculate at many frequencies, recalculate this for each freq. (Could also precalc it for each freq...)
        :param k: Frequency used for coordinate stretching.
        '''
        # Set up the PML
        def pml_stretch(y, x, x_dom=0, x_pml=1, n=3, R0=np.exp(-10)):
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
            k = 2*np.pi/self.lambda0
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
        
        x, y, z = ufl.SpatialCoordinate(self.mesh)
        r = ufl.real(ufl.sqrt(x**2 + y**2 + z**2))
        x_stretched = pml_stretch(x, r, x_dom=self.domain_radius, x_pml=self.pml_radius)
        y_stretched = pml_stretch(y, r, x_dom=self.domain_radius, x_pml=self.pml_radius)
        z_stretched = pml_stretch(z, r, x_dom=self.domain_radius, x_pml=self.pml_radius)
        x_pml = ufl.conditional(ufl.ge(abs(r), self.domain_radius), x_stretched, x) ## stretch when outside radius of the domain
        y_pml = ufl.conditional(ufl.ge(abs(r), self.domain_radius), y_stretched, y) ## stretch when outside radius of the domain
        z_pml = ufl.conditional(ufl.ge(abs(r), self.domain_radius), z_stretched, z) ## stretch when outside radius of the domain
        pml_coords = ufl.as_vector((x_pml, y_pml, z_pml))
        self.epsr_pml, self.murinv_pml = pml_epsr_murinv(pml_coords)

    def ComputeSolution(self):
        Eb = dolfinx.fem.Function(self.Vspace)
        E = ufl.TrialFunction(self.Vspace)
        v = ufl.TestFunction(self.Vspace)
        curl_E = ufl.curl(E)
        curl_v = ufl.curl(v)
        nvec = ufl.FacetNormal(self.mesh)
        k00 = dolfinx.fem.Constant(self.mesh, 1j)
        F = ufl.inner(1/self.mur*curl_E, curl_v)*self.dx_dom \
            - ufl.inner(k00**2*self.epsr*E, v)*self.dx_dom \
            + ufl.inner(self.murinv_pml*curl_E, curl_v)*self.dx_pml \
            - ufl.inner(k00**2*self.epsr_pml*E, v)*self.dx_pml \
            - ufl.inner(k00**2*(self.epsr - 1/self.mur*self.mur_bkg*self.epsr_bkg)*Eb, v)*self.dx_dom
        lhs, rhs = ufl.lhs(F), ufl.rhs(F)
        petsc_options = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
        problem = dolfinx.fem.petsc.LinearProblem(lhs, rhs, bcs=[], petsc_options=petsc_options)

        k0 = 2*np.pi/self.lambda0
        k00.value = k0
        PlaneWave = lambda x: np.array([np.exp(-1j*k0*x[2,:]), np.zeros(x.shape[1], dtype=complex), np.zeros(x.shape[1], dtype=complex)])
        Eb.interpolate(PlaneWave)

        self.Eh = problem.solve()

    def ComputeFarField(self, angles = np.array([[90, 180], [90, 0]])):
        numAngles = angles.shape[0]
        prefactor = dolfinx.fem.Constant(self.mesh, 0j)
        n = ufl.FacetNormal(self.mesh)('+')
        signfactor = ufl.sign(ufl.inner(n, ufl.SpatialCoordinate(self.mesh))) # Enforce outward pointing normal
        exp_kr = dolfinx.fem.Function(self.ScalarSpace)
        farfields = np.zeros((numAngles, 2), dtype=complex) ## for each angle, E_theta and E_phi

        k0 = float(2*np.pi/self.lambda0)
        eta0 = float(np.sqrt(mu0/eps0))
        for i in range(numAngles):
#            if self.comm.rank == self.model_rank:
#                print(f'Angle = {angles[i,:]}')
#                sys.stdout.flush()
            theta = angles[i,0]*np.pi/180 # convert to radians first
            phi = angles[i,1]*np.pi/180
            khat_ufl = ufl.as_vector([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]) ## in cartesian coordinates 
            phiHat = ufl.as_vector([-np.sin(phi), np.cos(phi), 0])
            thetaHat = ufl.as_vector([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)])

            E = self.Eh('+')
            H = -1/(1j*k0*eta0)*ufl.curl(E)
                
            ## can only integrate scalars
            F = signfactor*prefactor*ufl.cross(khat_ufl, ( ufl.cross(E, n) + eta0*ufl.cross(khat_ufl, ufl.cross(n, H))))*exp_kr
            self.F_theta = ufl.inner(thetaHat, ufl.conj(F))*self.dS_farfield
            self.F_phi = ufl.inner(phiHat, ufl.conj(F))*self.dS_farfield
#            self.F_theta = signfactor*prefactor*ufl.inner(thetaHat, ufl.cross(khat, ( ufl.cross(E, n) + eta0*ufl.cross(khat, ufl.cross(n, H))) ))*exp_kr*self.dS_farfield
#            self.F_phi = signfactor*prefactor* ufl.inner(phiHat, ufl.cross(khat, ( ufl.cross(E, n) + eta0*ufl.cross(khat, ufl.cross(n, H))) ))*exp_kr*self.dS_farfield
                
            khat = [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)] ## so I can use it in evalFs as regular numbers

            exp_kr.interpolate(lambda x: np.exp(1j*k0*(khat[0]*x[0] + khat[1]*x[1] + khat[2]*x[2])), self.farfield_cells)
            prefactor.value = 1j*k0/(4*np.pi)
                    
            F_theta = dolfinx.fem.assemble.assemble_scalar(dolfinx.fem.form(self.F_theta))
            F_phi = dolfinx.fem.assemble.assemble_scalar(dolfinx.fem.form(self.F_phi))

            farfieldpart = np.array([F_theta, F_phi])
            farfieldparts = self.comm.gather(farfieldpart, root=self.model_rank)
            if(self.comm.rank == 0): ## assemble each part as it is made
                farfields[i] = sum(farfieldparts)
            
        return farfields

    
if __name__ == '__main__':
    # Run a convergence test
        
    comm = MPI.COMM_WORLD
    model_rank = 0

    epsr = 2 - 0j
    a = 1e-2
    f0 = 10e9
    lambda0 = c0/f0
    fem_degree = 1
    hfactors = np.array([10, 20, 30, 40], dtype=float)
    hfactors = np.array([5, 10], dtype=float) # Quick choices for testing

    for hfactor in hfactors:
        h = lambda0/hfactor

        if comm.rank == model_rank:
            print(f'Rank {comm.rank}: Initiating problem, hfactor = {hfactor}')
            sys.stdout.flush()
        p = ScattSphereProblem(comm=comm, model_rank=model_rank, epsr=epsr, a=a, f0=f0, h=h, fem_degree=fem_degree)
        if comm.rank == model_rank:
            print(f'Rank {comm.rank}: Computing solution')
            sys.stdout.flush()
        p.ComputeSolution()
        if comm.rank == model_rank:
            print(f'Rank {comm.rank}: Computing far field')
            sys.stdout.flush()

        # Compute far fields
        cut = np.linspace(-180, 180.0, 3)
        Eplane_angles = []
        Hplane_angles = []
        for angle in cut:
            if angle > 0:
                Eplane_angles.append([angle, 0])
                Hplane_angles.append([angle, 90])
            else:
                Eplane_angles.append([-angle, 180])
                Hplane_angles.append([-angle, 270])
        Eplane_angles = np.array(Eplane_angles)
        Hplane_angles = np.array(Hplane_angles)
        ff_Eplane = p.ComputeFarField(Eplane_angles)
        ff_Hplane = p.ComputeFarField(Hplane_angles)

        # Compute near fields
        Np = 100
        line = np.linspace(-a-0.2*lambda0, a+0.2*lambda0, Np)
        points = np.zeros((3, Np))
        points[1] = line
        E_values = []

        # Interpolate the fields from Nedelec space to DG1 (to preserve discontinuities)
        InterpSpace = dolfinx.fem.functionspace(p.mesh, ('Discontinuous Lagrange', 1, (3,)))
        Ei = dolfinx.fem.Function(InterpSpace)
        Ei.interpolate(p.Eh)
                                                
        from dolfinx import geometry
        bb_tree = geometry.bb_tree(p.mesh, p.mesh.topology.dim)
        cells = []
        points_on_proc = []
        # Find cells whose bounding-box collide with the the points
        cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
        # Choose one of the cells that contains the point
        colliding_cells = geometry.compute_colliding_cells(p.mesh, cell_candidates, points.T)
        for i, point in enumerate(points.T):
            if len(colliding_cells.links(i)) > 0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])

        points_on_proc = np.array(points_on_proc, dtype=np.float64)
        # Introduce some interpolation from edge based to node base (DG1) here
        E_values = p.Eh.eval(points_on_proc, cells)
        if len(points_on_proc) == 1: # For only one point, results of eval need to be encapsulated in an array
            E_values = np.array([E_values])
        points_on_proc_all = comm.gather(points_on_proc, root=model_rank)
        E_values_all = comm.gather(E_values, root=model_rank)
        if comm.rank == model_rank:
            # There should be a more pretty way to sum up these things
            points_vec = []
            E_values_vec = []
            for x, E in zip(points_on_proc_all, E_values_all):
                if x.size > 0:
                    points_vec = points_vec + x.tolist()
                    E_values_vec = E_values_vec + E.tolist()
            points_vec = np.array(points_vec)
            E_values_vec = np.array(E_values_vec)            
            idx = np.argsort(points_vec[:,1])
            points_vec = points_vec[idx]
            E_values_vec = E_values_vec[idx]
        else:
            points_vec = None
            E_values_vec = None
            
        # Save data
        if comm.rank == model_rank:
            ffdata = np.vstack([cut, ff_Eplane[:,0], ff_Eplane[:,1], ff_Hplane[:,0], ff_Hplane[:,1]]).T
            np.savetxt(f'ffdata_{lambda0/h}.dat', ffdata, delimiter=',', header="""# Angle (deg), theta pol E-plane, phi pol E-plane, theta pol H-plane, phi pol H-plane""")
            # Read file with np.genfromtxt(filename, dtype=None, delimiter=',', skip_header=1)
            nfdata = np.hstack([points_vec, E_values_vec])
            np.savetxt(f'nfdata_{lambda0/h}.dat', nfdata, delimiter=',', header="""# x, y, z, Ex, Ey, Ez""")

    exit()

    
    
