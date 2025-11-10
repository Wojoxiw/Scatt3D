# encoding: utf-8
## this file makes the mesh

from mpi4py import MPI
import numpy as np
import dolfinx
import gmsh
from scipy.constants import pi, c as c0
from timeit import default_timer as timer
import sys

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

class MeshInfo():
    """Data structure for the mesh (all geometry) and related metadata."""
    def __init__(self,
                 comm,
                 fname = '',
                 reference = True,
                 f0 = 10e9,
                 verbosity = 0,
                 h = 1/15,
                 domain_geom = 'sphere',#'domedCyl', 
                 object_geom = 'cubic',#'complex1'
                 defect_geom = 'cylinder',#'complex1'
                 domain_radius = 1.92,
                 domain_height = 1.5,
                 PML_thickness = 0,
                 dome_height = 0.5,
                 antenna_width = 0.7625, 
                 antenna_height = 0.3625,
                 antenna_depth = 1/10,
                 antenna_type = 'waveguide',
                 N_antennas = 10,
                 antenna_radius = 0,
                 antenna_z_offset = 0,
                 object_radius = 1.06,
                 object_height = 1.25,
                 object_offset = np.array([0, 0, 0]),
                 defect_radius = 0.175,
                 defect_height = 0.3,
                 defect_angles = [0, 0, 0],
                 defect_offset = np.array([0, 0, 0]),
                 viewGMSH = True,
                 FF_surface = False,
                 order = 1,
                ):
        '''
        Makes it - given various inputs
        :param comm: the MPI communicator
        :param fname: Mesh filename + directory. If empty, does not save it (currently does not ever save it)
        :param reference: Does not include any defects in the mesh.
        :param f0: Design frequency - things will be scaled by the corresponding wavelength
        :param h: typical mesh size, in fractions of a wavelength
        :param verbosity: This is passed to gmsh, also if > 0, I print more stuff
        :param domain_geom: The geometry of the domain (and PML).
        :param object_geom: Geometry of the object ('sphere', 'cylinder', 'cubic', 'None')
        :param defect_geom: Geometry of the defect.
        :param domain_radius:
        :param domain_height:
        :param PML_thickness: If not specified, calculated to give x mesh cells between the domain and the edge of the PML
        :param dome_height:
        :param antenna_width: Width of antenna apertures, 22.86 mm
        :param antenna_height: Height of antenna apertures
        :param antenna_depth: Depth of antenna box
        :param antenna_type: If 'waveguide', use the above geometry. If 'patch', testing patch antenna
        :param N_antennas:
        :param antenna_radius: Radius at which antennas are placed
        :param antenna_z_offset: Height (from the middle of the sim.) at which antennas are placed. Default to centering on the x-y plane
        :param object_radius: If object is a sphere (or cylinder), the radius
        :param object_height: If object is a cylinder, the height
        :param object_offset: The object is shifted this far (in wavelengths)
        :param defect_radius: If defect is a sphere (or cylinder), the radius
        :param defect_height: If defect is a cylinder, the height
        :param defect_angles: [x, y, z] angles to rotate about these axes
        :param defect_offset: The defect is shifted this far from the centre of the object (in wavelengths)
        :param viewGMSH: If True, plots the mesh after creation then exits
        :param FF_surface: If True, creates a spherical shell with a radius slightly lower than the domain's, to calculate the farfield on (domain_geom should also be spherical)
        :param order: Order of the mesh elements - have to switch from xdmf to vtx or vtk when going above 2? They don't work straightforwardly
        '''
        
        self.comm = comm                               # MPI communicator
        self.model_rank = 0                   # Rank of the master model - for saving, plotting, etc.
        self.lambda0 = c0/f0                         # Design wavelength (things are scaled from this)
        if(fname == ''): # if not given a name, just use 'mesh'
            fname = 'mesh.msh'
        self.fname = fname                                  # Mesh filename (+location from script)
        self.h = h * self.lambda0
        self.domain_geom = domain_geom            # setting to choose domain geometry - current only 'domedCyl' exists
        self.verbosity = verbosity
        self.reference = reference
        
        self.tdim = 3 ## Tetrahedra dimensionality - 3D
        self.fdim = self.tdim - 1 ## Facet dimensionality - 2D
        self.FF_surface = FF_surface
        self.order = min(order, 2) ## capped at 2, since xdmf can't save above order 2. Also, doesn't seem to make a huge difference
        
        
        if(PML_thickness == 0): ## if not specified, calculate it
            PML_thickness = 1/2 #h*10 ## try to have some mesh cells in thickness, or wavelength
        
        ## Set up geometry variables - in units of lambda0 unless otherwise specified
        if(self.domain_geom == 'domedCyl'): ## a cylinder with an oblate-spheroid 'dome' added over and underneath
            self.domain_radius = domain_radius * self.lambda0
            self.domain_height = domain_height * self.lambda0
            
            self.PML_radius = self.domain_radius + PML_thickness * self.lambda0
            self.PML_height = self.domain_height + 2*PML_thickness * self.lambda0
            
            self.dome_height = dome_height * self.lambda0 # How far beyond the domain the cylindrical base the spheroid extends
            if(self.dome_height > 0): ## if 0, just make the cylinder. Otherwise, calculate the spheroid parameters R_extra and a
                self.domain_spheroid_extraRadius = self.domain_radius*(-1 + np.sqrt(1-( 1 - 1/(1- (self.domain_height/2/(self.domain_height/2+self.dome_height))**2 ) )) )       
                self.domain_a = (self.domain_height/2+self.dome_height)/(self.domain_radius+self.domain_spheroid_extraRadius)
                self.PML_spheroid_extraRadius = self.PML_radius*(-1 + np.sqrt(1-( 1 - 1/(1- (self.PML_height/2/(self.PML_height/2+self.dome_height))**2 ) )))
                self.PML_a = (self.PML_height/2+self.dome_height)/(self.PML_radius+self.PML_spheroid_extraRadius)
        elif(self.domain_geom == 'sphere'):
            self.domain_radius = domain_radius * self.lambda0
            self.PML_radius = self.domain_radius + PML_thickness * self.lambda0
            if(self.FF_surface):
                self.FF_surface_radius = self.domain_radius - self.lambda0*.25 #self.h*2 ## just a bit less than the domain's radius
        else:
            print('Invalid geometry type in MeshInfo, exiting...')
            exit()
        self.PML_thickness = PML_thickness*self.lambda0 ## for potential use later
        
        ## Antenna geometry/other parameters:
        
        self.N_antennas = N_antennas ## number of antennas
        self.antenna_type = antenna_type
        if(antenna_radius == 0): ## if not given a radius, put them near the edge of the domain
            self.antenna_radius = self.domain_radius - antenna_height * self.lambda0
        else:
            self.antenna_radius = antenna_radius * self.lambda0
        self.antenna_z_offset = antenna_z_offset * self.lambda0
        self.antenna_width = antenna_width * self.lambda0
        self.antenna_height = antenna_height * self.lambda0
        self.antenna_depth = antenna_depth * self.lambda0
        if(self.antenna_type == 'patch'): ## specify the dimensions here, for a patch active near 10 GHz
            self.antenna_width = 26e-3 ## the width
            self.antenna_height = 1e-3 ## the height
            self.antenna_depth = 18e-3 ## the length (in x)
            self.patch_length = 9e-3
            self.patch_width = 13e-3
            self.coax_inr = .65e-3; self.coax_outr = 2.1e-3; self.coax_outh = 1e-3 ## coaxial inner and outer radii, and the height it extends beyond the substrate
            self.feed_offsetx = 2e-3
        
        self.phi_antennas = np.linspace(0, 2*pi, N_antennas + 1)[:-1] ## placement angles
        self.pos_antennas = np.array([[self.antenna_radius*np.cos(phi), self.antenna_radius*np.sin(phi), self.antenna_z_offset] for phi in self.phi_antennas]) ## placement positions
        self.rot_antennas = self.phi_antennas + np.pi/2 ## rotation so that they face the center
        self.kc = pi/self.antenna_width ## cutoff wavenumber
        ## Object + defect(s) parameters
        if(object_geom == 'sphere'):
            self.object_radius = object_radius * self.lambda0
        elif(object_geom == 'cylinder'):
            self.object_radius = object_radius * self.lambda0
            self.object_height = object_height * self.lambda0
        elif(object_geom == 'cubic'):
            self.object_length = object_radius * self.lambda0
        elif(object_geom == 'complex1'):
            self.object_scale = object_radius * self.lambda0
        elif(object_geom == '' or object_geom is None):
            pass
        else:
            print('Nonvalid object geom, exiting...')
            exit()
        self.object_geom = object_geom
        self.object_offset = object_offset * self.lambda0
        
        self.defect_angles = defect_angles ## [x, y, z] rotations
        self.defect_offset = defect_offset * self.lambda0
        if(defect_geom == 'cylinder'):
            self.defect_radius = defect_radius * self.lambda0
            self.defect_height = defect_height * self.lambda0
        elif(defect_geom == 'complex1'):
            self.defect_radius = defect_radius * self.lambda0
            self.defect_height = defect_height * self.lambda0
        elif(defect_geom == '' or defect_geom is None):
            pass ## no defect
        else:
            print('Nonvalid defect geom, exiting...')
            exit()
        self.defect_geom = defect_geom
            
        ## Finally, actually make the mesh
        self.createMesh(viewGMSH)
        
    #@profile
    def createMesh(self, viewGMSH):
        t1 = timer()
        gmsh.initialize()
        if self.comm.rank == self.model_rank: ## make all the definitions through the master-rank process
            gmsh.model.add('The Model') ## name for the whole thing
            ## Give some mesh settings: verbosity, max. and min. mesh lengths
            gmsh.option.setNumber('General.Verbosity', self.verbosity)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", self.h/4)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.h*2)
            gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)
            gmsh.logger.start() ## I don't know what these logs are, or how to view them
             
            PECSurfacePts = []; ## points that are on only PEC surfaces
            antennaSurfacePts = []; ## points that are on only the antenna surface
            antennas_DimTags = []
            ## Make the antennas
            if(self.antenna_type == 'waveguide'):
                x_antenna = np.zeros((self.N_antennas, 3))
                x_pec = np.zeros((self.N_antennas, 5, 3)) ### for each antenna, and PEC surface (of which there are 5), a position of that surface
                for n in range(self.N_antennas): ## make each antenna, and prepare its surfaces to either be PEC or be the excitation surface
                    box = gmsh.model.occ.addBox(-self.antenna_width/2, -self.antenna_depth, -self.antenna_height/2, self.antenna_width, self.antenna_depth, self.antenna_height) ## the antenna surface at (0, 0, 0)
                    gmsh.model.occ.rotate([(self.tdim, box)], 0, 0, 0, 0, 0, 1, self.rot_antennas[n])
                    gmsh.model.occ.translate([(self.tdim, box)], self.pos_antennas[n,0], self.pos_antennas[n,1], self.pos_antennas[n,2])
                    antennas_DimTags.append((self.tdim, box))
                    x_antenna[n] = self.pos_antennas[n, :] ## the translation to the antenna's position
                    Rmat = np.array([[np.cos(self.rot_antennas[n]), -np.sin(self.rot_antennas[n]), 0],
                                     [np.sin(self.rot_antennas[n]), np.cos(self.rot_antennas[n]), 0],
                                     [0, 0, 1]]) ## matrix for rotation about the z-axis
                    x_pec[n, 0] = x_antenna[n] + np.dot(Rmat, np.array([0, -self.antenna_depth/2, -self.antenna_height/2])) ## bottom surface (in z)
                    x_pec[n, 1] = x_antenna[n] + np.dot(Rmat, np.array([0, -self.antenna_depth/2,  self.antenna_height/2])) ## top surface (in z)
                    x_pec[n, 2] = x_antenna[n] + np.dot(Rmat, np.array([-self.antenna_width/2, -self.antenna_depth/2, 0])) ## left surface (in x)
                    x_pec[n, 3] = x_antenna[n] + np.dot(Rmat, np.array([self.antenna_width/2, -self.antenna_depth/2, 0])) ## right surface (in x)
                    x_pec[n, 4] = x_antenna[n] + np.dot(Rmat, np.array([0, -self.antenna_depth, 0])) ## back surface (in y)
                    antennaSurfacePts.append(x_antenna[n]) ## (0, 0, 0) - the antenna surface
                    for i in range(5):
                        PECSurfacePts.append(x_pec[n, i])
            elif(self.antenna_type == 'patch'): # 1 patch antenna near the centre, for now
                box = gmsh.model.occ.addBox(-self.antenna_depth/2, -self.antenna_width/2, -self.antenna_height/2, self.antenna_depth, self.antenna_width, self.antenna_height) ## box for antenna surface + dielectric + GP at (0, 0, 0)
                patch = gmsh.model.occ.addRectangle(-self.patch_length/2, -self.patch_width/2, self.antenna_height/2, self.patch_length, self.patch_width) ## the patch itself
                ## now fragment the surface so that it will have the patch-shape in it
                gmsh.model.occ.fragment([(2, box)], [(2, patch)]) ## fragment the surface, otherwise the patch is subsumed
                
                coax_outer = gmsh.model.occ.addCylinder(self.feed_offsetx,0,-self.antenna_height/2-self.coax_outh,0,0,self.coax_outh, self.coax_outr)
                coax_inner = gmsh.model.occ.addCylinder(self.feed_offsetx,0,-self.antenna_height/2-self.coax_outh,0,0,self.coax_outh+self.antenna_height, self.coax_inr)
                ## subtract the outer coax from the box, then the inner coax from the outer
                box = gmsh.model.occ.cut([(self.tdim, box)], [(self.tdim, coax_outer)], removeTool=False)[0][0][1]
                coax_outer = gmsh.model.occ.cut([(self.tdim, coax_outer)], [(self.tdim, coax_inner)], removeTool=False)[0][0][1]
                dielectric = gmsh.model.occ.fuse([(self.tdim, box)], [(self.tdim, coax_outer)])[0][0]
                
                antennas_DimTags.append((self.tdim, coax_inner))
                
                antennaSurfacePts.append([self.feed_offsetx, self.coax_inr/2+self.coax_outr/2, -self.antenna_height/2-self.coax_outh]) ## the surface of the bottom of the outer cylinder of the coax - the radiating port
                PECSurfacePts.append([0, 0, self.antenna_height/2]) ## centre of the patch
                PECSurfacePts.append([-self.feed_offsetx, 0, -self.antenna_height/2]) ## should be in just the ground plane
                PECSurfacePts.append([self.feed_offsetx, 0, -self.antenna_height/2-self.coax_outh]) ## bottom circle of the inner coax
                PECSurfacePts.append([self.feed_offsetx, 0, self.antenna_height/2]) ## top circle of the inner coax
                
            matDimTags = []; defectDimTags = []; defectDimTags2 = []
            if(self.antenna_type == 'patch'): ## the interior of the patch is dielectric; for now it is the object
                matDimTags.append(dielectric)
            ## Make the object and defects (if not a reference case)
            if(self.object_geom == 'sphere'):
                obj = gmsh.model.occ.addSphere(0,0,0, self.object_radius) ## add it to the origin
                matDimTags.append((self.tdim, obj))
            elif(self.object_geom == 'cylinder'):
                obj = gmsh.model.occ.addCylinder(0,0,-self.object_height/2,0,0,self.object_height, self.object_radius) ## add it to the origin
                matDimTags.append((self.tdim, obj))
            elif(self.object_geom == 'cubic'):
                obj = gmsh.model.occ.addBox(-self.object_length/2,-self.object_length/2,-self.object_length/2,self.object_length,self.object_length,self.object_length) ## add it to the origin
                matDimTags.append((self.tdim, obj))
            elif(self.object_geom == 'complex1'): ## do a sort of plane-shaped thing, making sure to avoid symmetry. It is also large. Defect should have same name
                part1 = gmsh.model.occ.addSphere(0,0,0, self.object_scale*1.2) ## long ellipsoid in the centre
                gmsh.model.occ.dilate([(self.tdim, part1)], 0, 0, 0, 1, 0.374, 0.18)
                gmsh.model.occ.rotate([(self.tdim, part1)], 0, 0, 0, 0, 1, 0, 15*pi/180)
                
                part2 = gmsh.model.occ.addBox(0, 0, 0, self.object_scale*.3, self.object_scale*.95, self.object_scale*.2) ## the 'wings'
                gmsh.model.occ.rotate([(self.tdim, part2)], 0, 0, 0, 0, 1, 0, 22*pi/180)
                gmsh.model.occ.rotate([(self.tdim, part2)], 0, 0, 0, 0, 0, 1, 38*pi/180)
                gmsh.model.occ.translate([(self.tdim, part2)], self.object_scale*0.4, -self.object_scale*0.05, -self.object_scale*0.11)
                part3 = gmsh.model.occ.copy([(self.tdim, part2)])[0][1] ## [0][1] to get the tag
                gmsh.model.occ.mirror([(self.tdim, part3)], 0, 1, 0, 0)
                
                part4 = gmsh.model.occ.addSphere(-self.object_scale*2.8,0,0, self.object_scale*0.65) ## 'tail' in the back
                gmsh.model.occ.dilate([(self.tdim, part4)], 0, 0, 0, 0.2, 1, 0.4)
                gmsh.model.occ.rotate([(self.tdim, part4)], 0, 0, 0, 1, 0, 0, 12*pi/180)
                gmsh.model.occ.translate([(self.tdim, part4)], -self.object_scale*0.15, 0, self.object_scale*0.15)
                
                
                defectPart3 = gmsh.model.occ.copy([(self.tdim, part3)])[0][1] ## [0][1] to get the tag - make part 3 into part of the defect
                defect3 = gmsh.model.occ.cut([(self.tdim, defectPart3)], [(self.tdim, part1)], removeTool=False)[0][0][1] ## result should just be wing
                
                obj = gmsh.model.occ.fuse([(self.tdim, part1)], [(self.tdim, part2),(self.tdim, part3),(self.tdim, part4)])[0][0] ## fuse with everything except defect wing ## [0][0] to get  dimTags
                matDimTags.append(obj) ## the material fills the object
            gmsh.model.occ.translate(matDimTags, self.object_offset[0], self.object_offset[1], self.object_offset[2]) ## add offset
            defectDimTags = []
            defectDimTags2 = [] ## for some geometry, have a second set of dimTags so it can be set to a different epsr
            if(not self.reference):
                if(self.defect_geom == 'cylinder'):
                    defectDimTags = []
                    defect1 = gmsh.model.occ.addCylinder(0,0,-self.defect_height/2,0,0,self.defect_height, self.defect_radius) ## cylinder centered on the origin
                    ## apply some rotations around the origin, and each axis
                    gmsh.model.occ.rotate([(self.tdim, defect1)], 0, 0, 0, 1, 0, 0, self.defect_angles[0])
                    gmsh.model.occ.rotate([(self.tdim, defect1)], 0, 0, 0, 0, 1, 0, self.defect_angles[1])
                    gmsh.model.occ.rotate([(self.tdim, defect1)], 0, 0, 0, 0, 0, 1, self.defect_angles[2])
                    defectDimTags.append((self.tdim, defect1))
                    
                    ## also a second cylinder
                    defect2 = gmsh.model.occ.addCylinder(self.defect_radius*2.4,-self.defect_radius*2.4,-self.defect_height/4,0,0,self.defect_height/2, self.defect_radius/2)
                    defectDimTags.append((self.tdim, defect2))
                elif(self.defect_geom == 'complex1'): ## do a sort of plane-shaped thing, making sure to avoid symmetry
                    defectDimTags = []
                    defect1 = gmsh.model.occ.addCylinder(0,0,-self.defect_height/2,0,0,self.defect_height, self.defect_radius) ## small cylinder centered on the origin
                    gmsh.model.occ.dilate([(self.tdim, defect1)], 0, 0, 0, 1, 0.64, 0.17)
                    
                    defect2 = gmsh.model.occ.addCylinder(0,0,-self.defect_height/8,0,0,self.defect_height/4, self.defect_radius*0.3) ## tall cylinder back in the thing
                    gmsh.model.occ.rotate([(self.tdim, defect2)], 0, 0, 0, 0, 1, 0, 30*pi/180)
                    gmsh.model.occ.translate([(self.tdim, defect2)], -self.object_scale*0.61, -self.object_scale*0.18, self.object_scale*0.18)
                    defectDimTags.append((self.tdim, defect1))
                    defectDimTags.append((self.tdim, defect3))
                    defectDimTags2.append((self.tdim, defect2))
                gmsh.model.occ.translate(defectDimTags, self.object_offset[0]+self.defect_offset[0], self.object_offset[1]+self.defect_offset[1], self.object_offset[2]+self.defect_offset[2]) ## add offset
            
            ## Make the domain and the PML
            if(self.domain_geom == 'domedCyl'):
                domain_cyl = gmsh.model.occ.addCylinder(0, 0, -self.domain_height/2, 0, 0, self.domain_height, self.domain_radius)
                domain = [(self.tdim, domain_cyl)] # dim, tags
                pml_cyl = gmsh.model.occ.addCylinder(0, 0, -self.PML_height/2, 0, 0, self.PML_height, self.PML_radius)
                pml = [(self.tdim, pml_cyl)] # dim, tags
                if(self.dome_height>0): ## add a spheroid domed top and bottom with some specified extra height, that passes through the cylindrical 'corner' (to try to avoid waves being parallel to the PML)
                    domain_spheroid = gmsh.model.occ.addSphere(0, 0, 0, self.domain_radius+self.domain_spheroid_extraRadius)
                    gmsh.model.occ.dilate([(self.tdim, domain_spheroid)], 0, 0, 0, 1, 1, self.domain_a)
                    domain_extraheight_cyl = gmsh.model.occ.addCylinder(0, 0, -self.domain_height/2-self.dome_height, 0, 0, self.domain_height+self.dome_height*2, self.domain_radius)
                    domed_ceilings = gmsh.model.occ.intersect([(self.tdim, domain_spheroid)], [(self.tdim, domain_extraheight_cyl)])
                    domain = gmsh.model.occ.fuse([(self.tdim, domain_cyl)], domed_ceilings[0])[0] ## [0] to get  dimTags
                    
                    pml_spheroid = gmsh.model.occ.addSphere(0, 0, 0, self.PML_radius+self.PML_spheroid_extraRadius)
                    gmsh.model.occ.dilate([(self.tdim, pml_spheroid)], 0, 0, 0, 1, 1, self.PML_a)
                    pml_extraheight_cyl = gmsh.model.occ.addCylinder(0, 0, -self.PML_height/2-self.dome_height, 0, 0, self.PML_height+self.dome_height*2, self.PML_radius)
                    domed_ceilings = gmsh.model.occ.intersect([(self.tdim, pml_spheroid)], [(self.tdim, pml_extraheight_cyl)])
                    pml = gmsh.model.occ.fuse([(self.tdim, pml_cyl)], domed_ceilings[0])[0]
                else:
                    domain = [(self.tdim, domain)] # needs to be dim, tags
                    pml = [(self.tdim, pml)] # needs to be dim, tags
            elif(self.domain_geom == 'sphere'):
                domain = gmsh.model.occ.addSphere(0, 0, 0, self.domain_radius)
                pml = gmsh.model.occ.addSphere(0, 0, 0, self.PML_radius)
                domain = [(self.tdim, domain)] # needs to be dim, tags
                pml = [(self.tdim, pml)] # needs to be dim, tags
            FF_surface_dimTags = []
            if(self.FF_surface):
                FF_c = np.array([0, 0, 0]) ## centred on the origin.
                FF_surface = gmsh.model.occ.addSphere(FF_c[0], FF_c[1], FF_c[2], self.FF_surface_radius)
                FF_surface_dimTags = [(self.tdim, FF_surface)]
                
            # Create fragments and dimtags
            outDimTags, outDimTagsMap = gmsh.model.occ.fragment(pml, domain + matDimTags + defectDimTags + defectDimTags2 + FF_surface_dimTags + antennas_DimTags)
            
            removeDimTags = [] ## remove these surfaces for later addition to PEC or FF surfaces
            if(self.N_antennas > 0):
                if(self.antenna_type=='patch'):
                    removeDimTags = [x for x in [y[0] for y in outDimTagsMap[-self.N_antennas*2:]]] ## double from the regular waveguide since it has two volumes to remove
                else:
                    removeDimTags = [x for x in [y[0] for y in outDimTagsMap[-self.N_antennas:]]] ## last few should be the antennas
            if(not self.reference):
                ndefects = len(defectDimTags)
                mapHere = []
                for n in np.arange(ndefects):
                    mapHere = mapHere+outDimTagsMap[3+n]
                defectDimTags = [x for x in mapHere if x not in removeDimTags]
                ndefects2 = len(defectDimTags2)
                mapHere2 = []
                for n in np.arange(ndefects2):
                    mapHere2 = mapHere2+outDimTagsMap[3+ndefects+n]
                defectDimTags2 = [x for x in mapHere2 if x not in removeDimTags+defectDimTags]
            else:
                defectDimTags = []
                defectDimTags2 = []
            if(self.object_geom=='None'):
                matDimTags = []
            else:
                matDimTags = [x for x in outDimTagsMap[2] if x not in defectDimTags+defectDimTags2+removeDimTags]
            domainDimTags = [x for x in outDimTagsMap[1] if x not in removeDimTags+matDimTags+defectDimTags+defectDimTags2]
            pmlDimTags = [x for x in outDimTagsMap[0] if x not in domainDimTags+defectDimTags+defectDimTags2+matDimTags+removeDimTags]
            gmsh.model.occ.remove(removeDimTags)
            gmsh.model.occ.synchronize()
            # Make physical groups for domains and PML
            mat_marker = gmsh.model.addPhysicalGroup(self.tdim, [x[1] for x in matDimTags])
            defect_marker = gmsh.model.addPhysicalGroup(self.tdim, [x[1] for x in defectDimTags])
            defect_marker2 = gmsh.model.addPhysicalGroup(self.tdim, [x[1] for x in defectDimTags2])
            domain_marker = gmsh.model.addPhysicalGroup(self.tdim, [x[1] for x in domainDimTags])
            pml_marker = gmsh.model.addPhysicalGroup(self.tdim, [x[1] for x in pmlDimTags])
            
            # Identify antenna surfaces and make physical groups (by checking the Center-of-Mass of each surface entity)
            pec_surface = []
            antenna_surface = []
            farfield_surface = []
            
            distTol = 1e-6 ## close enough?
            for boundary in gmsh.model.occ.getEntities(dim=self.fdim):
                #print('boundary',boundary, gmsh.model.getType(2, boundary[1]))
                CoM = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1]) ## 'centre of mass'
                bbox = gmsh.model.getBoundingBox(boundary[0], boundary[1]) ## 'bounding box'
                
                for point in PECSurfacePts: ## any surface with this point is assumed to be PEC
                    pt = gmsh.model.occ.addPoint(point[0], point[1], point[2]) ## just for finding distance
                    dist = gmsh.model.occ.getDistance(boundary[0], boundary[1], 0, pt)[0]
                    if(dist<distTol):
                        pec_surface.append(boundary[1])
                        print('inPEC', boundary[1])
                    gmsh.model.occ.remove([(0, pt)]) ## clean-up
                    
                for point in antennaSurfacePts: ## any surface with this point is assumed to be an antenna radiating surface
                    pt = gmsh.model.occ.addPoint(point[0], point[1], point[2]) ## just for finding distance
                    dist = gmsh.model.occ.getDistance(boundary[0], boundary[1], 0, pt)[0]
                    if(dist<distTol):
                        antenna_surface.append(boundary[1])
                        print('inAnt', boundary[1])
                    gmsh.model.occ.remove([(0, pt)]) ## clean-up
                    
                if(self.FF_surface):
                    if(np.isclose(bbox[0], -self.FF_surface_radius)): ## bbox[0] should be the minimum x-coordinate? As a sphere, this should be the radius
                        farfield_surface.append(boundary[1])
                        
                        
            pec_surface_marker = gmsh.model.addPhysicalGroup(self.fdim, pec_surface)
            antenna_surface_markers = [gmsh.model.addPhysicalGroup(self.fdim, [s]) for s in antenna_surface]
            farfield_surface_marker = gmsh.model.addPhysicalGroup(self.fdim, farfield_surface)
            
            if(True): ### one option - Try to reduce the mesh size within the object to h/2, h outside. Reality has h/2 inside, slow dropoff to h outside
                objectMeshField = gmsh.model.mesh.field.add("Constant")
                gmsh.model.mesh.field.setNumber(objectMeshField, "VIn", self.h/2)
                gmsh.model.mesh.field.setNumber(objectMeshField, "VOut", self.h)
                gmsh.model.mesh.field.setNumbers(objectMeshField, 'VolumesList', [x[1] for x in matDimTags+defectDimTags+defectDimTags2])
                 
                domainMeshField = gmsh.model.mesh.field.add("Constant")
                gmsh.model.mesh.field.setNumber(domainMeshField, "VIn", self.h)
                 
                minMeshField = gmsh.model.mesh.field.add("Min") ## currently this is the same as just using the one constant field
                gmsh.model.mesh.field.setNumbers(minMeshField, "FieldsList", [objectMeshField, domainMeshField])
 
                gmsh.model.mesh.field.setAsBackgroundMesh(minMeshField)
                
            gmsh.model.occ.synchronize()
            
            for boundary in gmsh.model.occ.getEntities(dim=self.fdim):
                print('boundary',boundary, gmsh.model.getType(2, boundary[1]))
            
            gmsh.model.mesh.generate(self.tdim)
            #gmsh.write(self.fname) ## Write the mesh to a file. I have never actually looked at this, so I've commented it out
            ## Apparently gmsh's Python bindings don't allow all that I need for the below to work...
            #===================================================================
            # if(self.reference): ## mark the cells corresponding to the defect now. Need to make this after mesh generation so it does not affect the meshing... there must be a better way to do this
            #     tet_type = 4  # tetrahedron
            #     elem_tags, elem_node_tags = gmsh.model.mesh.getElementsByType(tet_type)
            #     elem_node_tags = np.array(elem_node_tags).reshape(-1, 4)
            #     elem_tags = np.array(elem_tags)
            #     node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            #     node_coords = node_coords.reshape(-1, 3)
            #     tag_to_index = {tag: i for i, tag in enumerate(node_tags)}
            #     tet_idx = np.vectorize(tag_to_index.get)(elem_node_tags)
            #     cellcenters = node_coords[tet_idx].mean(axis=1)
            #     
            #     gmsh.model.add('Defect Tool') ## just the defect, since apparently gmsh has no way to check if a cell is within another object without influencing the meshing
            #     defectDimTags = makeDefect()
            #     gmsh.model.occ.synchronize()
            #     gmsh.model.mesh.generate(2)
            #      
            #     node_tags_t, node_coords_t, _ = gmsh.model.mesh.getNodes()
            #     node_coords_t = node_coords_t.reshape(-1, 3)
            #     tag_to_index_t = {tag: i for i, tag in enumerate(node_tags_t)}
            #     tri_type = 2
            #     _, elem_node_tags_t = gmsh.model.mesh.getElementsByType(tri_type)
            #     tris = np.array(elem_node_tags_t).reshape(-1, 3)
            #     tris = np.vectorize(tag_to_index_t.get)(tris)
            #     tool_mesh = trimesh.Trimesh(vertices=node_coords_t, faces=tris, process=True)
            #     
            #     inside = tool_mesh.contains(cellcenters)
            #     inside_tags = elem_tags[inside]
            #     
            #     
            #     #===============================================================
            #     # defect_surfaces = gmsh.model.getBoundary(defectDimTags, oriented=False, recursive=True)
            #     # defect_faces = [s[1] for s in defect_surfaces if s[0] == 2] ## dimension 2 surfaces only
            #     # distf = gmsh.model.mesh.field.add("Distance")
            #     # gmsh.model.mesh.field.setNumbers(distf, "FacesList", defect_faces)
            #     # gmsh.model.mesh.field.setAsBackgroundMesh(0) ## not sure what this does
            #     # tet_type = 4  # tetrahedra
            #     # _, elem_nodes = gmsh.model.mesh.getElementsByType(tet_type)
            #     # elem_nodes = np.array(elem_nodes).reshape(-1, 4) - 1  # zero-based indexing
            #     # node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            #     # coords = node_coords.reshape(-1, 3)
            #     # barycenters = coords[elem_nodes].mean(axis=1)
            #     # distances = gmsh.model.mesh.field.evaluate(distf, barycenters.flatten().tolist())
            #     # defect_cells = np.array([1 if d < self.h/10 else 0 for d in distances])
            #     # print(f"Marked {defect_cells.sum()} / {len(defect_cells)} cells inside defect.")
            #     # defect_cell_idx = np.nonzero(defect_cells)[0]
            #     # entities_to_mark = [(3, int(elem)) for elem in defect_cell_idx.tolist()]
            #     # defect_marker = gmsh.model.addPhysicalGroup(self.tdim, entities_to_mark)
            #     #===============================================================
            #      
            #     gmsh.model.remove() ## Removes the activate model. 
            #     gmsh.model.setCurrent('The Model')
            #     defect_marker = gmsh.model.addPhysicalGroup(self.tdim, [])
            #     gmsh.model.mesh.setPhysicalGroup(self.tdim, defect_marker, inside_tags.tolist())
            #===================================================================
                
            if(viewGMSH):
                print(mat_marker, domain_marker, pml_marker, pec_surface_marker, farfield_surface_marker, antenna_surface_markers)
                print(matDimTags, domainDimTags, pmlDimTags, pec_surface, farfield_surface, antenna_surface)
                gmsh.fltk.run() ## gives a PETSc error when run in a spack installation
                exit()
            
        else: ## some data is also needed for subordinate processes
            mat_marker = None
            defect_marker = None
            defect_marker2 = None
            domain_marker = None
            pml_marker = None
            pec_surface_marker = None
            antenna_surface_markers = None
            farfield_surface_marker = None
        self.mat_marker = self.comm.bcast(mat_marker, root=self.model_rank)
        self.defect_marker = self.comm.bcast(defect_marker, root=self.model_rank)
        self.defect_marker2 = self.comm.bcast(defect_marker2, root=self.model_rank)
        self.domain_marker = self.comm.bcast(domain_marker, root=self.model_rank)
        self.pml_marker = self.comm.bcast(pml_marker, root=self.model_rank)
        self.pec_surface_marker = self.comm.bcast(pec_surface_marker, root=self.model_rank)
        self.antenna_surface_markers = self.comm.bcast(antenna_surface_markers, root=self.model_rank)
        self.farfield_surface_marker = self.comm.bcast(farfield_surface_marker, root=self.model_rank)
        gmsh.model.mesh.setOrder(self.order) ## It seems I get worse results when using setting the order before generating the mesh... for higher degree FEM elements, need to set this to the degree?
        #self.mesh, self.subdomains, self.boundaries = dolfinx.io.gmsh.model_to_mesh(gmsh.model, comm=self.comm, rank=self.model_rank, gdim=self.tdim, partitioner=dolfinx.mesh.create_cell_partitioner(dolfinx.cpp.mesh.GhostMode.shared_facet))
        self.meshData = dolfinx.io.gmsh.model_to_mesh(gmsh.model, comm=self.comm, rank=self.model_rank, gdim=self.tdim, partitioner=dolfinx.mesh.create_cell_partitioner(dolfinx.cpp.mesh.GhostMode.shared_facet))
        self.mesh = self.meshData.mesh
        ## self.mesh.facet_tags was previously boundaries, self.mesh.cell_tags was previously subdomains
        gmsh.finalize()
        
        self.ncells = self.mesh.topology.index_map(self.mesh.topology.dim).size_global ## all cells, rather than only those in this MPI process
        self.meshingTime = timer() - t1 ## Time it took to make the mesh. Given to mem-time estimator
        if(self.verbosity > 0 and self.comm.rank == self.model_rank): ## start by giving some estimated calculation times/memory costs
            nloc = self.mesh.topology.index_map(self.mesh.topology.dim).size_local
            print(f'Mesh generated in {self.meshingTime:.2e} s - {self.ncells} global cells, {nloc} local cells')
            sys.stdout.flush()
            
    def plotMeshPartition(self):
        '''
        Plots mesh partitions - only works for order 1 meshes currently
        '''
        import pyvista ## can I just import this here so it doesn't need to be installed on the cluster?
        
        V = dolfinx.fem.functionspace(self.mesh, ('CG', 1))
        u = dolfinx.fem.Function(V)
        u.interpolate(lambda x: np.ones(x.shape[1])*self.comm.rank)
        self.mesh.topology.create_connectivity(self.fdim, 0)
        cells, cell_types, x = dolfinx.plot.vtk_mesh(self.mesh, self.tdim)
        grid = pyvista.UnstructuredGrid(cells, cell_types, x)
        grid["rank"] = np.real(u.x.array)
        grids = self.comm.gather(grid, root=self.model_rank)
        if self.comm.rank == self.model_rank:
            plotter = pyvista.Plotter() ## first plot the 3D mesh
            
            #===================================================================
            # for g in grids:
            #     plotter.add_mesh(g, show_edges=True)
            # plotter.view_xy()
            # plotter.add_axes()
            # plotter.show()
            # plotter.clear() ## then plot orthogonal slices
            #===================================================================
            
            for g in grids:
                slices = g.slice_orthogonal()
                plotter.add_mesh(slices, show_edges=True)
            plotter.add_axes()
            plotter.show()