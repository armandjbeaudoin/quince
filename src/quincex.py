"""
FEA routines for use with HEDM data.
"""

from contextlib import ExitStack

import numpy as np
import copy
import ufl
# from ufl import sym, grad, dot, as_vector, as_matrix, dx

from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from petsc4py import PETSc

from dolfinx import fem, io, la
#from dolfinx.io import gmshio
from dolfinx import mesh as msh
from dolfinx.cpp.fem.petsc import discrete_gradient, interpolation_matrix

from dolfinx.io import VTXWriter


# There are a few GLOBAL variables

# stiffness matrix
estf = ufl.as_matrix( [[0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0]] )

# mesh dimesions
dof_min, dof_max = np.zeros((3)), np.ones((3))

# Helper functions for rotations and symmetric tensors (Don Boyce)
    
def to6vector(w3x3):
    # return 6-vector form of 3x3 matrix
    return ufl.as_vector([w3x3[0,0], w3x3[1,1], w3x3[2,2],
                          w3x3[1,2], w3x3[2,0], w3x3[0,1]])

def totensor(w6):
    # reconstruct tensor from 6-vector
    return ufl.as_matrix([[w6[0], w6[5], w6[4]],
                          [w6[5], w6[1], w6[3]],
                          [w6[4], w6[3], w6[2]]])

def tocrystal(w3x3):
    return orient.T*w3x3*orient

def tosample(w3x3):
    return orient*w3x3*orient.T

def sigc6(w):
    # for w a 3d vector field
    return ufl.dot(estf, to6vector(tocrystal(ufl.sym(ufl.grad(w)))))

def sigc3x3(w):
    # for w a 3d vector field (displacement)
    return totensor(sigc6(w))

def sigs3x3(w):
    # stress in sample frame from displacement
    return tosample(sigc3x3(w))

# Factor of 2, following Boyce; see elasticity3d.cpp (parameter VALFAC)

def Chcp(c11, c33, c44, c12, c13):
    global estf
    
    c = ufl.as_vector( [c11, c12, c13, c33, c44, (c11-c12)/2.0] )
    
    estf = ufl.as_matrix( [[c[0], c[1], c[2], 0,    0,    0],
                           [c[1], c[0], c[2], 0,    0,    0],
                           [c[2], c[2], c[3], 0,    0,    0],
                           [0,    0,    0,    2*c[4], 0,    0],
                           [0,    0,    0,    0,    2*c[4], 0],
                           [0,    0,    0,    0,    0,    2*c[5]]] )

    return estf

def Ccubic(c11, c12, c44):
    global estf
    
    c = ufl.as_vector( [c11, c12, c44] )

    estf = ufl.as_matrix( [[c[0], c[1], c[1],      0,      0,      0],
                           [c[1], c[0], c[1],      0,      0,      0],
                           [c[1], c[1], c[0],      0,      0,      0],
                           [0,    0,    0,    2*c[2],      0,      0],
                           [0,    0,    0,         0, 2*c[2],      0],
                           [0,    0,    0,         0,      0,  2*c[2]]] )

    return estf
    
    # To derive stress from strain tensor
def sigc6_e(eps):
    # For a strain tensor
    return ufl.dot(estf, to6vector(tocrystal(eps)) )


def sigs_e(eps):
    return tosample(totensor(sigc6_e(eps)))

# Boundary conditions

# Minimal "pointwise" boundary conditions

def boundary_min_max(x_min, x_max):
    """
    Set up global min and max coordinates for use with boundary conditions.
    """
    global dof_min, dof_max
    
    dof_min = copy.deepcopy(x_min)
    dof_max = copy.deepcopy(x_max)
    
    print(dof_min,dof_max)

def boundary_0_0(x):

    global dof_min, dof_max
    
    tol = 1E-9
    return  np.logical_and(
        np.logical_and(
            (np.abs(x[0]-dof_min[0]) < tol), (np.abs(x[1]-dof_min[1]) < tol) ),
            (np.abs(x[2]-dof_min[2]) < tol) )

def boundary_1_0(x):
    
    global dof_min, dof_max
    
    tol = 1E-9
    return  np.logical_and(
        np.logical_and( 
            (np.abs(x[0]-dof_max[0]) < tol), (np.abs(x[1]-dof_min[1]) < tol)),
            (np.abs(x[2]-dof_min[2]) < tol) )    

def boundary_0_1(x):
    
    global dof_min, dof_max
    
    tol = 1E-9
    return  (np.abs(x[0]-dof_min[0]) < tol) and (np.abs(x[1]-dof_min[1]) < tol) and (np.abs(x[2]-dof_max[2]) < tol)    

def boundary_lower(x):
    
    global dof_min, dof_max
    
    tol = 1E-9
    return (np.abs(x[1]-dof_min[1]) < tol)

def boundary_upper(x):
    
    global dof_min, dof_max
    
    tol = 1E-9
    return (np.abs(x[1]-dof_max[1]) < tol)

def boundary_left(x):
    
    global dof_min, dof_max
    
    tol = 1E-9
    return (np.abs(x[0]-dof_min[0]) < tol)

def boundary_right(x):
    
    global dof_min, dof_max
    
    tol = 1E-9
    return (np.abs(x[0]-dof_max[0]) < tol)

def boundary_back(x):
    
    global dof_min, dof_max
    
    tol = 1E-9
    return (np.abs(x[2]-dof_min[2]) < tol)

def boundary_front(x):
    
    global dof_min, dof_max
    
    tol = 1E-9
    return (np.abs(x[2]-dof_max[2]) < tol)

# Utility function, to assign tensor property (typically measured by HEDM) to grains

def _tgrain(tnsr, x):
    """
    Utility function for tprop2grains
    """
    return np.tile(tnsr.reshape(9,1), x.shape[1])

def tprop2grains(tprop,Tspace,cell_tags):
    """
    Assign a tensor property (e.g. rotation measured by HEDM) on grain by grain basis
    
    Follows approach developed by Don Boyce
    
    Arguments:
        tprop - matrix with 3x3 tensor for each grain
        Tspace - dolfinx (discontinuous) tensor function space
        cell_tags - cell tags from mesh to identify grains
        
    Returns:
        grain_t - Function with property assigned to grains
        
    """

    grain_t  = fem.Function(Tspace)
    
    # Assign grain by grain
    for i in range(len(tprop)):
        cells = cell_tags.find(i+1)
        tgrain = lambda x: _tgrain(tprop[i,:], x)
        grain_t.interpolate(tgrain,cells)
        
    return grain_t

def set_orientation(rot):
    global orient
    
    orient = rot
    
    return
    
def build_nullspace(V):
    """Build PETSc nullspace for 3D elasticity"""

    # Create list of vectors for building nullspace
    index_map = V.dofmap.index_map
    bs = V.dofmap.index_map_bs
    ns = [la.create_petsc_vector(index_map, bs) for i in range(6)]
    with ExitStack() as stack:
        vec_local = [stack.enter_context(x.localForm()) for x in ns]
        basis = [np.asarray(x) for x in vec_local]

        # Get dof indices for each subspace (x, y and z dofs)
        dofs = [V.sub(i).dofmap.list.array for i in range(3)]

        # Build the three translational rigid body modes
        for i in range(3):
            basis[i][dofs[i]] = 1.0

        # Build the three rotational rigid body modes
        x = V.tabulate_dof_coordinates()
        dofs_block = V.dofmap.list.array
        x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
        basis[3][dofs[0]] = -x1
        basis[3][dofs[1]] = x0
        basis[4][dofs[0]] = x2
        basis[4][dofs[2]] = -x0
        basis[5][dofs[2]] = x1
        basis[5][dofs[1]] = -x2

    # Orthonormalise the six vectors
    la.orthonormalize(ns)
    assert la.is_orthonormal(ns)

    return PETSc.NullSpace().create(vectors=ns)

# Incompatibility problem

class Incompatibility:
    """
    Project strain onto domain using curl to develop incompatible strain.
    """
    
    def __init__(self,domain,strain,use_solver='ams',view_solver=False):
        
        self.comm = domain.comm
        self.rank = self.comm.Get_rank()

        self.PN = fem.FunctionSpace(domain, ("Nedelec 1st kind H(curl)", 1))
        self.T0 = fem.TensorFunctionSpace(domain, ('DG', 0))
        
        # Define test and trial functions
        self.inc_v0 = ufl.TestFunction(self.PN)
        u0 = ufl.TrialFunction(self.PN)
        
        # Used to reconstruct incompatibility from sub-problems
        self.x_id = fem.Constant(domain, PETSc.ScalarType((1.0, 0.0, 0.0)))
        self.y_id = fem.Constant(domain, PETSc.ScalarType((0.0, 1.0, 0.0)))
        self.z_id = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0, 1.0)))
        
        # Create facet to cell connectivity required to determine boundary facets
        tdim = domain.topology.dim
        fdim = tdim - 1
        domain.topology.create_connectivity(fdim, tdim)
        boundary_facets = msh.exterior_facet_indices(domain.topology)
        
        # Boundary condition for incompatibility problem example:
        # https://fenicsproject.discourse.group/t/time-harmonic-eddy-currents-problem/8279/2

        boundary_dofs_X = fem.locate_dofs_topological(self.PN, fdim, boundary_facets)

        X_d = fem.Function(self.PN)
        X_d.interpolate(lambda x: 0*x)
        self.bc_X = fem.dirichletbc(X_d, boundary_dofs_X)
                
        # Set up the AMS solver, see
        # https://hypre.readthedocs.io/en/latest/solvers-ams.html
        # also, much taken from
        # https://fenicsproject.discourse.group/t/boomeramg-in-dolfinx/7893/4

        self.X_solver = PETSc.KSP().create(MPI.COMM_WORLD)


        # Set options
        opts = PETSc.Options()
        prefix = f"incompatibility_{id(self.X_solver)}"
        print("prefix:",prefix)
        self.X_solver.setOptionsPrefix(prefix)
        option_prefix = self.X_solver.getOptionsPrefix()
        
        # Set solver options
        opts = PETSc.Options()                

        opts[f"{option_prefix}ksp_rtol"] = 1.0e-10  # Poor results with 1e-5
        opts[f"{option_prefix}ksp_atol"] = 1.0e-16

        if use_solver=='ams':
            opts[f"{option_prefix}ksp_type"] = "cg" # "gmres" 
            opts[f"{option_prefix}pc_type"] = "hypre"
            opts[f"{option_prefix}pc_hypre_type"] = "ams" 
    #         opts[f"{option_prefix}pc_hypre_boomeramg_print_statistics"] = 1
    #         opts[f"{option_prefix}pc_hypre_ams_print_level"] = 3
    #         opts[f"{option_prefix}monitor_convergence"] = None

            # Are these next two lines needed????
    #         ksp_X = PETSc.KSP().create(domain.comm)
    #         ksp_X.setType(PETSc.KSP.Type.CG)        

        elif use_solver=='mumps':

            opts[f"{option_prefix}ksp_type"] = "preonly"
            opts[f"{option_prefix}pc_type"] = "lu"
            opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"

        # cg works with gamg with default arguments (uses jacobi and not ilu in subproblems)  
        # see https://fenicsproject.discourse.group/t/boomeramg-in-dolfinx/7893/4
        # for example of setting null space.  Try options from elasticity solve (with jacobi preconditioner)
        elif use_solver=="gamg":
            opts[f"{option_prefix}ksp_type"] = "cg"
            opts[f"{option_prefix}pc_type"] = "gamg"
                        # Use Chebyshev smoothing for multigrid
            opts[f"{option_prefix}mg_levels_ksp_type"] = "chebyshev"
            opts[f"{option_prefix}mg_levels_pc_type"] = "jacobi"

            # Improve estimate of eigenvalues for Chebyshev smoothing
            opts[f"{option_prefix}mg_levels_esteig_ksp_type"] = "cg"
            opts[f"{option_prefix}mg_levels_ksp_chebyshev_esteig_steps"] = 20
            
        # cg works with jacobi (bjacobi uses ilu and doesn't work)
        elif use_solver=="cg":
            opts[f"{option_prefix}ksp_type"] = "cg"
            opts[f"{option_prefix}pc_type"] = "jacobi"
        
        # Default to the cg with jacobi preconditioner
        else:
            opts[f"{option_prefix}ksp_type"] = "cg"
            opts[f"{option_prefix}pc_type"] = "jacobi"
            
        if view_solver:
            opts[f"{option_prefix}ksp_view"] = None
            opts[f"{option_prefix}ksp_monitor_true_residual"] = None
            
        self.X_solver.setConvergenceHistory()
        self.X_solver.setFromOptions()
        
        # LHS
        a_X = ufl.inner(ufl.curl(u0), ufl.curl(self.inc_v0))*ufl.dx

        self.bilinear_form = fem.form(a_X)

        AX = fem.petsc.assemble_matrix(self.bilinear_form, bcs=[self.bc_X])
        AX.assemble()
        
        # see https://fenicsproject.discourse.group/t/boomeramg-in-dolfinx/7893/4 for gamg nullspace with curl-curl            
        if use_solver=='gamg':
            nullspace = PETSc.NullSpace().create(constant=True)
            AX.setNearNullSpace(nullspace)

        self.X_solver.setOperators(AX)

        if use_solver=='ams':
            pc_X = self.X_solver.getPC()

            # Additional things needed by AMS solver
            # Discrete gradient matrix
            # The order of arguments for cpp is reversed order in old FEniCS python call
            # see
            # https://github.com/FEniCS/dolfinx/blob/main/cpp/dolfinx/fem/discreteoperators.h
            # where V0 is the Lagrange space and 
            # https://fenics.readthedocs.io/projects/dolfin/en/2017.2.0/apis/api_fem.html#discreteoperators
            # where V0 is the Nedelec space

            PL = fem.FunctionSpace(domain,("CG", 1))._cpp_object
            G = discrete_gradient(PL, self.PN._cpp_object)
            G.assemble()
            pc_X.setHYPREDiscreteGradient(G)

            # Constant vector fields
            cvec_0 = fem.Function(self.PN)
            cvec_0.interpolate(lambda x: np.vstack((np.ones_like(x[0]),
                                                    np.zeros_like(x[0]),
                                                    np.zeros_like(x[0]))))
            cvec_1 = fem.Function(self.PN)
            cvec_1.interpolate(lambda x: np.vstack((np.zeros_like(x[0]),
                                                    np.ones_like(x[0]),
                                                    np.zeros_like(x[0]))))
            cvec_2 = fem.Function(self.PN)
            cvec_2.interpolate(lambda x: np.vstack((np.zeros_like(x[0]),
                                                    np.zeros_like(x[0]),
                                                    np.ones_like(x[0]))))
            pc_X.setHYPRESetEdgeConstantVectors(cvec_0.vector,
                                                cvec_1.vector,
                                                cvec_2.vector)

            pc_X.setHYPRESetBetaPoissonMatrix(None) #ams            

        self.X  = fem.Function(self.T0)
        
        # self.linear_forms = [ fem.form( ufl.inner(strain[:,i], ufl.curl(self.inc_v0))*ufl.dx ) for i in range(3) ]
        
        self.Xh = [ fem.Function(self.PN) for i in range(3) ]
        
        self.X_0 =  ufl.outer(self.x_id,ufl.curl(self.Xh[0])) + \
            ufl.outer(self.y_id,ufl.curl(self.Xh[1])) + \
            ufl.outer(self.z_id,ufl.curl(self.Xh[2]))

        self.X_expr = fem.Expression(self.X_0,self.T0.element.interpolation_points())

        self.linear_forms = [ fem.form( ufl.inner(strain[:,i], ufl.curl(self.inc_v0))*ufl.dx ) for i in range(3) ]
        
    def solve_curl(self):
        """
        Perform solution of the incompatibility problem.
        """
        
        for i in range(3):
            
            # see https://fenicsproject.discourse.group/t/dirichlet-boundary-conditions-hcurl/9932/4
            # and link in answer to https://github.com/FEniCS/dolfinx/blob/main/python/demo/demo_stokes.py#L460-L466

#             self.comm.Barrier()
#             print(self.rank,i,'Before assemble', flush=True)    
            # bX = fem.petsc.assemble_vector(self.linear_forms[i])
            
            # Alternative.  If this works, maybe re-use vectors?
            bX = fem.petsc.create_vector(self.linear_forms[i])
            with bX.localForm() as loc_b:
                loc_b.set(0)
            fem.petsc.assemble_vector(bX, self.linear_forms[i])            
            
#             self.comm.Barrier()
#             print(self.rank,i,'Before lifting', flush=True) 
            fem.petsc.apply_lifting(bX, [self.bilinear_form], bcs=[[self.bc_X]])
            
#             self.comm.Barrier()           
#             print(self.rank,i,'Before ghost update', flush=True) 
            bX.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
                        
#             self.comm.Barrier()           
#             print(self.rank,i,'Before petsc.set_bc', flush=True)             
            fem.petsc.set_bc(bX, [self.bc_X])         

#             print(self.rank,i,'Before solver', flush=True)              
#             self.comm.Barrier()
            self.X_solver.solve(bX, self.Xh[i].vector)
            
#             print(self.rank,i,'Before scatter', flush=True)              
#             self.comm.Barrier()
            self.Xh[i].x.scatter_forward()
            
#             print(self.rank,i,'After scatter', flush=True)              
#             self.comm.Barrier()

        self.X.interpolate(self.X_expr)
        
        return

    def write_X(self,domain,X,filename):
        """
        Write incompatibility result
        """
            
        # For writing incompatibility as DG1
        T_DG1 = fem.TensorFunctionSpace(domain, ('DG', 1))
        X_DG1  = fem.Function(T_DG1)
        self.X_expr_DG1 = fem.Expression(self.X_0,T_DG1.element.interpolation_points())
        
        # X_DG1.interpolate(self.X_expr_DG1)
        
        X_DG1.interpolate(X)

        vtx_X = VTXWriter(domain.comm, filename, [X_DG1])
        vtx_X.write(0.)
        vtx_X.close()

# from https://fenicsproject.discourse.group/t/setting-vector-component-wise-dirichlet-bcs/9252
class displacement_expression:
    def __init__(self):
        self.displacement = 0.0
        
    def eval(self, x):
        # linearly incremented displacement
        return np.full(x.shape[1], self.displacement)
        
# Elasticity problem
class Elasticity:
    """
    Solve elasticity problem
    
    The solution is maintained the the Function uh
    """
    
    def __init__(self,domain,Up=None,use_MUMPS=False):

        self.V  = fem.VectorFunctionSpace(domain, ("CG", 1))
        
        # Constraints on two points, at lower corner & along extreme x-axis
        u_D = np.array([0,0,0], dtype=ScalarType)

        tdim = domain.topology.dim
        fdim = tdim - 1
        
        ubc1 = fem.dirichletbc(u_D, fem.locate_dofs_geometrical(self.V, boundary_0_0), self.V)

        Vz, u_z_dofs = self.V.sub(2).collapse()
        zD = fem.Function(Vz)
        zD.interpolate(lambda x: 0.0*x[0])
        z_dof = fem.locate_dofs_geometrical((self.V.sub(2),Vz),boundary_1_0)
        ubc2 = fem.dirichletbc(zD, z_dof, self.V.sub(2))

        # surface constraints on fixed lower surface and displacement of upper surface
        Vy, u_y_dofs = self.V.sub(1).collapse() 
        
        # lower_facets = msh.locate_entities_boundary(domain, fdim, boundary_lower)
        # ubc3 = fem.dirichletbc(ScalarType(0),
        #                        fem.locate_dofs_topological(self.V.sub(1), fdim, lower_facets), self.V.sub(1))
        lD = fem.Function(Vy)
        lD.interpolate(lambda x: 0.0*x[0])
        lower_y_dofs = fem.locate_dofs_geometrical((self.V.sub(1),Vy), boundary_lower)
        ubc3 = fem.dirichletbc(lD, lower_y_dofs, self.V.sub(1))
        
        self.applied_displacement_expr = displacement_expression()
        self.applied_displacement_expr.displacement = 0.0

        self.applied_displacement = fem.Function(Vy)
        self.applied_displacement.interpolate(self.applied_displacement_expr.eval)
        
        upper_y_dofs = fem.locate_dofs_geometrical((self.V.sub(1),Vy), boundary_upper)
        ubc4 = fem.dirichletbc(self.applied_displacement, upper_y_dofs, self.V.sub(1))
        
#         upper_facets = msh.locate_entities_boundary(domain, fdim, boundary_upper)
#         bc4 = fem.dirichletbc(ScalarType(0.00405), 
#                               fem.locate_dofs_topological(self.V.sub(1), fdim, upper_facets), self.V.sub(1))

        self.bcs = [ubc1, ubc2, ubc3, ubc4]

        # Traction
        self.T = fem.Constant(domain, ScalarType((0, 0, 0)))

        # Integral over boundary of the mesh

        self.ds = ufl.Measure("ds", domain=domain)

        # Variational formulation for the elasticity problem

        self.u = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)

        self.f = fem.Constant(domain, ScalarType((0, 0, 0)))
        a = ufl.inner(sigs3x3(self.u), ufl.sym(ufl.grad(self.v)))*ufl.dx
        
        self.uh = fem.Function(self.V)
        
        self.lhs_form = fem.form(a)
        A_ela = fem.petsc.assemble_matrix(self.lhs_form, bcs=self.bcs)
        A_ela.assemble()

        
        if use_MUMPS:
            # MUMPS solver (should put in option to choose solver type

            self.solver_E = PETSc.KSP().create(domain.comm)
            self.solver_E.setType(PETSc.KSP.Type.PREONLY)
            self.solver_E.getPC().setType(PETSc.PC.Type.LU)

            # Set options
            opts = PETSc.Options()
            prefix = f"elasticity_{id(self.solver_E)}"
            print("prefix:",prefix)
            self.solver_E.setOptionsPrefix(prefix)
            option_prefix = self.solver_E.getOptionsPrefix()
            opts[f"{option_prefix}ksp_type"] = "preonly"
            opts[f"{option_prefix}pc_type"] = "lu"
            opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"

            
        else:
            # AMG solver
            null_space = build_nullspace(self.V)
            A_ela.setNearNullSpace(null_space)

            self.solver_E = PETSc.KSP().create(domain.comm)

            # Set options
            opts = PETSc.Options()
            prefix = f"elasticity_{id(self.solver_E)}"
            print("prefix:",prefix)
            self.solver_E.setOptionsPrefix(prefix)
            option_prefix = self.solver_E.getOptionsPrefix()
            # Set solver options
            opts = PETSc.Options()
            opts[f"{option_prefix}ksp_type"] = "cg"
            opts[f"{option_prefix}ksp_rtol"] = 1.0e-10
            opts[f"{option_prefix}pc_type"] = "gamg"

            # Use Chebyshev smoothing for multigrid
            opts[f"{option_prefix}mg_levels_ksp_type"] = "chebyshev"
            opts[f"{option_prefix}mg_levels_pc_type"] = "jacobi"

            # Improve estimate of eigenvalues for Chebyshev smoothing
            opts[f"{option_prefix}mg_levels_esteig_ksp_type"] = "cg"
            opts[f"{option_prefix}mg_levels_ksp_chebyshev_esteig_steps"] = 20
            
        self.solver_E.setConvergenceHistory()
        self.solver_E.setFromOptions()
        
        # Assuming that the LHS is constant over time), just solving elasticity
        # Also, RHS form doesn't change (so, define it here).
        # see https://fenicsproject.discourse.group/t/problems-with-custom-newton-solver-for-elasticity/9250/2
        # to update A within loop
        
        self.solver_E.setOperators(A_ela)
        
        if Up:
            # Note use of sym(), assuming E_p to be the \chi field
            self.L = ufl.dot(self.f, self.v) * ufl.dx + ufl.dot(self.T, self.v) * self.ds + \
                     ufl.inner(sigs_e(ufl.sym(Up)), ufl.sym(ufl.grad(self.v)))*ufl.dx
        else:
            self.L = ufl.dot(self.f, self.v) * ufl.dx + ufl.dot(self.T, self.v) * self.ds 
            
        self.rhs_form = fem.form(self.L)        
        
    def solve_elasticity(self,y_disp):
        """
        Solve the elasticity problem
        
        Input arguments
            y_disp: displacement applied to the upper (+y) surface
            Up - the plastic "strain"
            
        Returns
            uf - the displacement
        """
        
        self.applied_displacement_expr.displacement = y_disp
        self.applied_displacement.interpolate(self.applied_displacement_expr.eval)
        
        # see https://fenicsproject.discourse.group/t/dirichlet-boundary-conditions-hcurl/9932/4
        # and link in answer to https://github.com/FEniCS/dolfinx/blob/main/python/demo/demo_stokes.py#L460-L466
        

        
        b_ela = fem.petsc.assemble_vector(self.rhs_form)
        fem.petsc.apply_lifting(b_ela, [self.lhs_form], bcs=[self.bcs])
        b_ela.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b_ela, self.bcs)
        
        self.solver_E.solve(b_ela, self.uh.vector)
        self.uh.x.scatter_forward()      
                
        return

class Compatibility:
    """
    Solve for the compatible part of deformation
    (the part that may be represented as the gradient of a vector field)
    """
    
    def __init__(self,domain,Up):
        # Define variational problem

        self.V  = fem.VectorFunctionSpace(domain, ("CG", 1))
        z       = ufl.TrialFunction(self.V)
        self.z_ = ufl.TestFunction(self.V)

        # Minimum BC: fix a single point
        u_D = np.array([0,0,0], dtype=ScalarType)

        tdim = domain.topology.dim
        fdim = tdim - 1
        
        zbc1 = fem.dirichletbc(u_D, fem.locate_dofs_geometrical(self.V, boundary_0_0), self.V)
        self.bcs = [zbc1]

        # Assemble the system
        
        a_z = ufl.inner(ufl.grad(z), ufl.grad(self.z_))*ufl.dx 
        
        self.lhs_form = fem.form(a_z)
        A_z = fem.petsc.assemble_matrix(self.lhs_form, bcs=self.bcs)
        A_z.assemble()
        
        # Solver
        self.solver_z = PETSc.KSP().create(domain.comm)

        # Set options
        opts = PETSc.Options()
        prefix = f"gradz_{id(self.solver_z)}"
        print("prefix:",prefix)
        self.solver_z.setOptionsPrefix(prefix)
        option_prefix = self.solver_z.getOptionsPrefix()
        
        # Set solver options
        opts = PETSc.Options()
        opts[f"{option_prefix}ksp_type"] = "cg"
        opts[f"{option_prefix}ksp_rtol"] = 1.0e-8
        # opts[f"{option_prefix}pc_type"] = "jacobi"
        opts[f"{option_prefix}monitor_convergence"] = None

        self.solver_z.setConvergenceHistory()
        self.solver_z.setFromOptions()
        
        self.solver_z.setOperators(A_z)
        
        self.L_z = ufl.inner( Up, ufl.grad(self.z_) )*ufl.dx

        # self.rhs_form = fem.form(self.L_z)
        
        # To hold the solution
        self.ze = fem.Function(self.V)        

    def compatibility_solve(self):
        """
        Solve for compatibile part of the plastic distortion.
        """        

        # TEST
        self.rhs_form = fem.form(self.L_z)
        
        b_z = fem.petsc.assemble_vector(self.rhs_form)
        fem.petsc.apply_lifting(b_z, [self.lhs_form], bcs=[self.bcs])
        b_z.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b_z, self.bcs)
        
        self.solver_z.solve(b_z, self.ze.vector)
        self.ze.x.scatter_forward()   
        

# project function from dolfiny

# def project(v, target_func, bcs=[]):
#     # Ensure we have a mesh and attach to measure
#     V = target_func.function_space
#     dx = ufl.dx(V.mesh)

#     # Define variational problem for projection
#     w = ufl.TestFunction(V)
#     Pv = ufl.TrialFunction(V)
#     a = fem.form(ufl.inner(Pv, w) * dx)
#     L = fem.form(ufl.inner(v, w) * dx)

#     # Assemble linear system
#     A = fem.petsc.assemble_matrix(a, bcs)
#     A.assemble()
#     b = fem.petsc.assemble_vector(L)
#     fem.petsc.apply_lifting(b, [a], [bcs])
#     b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
#     fem.petsc.set_bc(b, bcs)

#     # Solve linear system
#     solver = PETSc.KSP().create(A.getComm())
#     solver.setOperators(A)
#     solver.solve(b, target_func.vector)