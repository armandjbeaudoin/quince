from dolfin import *
import numpy as np
from petsc4py import PETSc

class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

def build_nullspace(V, x):
    """Function to build null space for 3D elasticity"""

    # Create list of vectors for null space
    nullspace_basis = [x.copy() for i in range(6)]

    # Build translational null space basis
    V.sub(0).dofmap().set(nullspace_basis[0], 1.0);
    V.sub(1).dofmap().set(nullspace_basis[1], 1.0);
    V.sub(2).dofmap().set(nullspace_basis[2], 1.0);

    # Build rotational null space basis
    V.sub(0).set_x(nullspace_basis[3], -1.0, 1);
    V.sub(1).set_x(nullspace_basis[3],  1.0, 0);
    V.sub(0).set_x(nullspace_basis[4],  1.0, 2);
    V.sub(2).set_x(nullspace_basis[4], -1.0, 0);
    V.sub(2).set_x(nullspace_basis[5],  1.0, 1);
    V.sub(1).set_x(nullspace_basis[5], -1.0, 2);

    for x in nullspace_basis:
        x.apply("insert")

    # Create vector space basis and orthogonalize
    basis = VectorSpaceBasis(nullspace_basis)
    basis.orthonormalize()

    return basis

# Class for equilibrium solution of polycrystal    
    
class elasGrains:
    """
    Solve incompatibility problem for post-processing HEDM results 
    using FEniCS
    """

    tol = 1E-9

    def __init__(self,mesh_file,rotations_file,strains_file):

        global dof_min, dof_max
        
        parameters["linear_algebra_backend"] = "PETSc"
        
        self.mesh = Mesh(mesh_file + '.xml')
        self.grains = MeshFunction('size_t',self.mesh,
                                   mesh_file + '_physical_region.xml')
        
        # Function spaces        
        self.V = VectorFunctionSpace(self.mesh, 'P', 1)
        self.V0 = TensorFunctionSpace(self.mesh, 'DG', 0)
        self.TFS = TensorFunctionSpace(self.mesh, 'DG', 0)
        self.VFS = VectorFunctionSpace(self.mesh, 'DG', 0)        
        self.I_TFS_1 = TensorFunctionSpace(self.mesh, 'CG', 1)  # Used in grad z problem   
        
        # mesh dimensions
        self.dim = self.V.dim()
        self.N = self.mesh.geometry().dim()

        dof_coordinates = self.V.tabulate_dof_coordinates()                      
        dof_coordinates.resize((self.dim, self.N))   

        self.dof_min = dof_coordinates.min(axis=0)
        self.dof_max = dof_coordinates.max(axis=0)
        print(self.dof_min)
        print(self.dof_max)
        
        # Set up grain orientation
        self.rots = np.loadtxt(rotations_file)
        self.orient  = Function(self.V0)

        # Vectorized version, used for processing averages/differences
        self.cell_num = np.arange(len(self.grains.array()))
        self.subdomain_num = self.grains.array()[:] - 1
        
        for n in range(9):
            cell_num_list = list((9*self.cell_num)+n)
            self.orient.vector()[cell_num_list] = self.rots[self.subdomain_num,n]

        # Strains from hexrd
        self.exp_strn = np.loadtxt(strains_file)

        self.sim_avg = np.zeros((self.grains.array().max(),9))
        self.ref_strain = np.zeros( (len(self.grains.array()),9) )

        self.dVol = np.fromiter( (c.volume() for c in cells(self.mesh)), float, count=self.mesh.num_cells() )
        self.dVol = self.dVol / self.dVol.sum()
                
        # For difference between lattice strain and experimental average
        self.strain_diff_1 = Function(self.VFS)
        self.strain_diff_2 = Function(self.VFS)
        self.strain_diff_3 = Function(self.VFS)
        
        # To reconstruct tensor from (three) solutions to incompatibility problem
        self.x_id = Expression(("1.0", "0.0", "0.0"), degree=1)
        self.y_id = Expression(("0.0", "1.0", "0.0"), degree=1)
        self.z_id = Expression(("0.0", "0.0", "1.0"), degree=1)

    # Helper functions for rotations and symmetric tensors (Don Boyce)
    
    def to6vector(self,w3x3):
        # return 6-vector form of 3x3 matrix
        return as_vector([w3x3[0,0], w3x3[1,1], w3x3[2,2],
                          w3x3[1,2], w3x3[2,0], w3x3[0,1]])

    def totensor(self,w6):
        # reconstruct tensor from 6-vector
        return as_matrix([[w6[0], w6[5], w6[4]],
                          [w6[5], w6[1], w6[3]],
                          [w6[4], w6[3], w6[2]]])

    def tocrystal(self,w3x3):
        return self.orient.T*w3x3*self.orient

    def tosample(self,w3x3):
        return self.orient*w3x3*self.orient.T

    def sigc6(self,w):
        # for w a 3d vector field
        return dot(self.estf, self.to6vector(self.tocrystal(sym(grad(w)))))

    def sigc3x3(self,w):
        # for w a 3d vector field (displacement)
        return self.totensor(self.sigc6(w))

    def sigs3x3(self,w):
        # stress in sample frame from displacement
        return self.tosample(self.sigc3x3(w))

    # Factor of 2, following Boyce; see elasticity3d.cpp (parameter VALFAC)

    def Chcp(self,c11, c33, c44, c12, c13):
        c = as_vector( [c11, c12, c13, c33, c44, (c11-c12)/2.0] )
        return as_matrix( [[c[0], c[1], c[2], 0,    0,    0],
                           [c[1], c[0], c[2], 0,    0,    0],
                           [c[2], c[2], c[3], 0,    0,    0],
                           [0,    0,    0,    2*c[4], 0,    0],
                           [0,    0,    0,    0,    2*c[4], 0],
                           [0,    0,    0,    0,    0,    2*c[5]]] )

    def Ccubic(self,c11, c12, c44):
        c = as_vector( [c11, c12, c44] )

        return as_matrix( [[c[0], c[1], c[1],      0,      0,      0],
                           [c[1], c[0], c[1],      0,      0,      0],
                           [c[1], c[1], c[0],      0,      0,      0],
                           [0,    0,    0,    2*c[2],      0,      0],
                           [0,    0,    0,         0, 2*c[2],      0],
                           [0,    0,    0,         0,      0,  2*c[2]]] )

    # To derive stress from strain tensor
    def sigc6_e(self,eps):
        # For a strain tensor
        return dot(self.estf, self.to6vector(self.tocrystal(eps)) )

    def sigs_e(self,eps):
        return self.tosample(self.totensor(self.sigc6_e(eps)))
            
    def X_0(self,u,v,w):
        return outer(self.x_id,u) + outer(self.y_id,v) + outer(self.z_id,w)

    def sym_dev(self,U):
        E = sym(U)
        return E - ( (1./3)*tr(E)*Identity(3) )

    def applyBC(self,bc_list=None):
        
        self.bc_elas = bc_list
        
    def elasticity_problem(self,reuse_PC=False, rtol=1e-8, atol=1e-12):
        """Setup the elasticity solver.
        
        The petsc_amg preconditioner is used, with code taken from 
        the (undocumented) FEniCS example demo_elasticity.py
        
        Keyword Arguments:
        reuse_PC -- reuse the preconditioner (default False)
        """
        
        self.u = TrialFunction(self.V)
        self.d = self.u.geometric_dimension()  # space dimension
        self.v = TestFunction(self.V)

        self.L_elas = dot(Constant((0,0,0)),self.v)*ds

        # Create PETSC smoothed aggregation AMG preconditioner 
        self.pc_Eq = PETScPreconditioner("petsc_amg")

        # Use Chebyshev smoothing for multigrid
        PETScOptions.set("mg_levels_ksp_type", "chebyshev")
        PETScOptions.set("mg_levels_pc_type", "jacobi")

        # Improve estimate of eigenvalues for Chebyshev smoothing
        PETScOptions.set("mg_levels_esteig_ksp_type", "cg")
        PETScOptions.set("mg_levels_ksp_chebyshev_esteig_steps", 50)

        # Create CG Krylov solver and turn convergence monitoring on
        self.elasticity_solver = PETScKrylovSolver("cg", self.pc_Eq)
        self.elasticity_solver.parameters["monitor_convergence"] = True
        self.elasticity_solver.parameters["relative_tolerance"] = rtol
        self.elasticity_solver.parameters["absolute_tolerance"] = atol
        
        if reuse_PC:
            self.elasticity_solver.set_reuse_preconditioner(True)
            
        # For the solution
        self.ue = Function(self.V)
#         self.Ue = self.ue.vector()
        
        # Create near null space basis (required for smoothed aggregation
        # AMG). The solution vector is passed so that it can be copied to
        # generate compatible vectors for the nullspace.
        self.null_space = build_nullspace(self.V, self.ue.vector())
        print(self.null_space)

        
    def solve_elas(self,x,E_p=None):
        """Solve the elasticity equilibrium problem.
        
        Keyword Arguments:
        E_p -- plastic distortion to be included in RHS (default None)
        
        Returns:
        res -- the residual error between experimental and simulated grain averages
        """
        
        if x['Crystal_Structure'] == "Cubic":
            self.estf = self.Ccubic( x['Stiffness'][0], x['Stiffness'][1], x['Stiffness'][2] )

        elif x['Crystal_Structure'] == "HCP":
            self.estf = self.Chcp( x['Stiffness'][0], x['Stiffness'][1], x['Stiffness'][2], x['Stiffness'][3], x['Stiffness'][4] )

        # Update orientation
        for n in range(9):
            cell_num_list = list((9*self.cell_num)+n)
            self.orient.vector()[cell_num_list] = self.rots[self.subdomain_num,n]
            
        self.a = inner(self.sigs3x3(self.u), sym(grad(self.v)))*dx
        
        if E_p:
            # Note use of sym(), assuming E_p to be the \chi field
            L_elas_rhs = self.L_elas + inner(self.sigs_e(sym(E_p)), sym(grad(self.v)))*dx
        else:
            L_elas_rhs = self.L_elas            

        self.A_elas, self.b_elas = assemble_system(self.a, L_elas_rhs, self.bc_elas)       
        
        # Attach near nullspace to matrix
        as_backend_type(self.A_elas).set_near_nullspace(self.null_space)

        # Set matrix operator
        self.elasticity_solver.set_operator(self.A_elas);

        # Compute solution
        self.elasticity_solver.solve(self.ue.vector(), self.b_elas);
        
        if E_p:
            self.Ue_sym = project( sym(grad(self.ue) - E_p), self.TFS, solver_type="cg", preconditioner_type="ilu")
        else:
            self.Ue_sym = project( sym(grad(self.ue)), self.TFS, solver_type="cg", preconditioner_type="ilu")
                        
        self.sim_strn = np.reshape(self.Ue_sym.vector().get_local(),(len(self.grains.array()),9))

        for grain_no in range(self.grains.array().max()):
            # Grain numbering is 1 index origin
            cell_subset = self.grains.array()==(grain_no+1)
            if np.any(cell_subset):
                self.sim_avg[grain_no,:] = np.average(self.sim_strn[cell_subset,:],
                                                 axis=0,weights=self.dVol[cell_subset]) 
        
        deps = self.exp_strn - self.sim_avg
        resid = np.linalg.norm(deps.ravel())
        print(resid) #,self.its)
        return resid

    
    def incompatibility_problem(self,reuse_PC=False, rtol=1e-8, atol=1e-12):
        """Setup the incompatibility solver.
        
        Keyword Arguments:
        reuse_PC -- reuse the preconditioner (default False)
        """

        P1 = VectorFunctionSpace(self.mesh, 'CG', 1)
        self.PN = FunctionSpace(self.mesh, "Nedelec 1st kind H(curl)", 1)

        # Define test and trial functions
        self.inc_v0 = TestFunction(self.PN)
        u0 = TrialFunction(self.PN)

        self.T1 = Function(self.PN)        # Solution for the curl curl problem
        self.T2 = Function(self.PN)        # Solution for the curl curl problem
        self.T3 = Function(self.PN)        # Solution for the curl curl problem

        # Boundary condition
        zero = Expression(("0.0", "0.0", "0.0"), degree=1)
        self.bc_X = DirichletBC(self.PN, zero, DirichletBoundary())

        # LHS
        self.a_X = inner(curl(u0), curl(self.inc_v0))*dx
        
        # Create PETSc Krylov solver (from petsc4py)
        self.ksp_X = PETSc.KSP()
        self.ksp_X.create(PETSc.COMM_WORLD)
            
        # Set the Krylov solver type and set tolerances
        self.ksp_X.setType("cg")
#         self.ksp_X.setTolerances(rtol=1.0e-6, atol=1.0e-10, divtol=1.0e10, max_it=50)
        self.ksp_X.setTolerances(rtol=rtol, atol=atol, divtol=1.0e10, max_it=50)
        
        # Get the preconditioner and set type (HYPRE AMS)
        self.pc_X = self.ksp_X.getPC()
        self.pc_X.setType("hypre")
        self.pc_X.setHYPREType("ams")

        # Build discrete gradient
        PL = FunctionSpace(self.mesh, "Lagrange", 1)
        G = DiscreteOperators.build_gradient(self.PN, PL)

        # Attach discrete gradient to preconditioner
        self.pc_X.setHYPREDiscreteGradient(as_backend_type(G).mat())

        # Build constants basis for the Nedelec space
        constants = [Function(self.PN) for i in range(3)]
        for i, c in enumerate(constants):
            direction = [1.0 if i == j else 0.0 for j in range(3)]
            c.interpolate(Constant(direction))

        # Inform preconditioner of constants in the Nedelec space
        cvecs = [as_backend_type(constant.vector()).vec() for constant in constants]
        self.pc_X.setHYPRESetEdgeConstantVectors(cvecs[0], cvecs[1], cvecs[2])

        # no 'mass' term)
        self.pc_X.setHYPRESetBetaPoissonMatrix(None)

        # preconditioner does not change
        if reuse_PC:
            self.pc_X.setReusePreconditioner(True)
            
        # Set options prefix
        self.ksp_X.setOptionsPrefix("inc_")

        # Turn on monitoring of residual
        self.opts = PETSc.Options()
        self.opts.setValue("inc_ksp_monitor_true_residual", None)
        
        # Tolerances are set above, could be modified using inc_ prefix
#         self.opts.setValue("inc_ksp_rtol", 1e-10)
#         self.opts.setValue("inc_ksp_atol", 1e-16)
        
        self.pc_X.setOptionsPrefix("inc_")
        self.pc_X.setFromOptions()

        # Solve eddy currents equation (using potential T)
        self.ksp_X.setFromOptions()
    
    def incompatibility_solve_cg(self, useAMS=True):
        """Solve the incompatibility problem.
        
        Keyword Arguments:
        useAMS -- use the HYPRE AMS preconditioner (default True)
                  [alternative is jacobi preconditioner
        """
        
        zero = Expression(("0.0", "0.0", "0.0"), degree=1)
        bc = DirichletBC(self.PN, zero, DirichletBoundary())
        
        T1 = Function(self.PN)        # Solution for the curl curl problem
        T2 = Function(self.PN)        # Solution for the curl curl problem
        T3 = Function(self.PN)        # Solution for the curl curl problem

        if useAMS:
            
            # Set operator for the linear solver
            L_X = inner(self.strain_diff_1, curl(self.inc_v0))*dx
            A_X, b_X = assemble_system(self.a_X, L_X, bc)
            self.ksp_X.setOperators(as_backend_type(A_X).mat())
            self.ksp_X.solve(as_backend_type(b_X).vec(), as_backend_type(T1.vector()).vec())

            # Show linear solver details
            self.ksp_X.view()

            # Solve 2nd system
            L_X = inner(self.strain_diff_2, curl(self.inc_v0))*dx
            A_X, b_X = assemble_system(self.a_X, L_X, bc)
            self.ksp_X.setOperators(as_backend_type(A_X).mat())
            self.ksp_X.solve(as_backend_type(b_X).vec(), as_backend_type(T2.vector()).vec())

            # Solve 3nd system
            L_X = inner(self.strain_diff_3, curl(self.inc_v0))*dx
            A_X, b_X= assemble_system(self.a_X, L_X, bc)
            self.ksp_X.setOperators(as_backend_type(A_X).mat())
            self.ksp_X.solve(as_backend_type(b_X).vec(), as_backend_type(T3.vector()).vec())
            
        else:

            ### vanilla CG works with potential as RHS

            L_X = inner(self.strain_diff_1, curl(self.inc_v0))*dx
            solve(self.a_X == L_X, T1, bc, 
                  solver_parameters={'linear_solver': 'cg', 'preconditioner': 'jacobi'}) 

            L_X = inner(self.strain_diff_2, curl(self.inc_v0))*dx
            solve(self.a_X == L_X, T2, bc, 
                  solver_parameters={'linear_solver': 'cg', 'preconditioner': 'jacobi'})  

            L_X = inner(self.strain_diff_3, curl(self.inc_v0))*dx
            solve(self.a_X == L_X, T3, bc, 
                  solver_parameters={'linear_solver': 'cg', 'preconditioner': 'jacobi'})

        return project( self.X_0(curl(T1),curl(T2),curl(T3)), 
                        self.TFS, solver_type="cg", preconditioner_type="ilu")
