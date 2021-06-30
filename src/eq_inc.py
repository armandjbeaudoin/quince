#!/usr/bin/env python
# coding: utf-8

# ### Process single scan
# 

# In[1]:


# add parent directory to path
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

    
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
    
from src.incompatibility_FEA import *


# ### Define input files & set up the simulation

# In[2]:


# x_0 is the values of the stiffness tensor #MPa
# Crystal_Structure = 'HCP' order of stiffness values Stiffness: np.array((c11, c33, c44, c12, c13))
# Crystal_Structure = 'Cubic' order of stiffness values Stiffness: np.array((c11, c12, c44))

Crystal_Structure = 'HCP'
#Stiffness =  np.array(( 174800.0, 108900.0, 106700.0 ))
Stiffness = np.array((162400.0, 180700.0, 46700.0, 92000.0, 69000.0))
x_0 = {"Crystal_Structure": Crystal_Structure, "Stiffness": Stiffness}

mesh_dir   = '../mesh/'

mesh_name  = 'creep_0020'
data_dir   = '../data/'
output_dir = '../results/'

# Pick single step
step = '0020'

mesh_file      = mesh_dir + mesh_name
rotations_file = data_dir + 'rotations_0020.dat'
strains_file   = data_dir + 'strains_0020.dat'

eg = elasGrains(mesh_file,rotations_file,strains_file)


# ### Prescribe the boundary conditions for the equilibrium problem

# In[3]:


# Minimal "pointwise" boundary conditions
def boundary_0_0(x, on_boundary):
    tol = 1E-9
    return  (np.abs(x[0]-eg.dof_min[0]) < tol) and             (np.abs(x[1]-eg.dof_min[1]) < tol) and             (np.abs(x[2]-eg.dof_min[2]) < tol)

def boundary_1_0(x, on_boundary):
    tol = 1E-9
    return  (np.abs(x[0]-eg.dof_max[0]) < tol) and             (np.abs(x[1]-eg.dof_min[1]) < tol) and             (np.abs(x[2]-eg.dof_min[2]) < tol)    

def boundary_0_1(x, on_boundary):
    tol = 1E-9
    return  (np.abs(x[0]-eg.dof_min[0]) < tol) and             (np.abs(x[1]-eg.dof_min[1]) < tol) and             (np.abs(x[2]-eg.dof_max[2]) < tol)    

def boundary_lower(x, on_boundary):
    tol = 1E-9
    return (np.abs(x[1]-eg.dof_min[1]) < tol)

def boundary_upper(x, on_boundary):
    tol = 1E-9
    return (np.abs(x[1]-eg.dof_max[1]) < tol)

bc1  = DirichletBC(eg.V, Constant((0, 0, 0)), boundary_0_0, method="pointwise")
bc2a = DirichletBC(eg.V.sub(1), Constant(0), boundary_lower)
bc2b = DirichletBC(eg.V.sub(2), Constant(0), boundary_1_0, method="pointwise")
bc4  = DirichletBC(eg.V.sub(1), Constant(0.00405), boundary_upper)
eg.applyBC( [bc1, bc2a, bc2b, bc4] )


# ### Instantiate solution procedures for both equilibrium & incompatibility problems

# In[4]:


get_ipython().run_line_magic('time', 'eg.elasticity_problem(reuse_PC=True)')


# In[5]:


get_ipython().run_line_magic('time', 'eg.incompatibility_problem(reuse_PC=True)')


# ### Iterative loop

# In[6]:


# Convergence is improved if solution from a prior time step exists

# if int(step) > 0:
#     last_step = '%04d' % (int(step)-1)
#     X_filename = output_dir + "X_" + last_step + ".xdmf"
#     print('Taking initial X from checkpoint ' + X_filename)
#     fFile = XDMFFile(X_filename)
# #     fFile.read_checkpoint(ue,"ue")
#     X_i = Function(eg.TFS)
#     fFile.read_checkpoint(X_i,"X")
#     fFile.close()

#     X_init = X_i
# else:
#     X_init = None

# No prior solution is available
X_init = None


print('First equilibrium solve')
get_ipython().run_line_magic('time', 'rsd = eg.solve_elas(x_0)')

# save the strain from the original problem for development
eg.ref_strn = np.copy(eg.sim_strn)
eg.ref_avg = np.copy(eg.sim_avg)

### Development: will want to develop strain diff in routine contained in class ###

for n in range(3):
    cell_num_list = list((3*eg.cell_num)+n)
    eg.strain_diff_1.vector()[cell_num_list] =         eg.exp_strn[eg.subdomain_num,n]
    eg.strain_diff_2.vector()[cell_num_list] =         eg.exp_strn[eg.subdomain_num,3+n]
    eg.strain_diff_3.vector()[cell_num_list] =         eg.exp_strn[eg.subdomain_num,6+n]

if X_init:
    eg.X = X_init
else:
    print('First incompatibility solve')
    get_ipython().run_line_magic('time', 'eg.X = eg.incompatibility_solve_cg()')

e_inc_elas = project( sym(-eg.X), eg.TFS, solver_type="cg", preconditioner_type="ilu")

print('Second equilibrium solve')
get_ipython().run_line_magic('time', 'res = eg.solve_elas(x_0,E_p=e_inc_elas)')

# Cell to develop average of the latest simulation strain
s_avg = np.zeros((eg.grains.array().max(),9))
s_check = np.zeros((eg.grains.array().max(),9))

# Iterative loop 
last_res = 1e6
for nn in range(24):

    for grain_no in range(eg.grains.array().max()):
        # Grain numbering is 1 index origin
        cell_subset = eg.grains.array()==(grain_no+1)
        if np.any(cell_subset):
            s_avg[grain_no,:] = np.average(eg.sim_strn[cell_subset,:],
                                         axis=0,weights=eg.dVol[cell_subset])
        
    # Set up the potential for the curl problem
    # sim_avg has the volume-weighted average.  Develop a scaled correction,
    # to go in the rhs that has the same grain average as the hedm result

    for n in range(3):
        cell_num_list = list((3*eg.cell_num)+n)
        eg.strain_diff_1.vector()[cell_num_list] =             eg.exp_strn[eg.subdomain_num,n] +             (eg.sim_strn[eg.cell_num,n]-s_avg[eg.subdomain_num,n])
        eg.strain_diff_2.vector()[cell_num_list] =             eg.exp_strn[eg.subdomain_num,3+n] +             (eg.sim_strn[eg.cell_num,3+n]-s_avg[eg.subdomain_num,3+n])                  
        eg.strain_diff_3.vector()[cell_num_list] =             eg.exp_strn[eg.subdomain_num,6+n] +             (eg.sim_strn[eg.cell_num,6+n]-s_avg[eg.subdomain_num,6+n])            

    print('INCOMPATIBILITY')
    get_ipython().run_line_magic('time', 'eg.X = eg.incompatibility_solve_cg()')

#     print('PROJECT')
#     %time e_inc_elas = project( sym(-eg.X), eg.TFS, solver_type="cg", preconditioner_type="ilu")
    
    print('EQUILIBRIUM')
#     %time res = eg.solve_elas(x_0,E_p=e_inc_elas)
    # eg.X is symmetrized in forming RHS
    get_ipython().run_line_magic('time', 'res = eg.solve_elas(x_0,E_p=-eg.X)')
    
    if ( ((np.abs(res-last_res)/res) < 0.005) and (res<last_res) ):
        break

    last_res = res  

# To be deleted
# X_np = np.reshape(eg.X.vector().get_local(),(len(eg.grains.array()),9))


# In[8]:


# compatible part of the elastic distortion

Ue_comp = project( grad(eg.ue), eg.TFS, solver_type="cg", preconditioner_type="ilu")

# If the symmetric part is needed:
# Ue_sym = project( sym(grad(eg.ue)), self.TFS, solver_type="cg", preconditioner_type="ilu")


# In[9]:


# Write out compatible and incompatible distortions

fFile = XDMFFile(output_dir + "Ue_comp_" + step + ".xdmf")
fFile.write_checkpoint (Ue_comp,"Ue_comp")
fFile.close()

fFile = XDMFFile(output_dir + "X_" + step + ".xdmf")
fFile.write_checkpoint(eg.X,"X")
fFile.close()


# ### Comparison of extension strain with & w/o incompatibility

# In[10]:


# before
idx = 4
before = eg.ref_avg[:,idx]-eg.exp_strn[:,idx]
#before = np.linalg.norm(sim_avg-exp_strn,axis=1)
before = before[~np.isnan(before)]
after = eg.sim_avg[:,idx]-eg.exp_strn[:,idx]
# after = np.linalg.norm(sim_avg_cor-exp_strn,axis=1)
after = after[~np.isnan(after)]

bins = np.linspace(-0.002, 0.002, 40)
plt.hist(before, bins, alpha=0.5, label='sim')
plt.hist(after, bins, alpha=0.5, label='cor')
plt.legend(loc='upper right')
plt.xticks(np.arange(-0.002, 0.002, step=0.001))

