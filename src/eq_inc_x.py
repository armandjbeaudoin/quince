#!/usr/bin/env python
# coding: utf-8

# # Introduce incompatible deformation from HEDM into elasticity calculation

# In[ ]:


# Input files:

mesh_file      = "../mesh/creep_0020"
rotations_file = "../data/rotations_0020.dat"
strains_file   = "../data/strains_0020.dat"

# Directory for compiled forms
cache_dir      = "../fenicsx_cache"

# Output file name
output_file_name = '../results/quincex.xdmf'
stress_field_VTX_name = '../results/quince_stress.bp'

y_displacement = 0.00405

# Tolerance on relative strain norm
strn_tolerance = 0.02


# In[ ]:


# add parent directory to path
import copy
import os
import sys
import time

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np

import src.quincex as qx

from petsc4py import PETSc

from mpi4py import MPI

from dolfinx import fem, io
from dolfinx.io import gmshio, VTXWriter
from dolfinx import mesh as msh

import ufl

import matplotlib.pyplot as plt


# In[ ]:


# Elasticity properties

Crystal_Structure = 'HCP'
#Stiffness =  np.array(( 174800.0, 108900.0, 106700.0 ))
Stiffness = np.array((162400.0, 180700.0, 46700.0, 92000.0, 69000.0))
x_0 = {"Crystal_Structure": Crystal_Structure, "Stiffness": Stiffness}

estf = qx.Chcp(x_0['Stiffness'][0],
               x_0['Stiffness'][1],
               x_0['Stiffness'][2],
               x_0['Stiffness'][3],
               x_0['Stiffness'][4] )


# In[ ]:


# Find rank of this process
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


# In[ ]:


# Read mesh from gmsh .msh file, using gmshio
# domain, cell_tags, facet_tags = gmshio.read_from_msh(mesh_file + '.msh', MPI.COMM_WORLD, 0, gdim=3)


# In[ ]:


# Read from .xdmf
f_mesh = io.XDMFFile(MPI.COMM_WORLD, mesh_file + '.xdmf', "r")
domain = f_mesh.read_mesh()
cell_tags = f_mesh.read_meshtags(domain,name='Cell tags')
f_mesh.close()


# In[ ]:


# Create facet to cell connectivity required to determine boundary facets
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = msh.exterior_facet_indices(domain.topology)


# In[ ]:


dof_min, dof_max =  np.zeros((3)), np.zeros((3))
minVal = np.zeros((1))
for i in range( domain.geometry.dim):
    dof_min[i] = domain.geometry.x[:, i].min()
    dof_max[i] = domain.geometry.x[:, i].max()

comm.Barrier()
comm.Allreduce(MPI.IN_PLACE, dof_min, op=MPI.MIN)
comm.Allreduce(MPI.IN_PLACE, dof_max, op=MPI.MAX)
comm.Barrier()
        
print(rank,np.min(domain.geometry.x,axis=0),dof_min)
print(rank,np.max(domain.geometry.x,axis=0),dof_max,flush=True)

qx.boundary_min_max(dof_min,dof_max)


# In[ ]:


n_grains = cell_tags.values.max()


# In[ ]:


#  Get number of grains on each processor
print(rank,'before',n_grains)
n_grains = comm.allreduce( n_grains, op = MPI.MAX)
print(rank,'after',n_grains,flush=True)


# In[ ]:


T0 = fem.TensorFunctionSpace(domain, ('DG', 0))

# Load in rotation file.  Assign rotations to the tensor function "orient"
# using interpolation with lamda function (following Don Boyce)

rots = np.loadtxt(rotations_file)
orient = qx.tprop2grains(rots,T0,cell_tags)
qx.set_orientation(orient)

exp_strn = np.loadtxt(strains_file)
exp_strain  = qx.tprop2grains(exp_strn,T0,cell_tags)


# In[ ]:


# Fields for strain, plastic deformation, and (strain) result from simulation
strain_field = fem.Function(T0)

V_0 = fem.FunctionSpace(domain, ('DG',0))

Up = fem.Function(T0)
Up.interpolate(lambda x: np.zeros((9,x.shape[1])))

sim_strn = fem.Function(T0)
sim_strn.interpolate(lambda x: np.zeros((9,x.shape[1])))


# In[ ]:


# Develop integration measure for grains
dx_grain = []

for i in range(n_grains):
    dx_grain.append( ufl.Measure("dx", domain=domain, subdomain_data=cell_tags, subdomain_id=(i+1)) )


# In[ ]:


# Expression to handle single component of strain tensor

eij = fem.Function(V_0)
eij.interpolate(lambda x: np.zeros((x.shape[1])))

eij_expr = []

for ii in range(3):
    for jj in range(3):
        eij_expr.append( fem.Expression(sim_strn[ii,jj],
                                        V_0.element.interpolation_points()) )


# In[ ]:


# Pre-compile all of the forms for averaging grain values (this takes a while)

# jit_parameters = {"cffi_extra_compile_args": ["-Ofast", "-march=native"], 
#                   "cache_dir": cache_dir, "cffi_libraries": ["m"]}

jit_parameters = {"cffi_extra_compile_args": ["-O2"], 
                  "cache_dir": cache_dir, "cffi_libraries": ["m"]}

grain_forms_compiled = []
start_time = time.process_time()
for i in range(n_grains):
    f = []
    if rank==0 and (i%100)==0:
        print(i,flush=True)

    grain_forms_compiled.append( fem.form(eij*dx_grain[i], jit_options=jit_parameters) )               
            
print(rank,'form compilation',time.process_time() - start_time)


# In[ ]:


vol_grain = np.zeros((n_grains))
v_total = 0.0

eij.interpolate(lambda x: np.ones((x.shape[1])))

for i in range(n_grains):
    vol_grain[i] = comm.allreduce( fem.assemble_scalar(grain_forms_compiled[i]), op = MPI.SUM)
    v_total += vol_grain[i]
    
if rank==0:
    print('Volume is',v_total)


# In[ ]:


sim_avg = np.zeros((n_grains,9))

start_time = time.process_time()

for j in range(9):
        
    eij.interpolate(eij_expr[j])
    
    for i in range(n_grains):

        sim_avg[i,j] = comm.allreduce( \
            fem.assemble_scalar( grain_forms_compiled[i]) / vol_grain[i],
            op = MPI.SUM)

comm.Barrier()
print(rank,'Form Execution',time.process_time() - start_time)


# In[ ]:


# Develop reference solution, with no experimental correction

sim_avg = np.zeros((n_grains,9))

Up.interpolate(lambda x: np.zeros((9,x.shape[1])))

ela = qx.Elasticity(domain,Up=Up)
ela.solve_elasticity(y_displacement)

sim_strn_expr = fem.Expression( ufl.sym(ufl.grad(ela.uh) - Up), T0.element.interpolation_points())

sim_strn.interpolate(sim_strn_expr)

for j in range(9):

    eij.interpolate(eij_expr[j])

    for i in range(n_grains):
        sim_avg[i,j] = comm.allreduce( \
            fem.assemble_scalar( grain_forms_compiled[i]) / vol_grain[i], op = MPI.SUM)

comm.Barrier()

# if (rank==0):
print(rank, 'Experimental strain error',np.linalg.norm(sim_avg[:] - exp_strn[:]))
print(rank, 'Average y displacement is', np.mean(sim_avg[:,4]), flush=True)

ref_avg = copy.deepcopy(sim_avg)


# In[ ]:


# Loop

# Initialize potential for incompatibility with piecewise constant experimental strain field
strain_field.interpolate(exp_strain)
# comm.Barrier()    
# print(rank, 'Before Incompatibility initialize', flush=True)
inc = qx.Incompatibility(domain,strain_field,use_solver='cg',view_solver=True)
# comm.Barrier()    
# print(rank, 'Incompatibility initialize', flush=True)

Up_expr = fem.Expression( -inc.X, T0.element.interpolation_points() )

last_sim_strn = fem.Function(T0)
diff_sim_strn = fem.Function(T0)
sim_strn_mag  = fem.Function(T0)
diff_strn_expr = fem.Expression( (sim_strn-last_sim_strn)*(sim_strn-last_sim_strn), 
                                  T0.element.interpolation_points() )
sim_strn_mag_expr = fem.Expression( sim_strn*sim_strn, T0.element.interpolation_points() )

for nn in range(16):

    # strain_field should be initialized with experimental strain for first iteration

    inc.solve_curl()
    
    Up.interpolate(Up_expr)
    
    ela.solve_elasticity(y_displacement)
    
    last_sim_strn.x.array[:] = sim_strn.x.array[:]
    sim_strn.interpolate(sim_strn_expr)
    
    # Develop average strain from elasticity simulation   
    for j in range(9):

        eij.interpolate(eij_expr[j])

        for i in range(n_grains):
            sim_avg[i,j] = comm.allreduce( \
                fem.assemble_scalar( grain_forms_compiled[i]) / vol_grain[i], op = MPI.SUM)
    
    # Develop the relative error norm as change in strain
    diff_sim_strn.interpolate(diff_strn_expr)
    sim_strn_mag.interpolate(sim_strn_mag_expr)
    
    diff_sim = comm.allreduce(np.sum(diff_sim_strn.x.array), op=MPI.SUM)
    comm.Barrier()
    
    sim_mag = comm.allreduce(np.sum(sim_strn_mag.x.array), op=MPI.SUM)
    comm.Barrier()  
    
    strn_norm = np.sqrt(diff_sim) / np.sqrt(sim_mag)
    
    if (rank==0):
        print(rank, nn,'Experimental strain error',np.linalg.norm(sim_avg[:] - exp_strn[:]), flush=True)
        print(rank, '  Average y displacement is', np.mean(sim_avg[:,4]), flush=True)
        print(rank, '  Relative strain error norm', strn_norm, flush=True)
            
    s_avg  = qx.tprop2grains(sim_avg,T0,cell_tags)
    
    # should be fixed, not re-defining s_avg, to compile only once
    # if nn==0:
    strain_field_expr = fem.Expression( exp_strain + (sim_strn-s_avg), T0.element.interpolation_points() )
        
    strain_field.interpolate( strain_field_expr )
    
    if strn_norm < strn_tolerance:
        break    


# In[ ]:


results_xdmf_file = io.XDMFFile(comm,output_file_name,'w')
results_xdmf_file.write_mesh(domain)

ela.uh.name = 'displacement'
results_xdmf_file.write_function(ela.uh, 0.0)

strain_field.name = 'strain'
results_xdmf_file.write_function(strain_field, 0.0)

sigma  = fem.Function(T0)
sigma_expr = fem.Expression(qx.sigs_e(strain_field), T0.element.interpolation_points())
sigma.interpolate(sigma_expr)
sigma.name = 'sigma'
results_xdmf_file.write_function(sigma, 0.0)

results_xdmf_file.close()


# In[ ]:


# Write stress field to .bp file

# DG1 space needed to write stress field

# T_DG1 = fem.TensorFunctionSpace(domain, ('DG', 1))
# sigma_DG1  = fem.Function(T_DG1)
# sigma_expr_DG1 = fem.Expression(qx.sigs_e(strain_field),
#                                     T_DG1.element.interpolation_points())

# sigma_DG1.interpolate(sigma_expr_DG1)

# vtx_sigma = io.VTXWriter(domain.comm, stress_field_VTX_name, [sigma_DG1._cpp_object])
# vtx_sigma.write(0)
# vtx_sigma.close()


# In[ ]:


# idx = 4
# before = ref_avg[:,idx]-exp_strn[:,idx]
# # #before = np.linalg.norm(sim_avg-exp_strn,axis=1)
# before = before[~np.isnan(before)]
# after = sim_avg[:,idx]-exp_strn[:,idx]
# # after = np.linalg.norm(sim_avg_cor-exp_strn,axis=1)
# after = after[~np.isnan(after)]

# bins = np.linspace(-0.002, 0.002, 40)
# plt.hist(before, bins, alpha=0.5, label='sim')
# plt.hist(after, bins, alpha=0.5, label='cor')
# plt.legend(loc='upper right')
# plt.xticks(np.arange(-0.002, 0.002, step=0.001))

