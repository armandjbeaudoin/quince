# quince
Post-processor for grain-average data with consideration of equilibrium and incompatibility

**quince** is an anagram of the first three letters from *equ*ilibrium and *inc*ompatibility.

## Procedure

* Generate files with coordinates and orientation matrices
  * files for the single measurement (#20) of the Ti7Al creep are available in data subdirectory
  * an example  spyder script to generate the files from hexrd grains.out is also in the data subdirectory 
  * _hexrd3 is required_
* Generate the mesh using neper
  * tesselation (using center of mass positions in data/coords_0020.dat)

        neper -T -n 716 -morphooptiini "coo:file(data/coords_0020.dat)" -dim 3 -domain "cube(1.0,1.0,1.0):translate(-0.5,-0.5,-0.5)" -morpho voronoi -ori uniform -oricrysym hexagonal -regularization 1  -format tess -o mesh/creep_0020.tess
  * mesh

        neper -M -rcl 0.7 mesh/creep_0020.tess -o mesh/creep_0020.xml
  * convert the mesh (in your fenics enviroment)

        dolfin-convert mesh/creep_0020.msh mesh/creep_0020.xml
* Run model
  * jupyter notebook is notebooks/eq_inc.ipynb
  * a python script, derived from the notebook, is in the src directory
  * the output **xdmf** files will be placed in the results directory, and may be visualized using paraview
  
