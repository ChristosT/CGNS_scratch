# PETSc-CGNS reader

Load cgns files written using the  DMPlex structure of PETSc.

# Building

## Build PETSc

Step 1: Follow the steps as provided in the website:

```
https://petsc.org/release/install/download/
```

Download through terminal:

```
git clone -b release https://gitlab.com/petsc/petsc.git petsc
```

Step 2: Configure PETSc:

```
./configure --with-cc=mpicc --with-cxx=mpicxx --with-fc=mpif90 --download-f2cblaslapack --with-cgns --download-cgns --download-hdf5 --prefix=<petsc install path>
```

If you have installed multiple mpi implementations make sure to specify the right one. For example on Ubuntu:

```
./configure --with-cc=mpicc.mpich --with-cxx=mpicxx.mpich --with-fc=mpif90.mpich ... 
```

Unless you need to debug the plugin make sure to compile Petsc without debugging  `--with-debugging=0`

* follow the instructions in the terminal and make + make install PETSc

## Build PETScCGNS reader plugin

1. Build latest ParaView with MPI enabled.
2. Build the plugin
   
   ```
   export PKG_CONFIG_PATH=<petsc install path>/lib/pkgconfig
   mkdir build
   cd build
   cmake -DParaView_DIR=<build or install directory of paraview>/lib/cmake/paraview-XX
   make 
   ```

# Using the plugin

## pvbatch

 `pvbatch script.py` with the following script:

```python
import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 13

#### import the simple module from the paraview
from paraview.simple import *
LoadPlugin("<build build/install dir>/lib/paraview-X.XX/plugins/PETScCGNS/PETScCGNS.so", remote=False, ns=globals())

reader = PETScCGNSReader(FileName="<path for file")

reader.UpdatePipeline()

dinfo = reader.GetDataInformation().DataInformation

print("# points",dinfo.GetNumberOfPoints())
print("# cells",dinfo.GetNumberOfCells())
```

## ParaView

1. In ParaView go to Tools>Manage Plugins>Load New and pick  `<plugin build/install dir>/lib/paraview-X.XX/plugins/PETScCGNS/PETScCGNS.so`
2. Open a new CGNS file and pick PETScCGNSReader.
