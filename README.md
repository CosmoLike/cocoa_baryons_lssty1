## Running Cosmolike projects <a name="running_cosmolike_projects"></a> 

In this tutorial, we assume the user installed Cocoa via the *Conda installation* method, and the name of the Conda environment is `cocoa`. We also presume the user's terminal is in the folder where Cocoa was cloned.

:one: **Step 1 of 5**: activate the Conda Cocoa environment
    
        $ conda activate cocoa

:two: **Step 2 of 5**: go to the project folder (`./cocoa/Cocoa/projects`) and clone the Cosmolike LSST-Y1 project:
    
        $(cocoa) cd ./cocoa/Cocoa/projects
        $(cocoa) $CONDA_PREFIX/bin/git clone git@github.com:CosmoLike/cocoa_lsst_y1.git lsst_y1

By convention, the Cosmolike Organization hosts a Cobaya-Cosmolike project named XXX at `CosmoLike/cocoa_XXX`. However, our scripts and YAML files assume the removal of the `cocoa_` prefix when cloning the repository.
 
:three: **Step 3 of 5**: go back to Cocoa main folder, and activate the private python environment
    
        $(cocoa) cd ../
        $(cocoa) source start_cocoa
 
Remember to run the start_cocoa script only after cloning the project repository. The script *start_cocoa* creates necessary symbolic links and also adds the *Cobaya-Cosmolike interface* of all projects to `LD_LIBRARY_PATH` and `PYTHONPATH` paths.

:four: **Step 4 of 5**: compile the project
 
        $(cocoa)(.local) source ./projects/lsst_y1/scripts/compile_lsst_y1
  
:five:  **Step 5 of 5**: select the number of OpenMP cores and run a template yaml file
    
        $(cocoa)(.local) export OMP_PROC_BIND=close; export OMP_NUM_THREADS=4
        $(cocoa)(.local) mpirun -n 1 --mca btl tcp,self --bind-to core:overload-allowed --rank-by core --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run ./projects/lsst_y1/EXAMPLE_EVALUATE1.yaml -f

PS: We offer the flag `COCOA_RUN_EVALUATE` as an alias (syntax-sugar) for `mpirun -n 1 --mca btl tcp,self --bind-to core:overload-allowed --rank-by core --map-by numa:pe=4 cobaya-run`.
